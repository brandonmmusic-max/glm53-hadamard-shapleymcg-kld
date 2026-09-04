"""Tune output-aware native endpoints on unrotated gate/up tensors."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from .block_gptq import full_gptq_quantize, refine_global_scale
from .capture import LayerCapture
from .endpoint_codec import optimize_identity_k4_endpoint, pack_identity_k4_stream
from .modelopt import PackedNVFP4, dequantize
from .output_aware import route_weighted_hessian
from .screen_blocklocal_h16 import _full_nmse, _middle
from .shard_index import IndexedCheckpoint, sha256_file


@torch.no_grad()
def main() -> None:
    p=argparse.ArgumentParser()
    for name in ("plan","source","source-index","capture-root","roles","output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--expert-start-index", type=int, required=True)
    p.add_argument("--expert-end-index", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    plan=json.loads(a.plan.read_text()); experts=plan["experts"][a.expert_start_index:a.expert_end_index]
    device=torch.device(a.device); torch.cuda.set_device(device)
    source=IndexedCheckpoint(a.source,a.source_index)
    fit=plan["sampling"]["fit"]; ev=plan["sampling"]["evaluation"]
    fitcap=LayerCapture(a.capture_root,3,a.roles,max_samples=fit["count"],sample_offset=fit["offset"],sampling_strategy="domain-balanced")
    evalcap=LayerCapture(a.capture_root,3,a.roles,max_samples=ev["count"],sample_offset=ev["offset"],sampling_strategy="domain-balanced")
    prefix=source.expert_prefix(3,experts[0]).split("layers.3.")[0]
    rows=[]; started=time.time()
    for expert in experts:
        x0,r0=fitcap.samples(expert); x1,r1=evalcap.samples(expert)
        x0,r0,x1,r1=x0.to(device),r0.to(device),x1.to(device),r1.to(device)
        base=f"{prefix}layers.3.mlp.experts.{expert}"
        gate=source.get(f"{base}.gate_proj.weight").to(device).float(); up=source.get(f"{base}.up_proj.weight").to(device).float(); down=source.get(f"{base}.down_proj.weight").to(device).float()
        h=route_weighted_hessian(x0,r0); stacked=torch.cat((gate,up)); gs=refine_global_scale(gate,up,search_grid=8,iterations=2)
        initial=full_gptq_quantize(stacked,h,global_scale=gs,search_grid=8)
        split=gate.shape[0]
        def halves(endpoint):
            return dequantize(PackedNVFP4(endpoint.weight[:split],endpoint.weight_scale[:split],endpoint.weight_scale_2)).to(device), dequantize(PackedNVFP4(endpoint.weight[split:],endpoint.weight_scale[split:],endpoint.weight_scale_2)).to(device)
        qgate,qup=halves(initial)
        reference=F.linear(_middle(x1,gate,up),down)
        control=F.linear(_middle(x1,qgate,qup),down)
        rows.append({"expert":expert,"ridge_ratio":None,"arm":"gptq","full_expert_nmse":_full_nmse(reference,control,r1)})
        for ridge in plan["candidate"]["ridge_ratio_grid"]:
            endpoint,history=optimize_identity_k4_endpoint(stacked,h,initial,sweeps=1,ridge_ratio=ridge)
            stream=pack_identity_k4_stream(endpoint); cg,cu=halves(endpoint)
            actual=F.linear(_middle(x1,cg,cu),down)
            rows.append({"expert":expert,"ridge_ratio":ridge,"arm":"identity-k4","full_expert_nmse":_full_nmse(reference,actual,r1),"codec":{"trellis_bpw":stream.trellis_bpw,"scale_bpw":stream.scale_bpw,"bit_exact":True,"history":[z.to_dict() for z in history]}})
        print(json.dumps({"expert":expert,"completed":True}),flush=True); torch.cuda.empty_cache()
    payload={"schema":"glm53-gateup-identity-k4-tuning-raw.v1","plan":{"path":str(a.plan),"sha256":sha256_file(a.plan)},"expert_slice":[a.expert_start_index,a.expert_end_index],"rows":rows,"elapsed_seconds":time.time()-started,"protected_roles_opened":[]}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(a.output),"sha256":sha256_file(a.output)},sort_keys=True))
if __name__ == "__main__": main()
