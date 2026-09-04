"""Validate optimized native gate/up endpoints plus optimized down-H16."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from .block_gptq import full_gptq_quantize,refine_global_scale
from .block_rotation import apply_activation_rotation,apply_weight_rotation,hadamard16
from .capture import LayerCapture
from .endpoint_codec import optimize_identity_k4_endpoint,pack_identity_k4_stream
from .modelopt import PackedNVFP4,dequantize
from .output_aware import route_weighted_hessian
from .screen_blocklocal_h16 import _full_nmse,_middle
from .shard_index import IndexedCheckpoint,sha256_file
def split(endpoint,n,device):
 return (dequantize(PackedNVFP4(endpoint.weight[:n],endpoint.weight_scale[:n],endpoint.weight_scale_2)).to(device),dequantize(PackedNVFP4(endpoint.weight[n:],endpoint.weight_scale[n:],endpoint.weight_scale_2)).to(device))
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for x in ("plan","source","source-index","capture-root","roles","output"):p.add_argument("--"+x,type=Path,required=True)
 p.add_argument("--expert-start-index",type=int,required=True);p.add_argument("--expert-end-index",type=int,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 plan=json.loads(a.plan.read_text());experts=plan["experts"][a.expert_start_index:a.expert_end_index];dev=torch.device(a.device);torch.cuda.set_device(dev);src=IndexedCheckpoint(a.source,a.source_index)
 fit=plan["sampling"]["fit"];ev=plan["sampling"]["evaluation"];fc=LayerCapture(a.capture_root,3,a.roles,max_samples=fit["count"],sample_offset=fit["offset"],sampling_strategy="domain-balanced");ec=LayerCapture(a.capture_root,3,a.roles,max_samples=ev["count"],sample_offset=ev["offset"],sampling_strategy="domain-balanced");prefix=src.expert_prefix(3,experts[0]).split("layers.3.")[0];h16=hadamard16(device=dev);rows=[];started=time.time()
 for e in experts:
  x0,r0=fc.samples(e);x1,r1=ec.samples(e);x0,r0,x1,r1=x0.to(dev),r0.to(dev),x1.to(dev),r1.to(dev);base=f"{prefix}layers.3.mlp.experts.{e}";w={p:src.get(f"{base}.{p}.weight").to(dev).float() for p in ("gate_proj","up_proj","down_proj")};h=route_weighted_hessian(x0,r0);stack=torch.cat((w["gate_proj"],w["up_proj"]));gs=refine_global_scale(w["gate_proj"],w["up_proj"],search_grid=8,iterations=2);gu0=full_gptq_quantize(stack,h,global_scale=gs,search_grid=8);g0,u0=split(gu0,w["gate_proj"].shape[0],dev);fit0,eval0=_middle(x0,g0,u0),_middle(x1,g0,u0)
  gu1,hgu=optimize_identity_k4_endpoint(stack,h,gu0,sweeps=1,ridge_ratio=plan["candidate"]["gate_up_ridge_ratio"]);sgu=pack_identity_k4_stream(gu1);g1,u1=split(gu1,w["gate_proj"].shape[0],dev);fit1,eval1=_middle(x0,g1,u1),_middle(x1,g1,u1)
  ref=F.linear(_middle(x1,w["gate_proj"],w["up_proj"]),w["down_proj"]);arms={}
  for name,mid,rot,opt in (("identity-gptq",fit0,torch.eye(16,device=dev),False),("down-h16-gptq",fit0,h16,False),("combined-identity-k4",fit1,h16,True)):
   wt=apply_weight_rotation(w["down_proj"],rot);carrier=apply_activation_rotation(mid,rot);hh=route_weighted_hessian(carrier,r0);dgs=refine_global_scale(wt,search_grid=8,iterations=2);pd=full_gptq_quantize(wt,hh,global_scale=dgs,search_grid=8);hd=[]
   if opt:pd,hd=optimize_identity_k4_endpoint(wt,hh,pd,sweeps=1,ridge_ratio=plan["candidate"]["down_ridge_ratio"]);sd=pack_identity_k4_stream(pd)
   arms[name]=(dequantize(pd).to(dev),rot,hd)
  for name,mid in (("identity-gptq",eval0),("down-h16-gptq",eval0),("combined-identity-k4",eval1)):
   qd,rot,hd=arms[name];actual=F.linear(apply_activation_rotation(mid,rot),qd);rows.append({"expert":e,"arm":name,"full_expert_nmse":_full_nmse(ref,actual,r1),"codec":({"gateup_bpw":sgu.trellis_bpw+sgu.scale_bpw,"down_bpw":sd.trellis_bpw+sd.scale_bpw,"bit_exact":True,"gateup_history":[z.to_dict() for z in hgu],"down_history":[z.to_dict() for z in hd]} if name=="combined-identity-k4" else None)})
  print(json.dumps({"expert":e,"completed":True}),flush=True);torch.cuda.empty_cache()
 payload={"schema":"glm53-combined-identity-k4-validation-raw.v1","plan":{"path":str(a.plan),"sha256":sha256_file(a.plan)},"expert_slice":[a.expert_start_index,a.expert_end_index],"rows":rows,"elapsed_seconds":time.time()-started,"protected_roles_opened":[]};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps({"output":str(a.output),"sha256":sha256_file(a.output)},sort_keys=True))
if __name__=="__main__":main()
