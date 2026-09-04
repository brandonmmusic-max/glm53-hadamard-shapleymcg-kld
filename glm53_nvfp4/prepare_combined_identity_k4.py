"""Freeze disjoint validation of optimized gate/up plus down-H16 endpoints."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from .shard_index import sha256_file
EXPERTS=[3,19,35,51,67,83,99,115,131,147,163,179,195,211,227,243]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--gateup-analysis",type=Path,required=True); p.add_argument("--down-analysis",type=Path,required=True); p.add_argument("--source-index",type=Path,required=True); p.add_argument("--capture-manifest",type=Path,required=True); p.add_argument("--roles",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 g=json.loads(a.gateup_analysis.read_text()); d=json.loads(a.down_analysis.read_text())
 if g["decision"]!="pass-validate" or d["decision"]!="pass-build-full-layer-identity-k4": raise ValueError("component gates did not pass")
 out={"schema":"glm53-combined-identity-k4-validation-plan.v1","created_at":datetime.now(timezone.utc).isoformat(),"role":"disjoint-expert fit-only validation; no teacher logits","layer":3,"experts":EXPERTS,"sampling":{"strategy":"domain-balanced","fit":{"offset":0,"count":256},"evaluation":{"offset":1792,"count":128}},"candidate":{"gate_up":"unrotated lossless identity-K4 endpoint optimization","down":"fixed H16 plus lossless identity-K4 endpoint optimization","gate_up_ridge_ratio":g["selected_ridge_ratio"],"down_ridge_ratio":3.0,"sweeps":1,"stored_bpw":4.5,"runtime":"direct native NVFP4 endpoints plus down-input H16","runtime_table_bytes":0,"ldlq":False},"controls":["identity GPTQ all projections","down-H16 GPTQ with identity GPTQ gate/up"],"bootstrap":{"unit":"expert","replicates":20000,"seed":2026090422},"decision_rule":"promote only if combined endpoint improves geometric-mean routed output NMSE at least 10 percent versus identity GPTQ and 3 percent versus down-H16 GPTQ, wins at least 12/16 versus each, and both paired BCa upper bounds are below zero","protected_boundary":"teacher-logit roles remain unopened","inputs":{"gateup_analysis":{"path":str(a.gateup_analysis),"sha256":sha256_file(a.gateup_analysis)},"down_analysis":{"path":str(a.down_analysis),"sha256":sha256_file(a.down_analysis)},"source_index":{"path":str(a.source_index),"sha256":sha256_file(a.source_index)},"capture_manifest":{"path":str(a.capture_manifest),"sha256":sha256_file(a.capture_manifest)},"roles":{"path":str(a.roles),"sha256":sha256_file(a.roles)}}}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(sha256_file(a.output))
if __name__=="__main__": main()
