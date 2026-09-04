"""Freeze the powered V5 down-H16 KLD test before teacher download."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from .shard_index import sha256_file
def main():
 p=argparse.ArgumentParser();p.add_argument("--roles",type=Path,required=True);p.add_argument("--candidate-index",type=Path,required=True);p.add_argument("--stock-index",type=Path,required=True);p.add_argument("--validation",type=Path,required=True);p.add_argument("--runner",type=Path,required=True);p.add_argument("--runtime-manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 roles=json.loads(a.roles.read_text());
 if len(roles["roles"]["selection"])!=63:raise ValueError("V5 must contain 63 selection windows")
 out={"schema":"glm53-down-h16-v5-powered-kld-freeze.v1","created_at":datetime.now(timezone.utc).isoformat(),"role":"selection","selection_wave":1,"windows":63,"arms":{"stock":{"rotation":"identity","scope":"gate-up","layers":"3"},"candidate":{"rotation":"had16","scope":"mid-only","layers":"3"}},"estimand":"paired per-window candidate minus stock teacher KLD and relative geometric-mean improvement","bootstrap":{"method":"BCa","unit":"window","replicates":50000,"seed":2026090423},"decision_rule":"provisionally qualify down-only H16 as beating stock NVFP4 on the represented domains only if relative geometric-mean KLD improvement is at least 3 percent, the paired BCa upper bound for candidate-minus-stock mean KLD is below zero, and all three represented domain mean deltas are negative; four-domain confirmation remains required","represented_domains":["axis1_general","axis2_legal","axis3_code_agentic"],"runtime":{"tp":4,"ep":4,"dcp":4,"mtp":0,"moe_backend":"humming","activation_dtype":"bf16","load_format":"instanttensor"},"confirmation_boundary":"do not open the 28 confirmation logits; they are reserved for the eventual 6-bpw allocation and four-domain confirmation","inputs":{k:{"path":str(v),"bytes":v.stat().st_size,"sha256":sha256_file(v)} for k,v in {"roles":a.roles,"candidate_index":a.candidate_index,"stock_index":a.stock_index,"validation":a.validation,"runner":a.runner,"runtime_manifest":a.runtime_manifest}.items()}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(sha256_file(a.output))
if __name__=="__main__":main()
