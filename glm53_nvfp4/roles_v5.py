"""Build the final fresh 64-window conditional-fit selection role."""
from __future__ import annotations

import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from .roles import sha256_file

SALT="glm53-nvfp4-v5-down-h16-powered-selection"
DOMAINS=("axis1_general","axis2_legal","axis3_code_agentic","axis4_reasoning_termination")
def rank(item): return hashlib.sha256(f"{SALT}:{item['window_id']}:{item['token_ids_sha256']}".encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--panel",type=Path,required=True);p.add_argument("--arrays",type=Path,required=True);p.add_argument("--v4-roles",type=Path,required=True);p.add_argument("--codec-roles",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 panel=json.loads(a.panel.read_text());v4=json.loads(a.v4_roles.read_text());codec=json.loads(a.codec_roles.read_text());used=set()
 for prior in (v4,codec):
  for role,items in prior["roles"].items():
   if role!="fit":used.update(x["id"] for x in items)
 pool=[x for x in panel["windows"] if x["role"]=="conditional-fit" and x["window_id"] not in used]
 if len(pool)!=64:raise ValueError(f"expected exactly 64 never-used conditional-fit windows, found {len(pool)}")
 counts=Counter(x["domain"] for x in pool)
 expected_remaining=Counter({"axis1_general":22,"axis2_legal":21,"axis3_code_agentic":21})
 if counts!=expected_remaining:raise ValueError(f"unexpected remaining-domain inventory: {counts}")
 def entry(x,role):
  token=a.arrays/f"{x['window_id']}.tokens.npy"
  if sha256_file(token)!=x["token_ids_sha256"]:raise ValueError(f"token mismatch {x['window_id']}")
  return {"id":x["window_id"],"domain":x["domain"],"input_sha256":x["token_ids_sha256"],"token_path":str(token),"teacher_path":f"logits/full-panel/{x['role']}/{x['window_id']}.safetensors","teacher_source_role":x["role"],"experimental_role":role,"prediction_positions":x["prediction_positions"],"rank_sha256":rank(x)}
 by_domain={d:sorted((x for x in pool if x["domain"]==d),key=rank) for d in expected_remaining}
 chosen=[x for d in sorted(by_domain) for x in by_domain[d][:21]]
 unused=[x["window_id"] for d in by_domain for x in by_domain[d][21:]]
 selection=[entry(x,"selection") for x in sorted(chosen,key=rank)]
 out={"schema":"glm53-nvfp4-v5.roles.v1","selection_salt":SALT,"selection_rule":"21 lowest salted hashes in each represented domain among source conditional-fit windows absent from V4 non-fit roles and the codec conditional-fit32 role","source_panel":str(a.panel),"source_panel_sha256":sha256_file(a.panel),"prior_roles":[{"path":str(x),"sha256":sha256_file(x)} for x in (a.v4_roles,a.codec_roles)],"remaining_inventory":dict(counts),"unused_remaining_ids":unused,"roles":{"fit":v4["roles"]["fit"],"conditional-fit":[],"selection":selection,"confirmation":v4["roles"]["confirmation"],"final":[]},"counts":{"fit":len(v4["roles"]["fit"]),"conditional-fit":0,"selection":63,"confirmation":len(v4["roles"]["confirmation"]),"final":0},"domains":{"selection":dict(Counter(x["domain"] for x in chosen))},"selection_waves":{"1":[x["id"] for x in selection]},"protection_boundary":"28 confirmation logits remain unopened; one extra general conditional-fit window remains unused"}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"output":str(a.output),"sha256":sha256_file(a.output),"counts":out["counts"],"domains":out["domains"]},sort_keys=True))
if __name__=="__main__":main()
