"""Analyze disjoint combined-endpoint validation."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from .paired_role_analysis import bca_mean_interval
from .shard_index import sha256_file
ARMS=("identity-gptq","down-h16-gptq","combined-identity-k4")
def gm(v):return math.exp(np.log(np.asarray(v)).mean())
def main():
 p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--input",type=Path,action="append",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();
 if a.output.exists():raise FileExistsError(a.output)
 ps=sha256_file(a.plan);rows=[];ins=[]
 for f in a.input:
  x=json.loads(f.read_text());
  if x["plan"]["sha256"]!=ps or x["protected_roles_opened"]:raise ValueError("bad input")
  rows+=x["rows"];ins.append({"path":str(f),"sha256":sha256_file(f)})
 by={}
 for r in rows:by.setdefault(r["expert"],{})[r["arm"]]=r["full_expert_nmse"]
 plan=json.loads(a.plan.read_text());ex=sorted(by)
 if ex!=sorted(plan["experts"]) or any(set(v)!=set(ARMS) for v in by.values()):raise ValueError("coverage")
 vals={arm:[by[e][arm] for e in ex] for arm in ARMS};c=np.asarray(vals[ARMS[-1]]);rng=np.random.default_rng(plan["bootstrap"]["seed"]);idx=rng.integers(0,len(ex),size=(plan["bootstrap"]["replicates"],len(ex)));comps={};passed=True
 for ctl,thr in (("identity-gptq",.10),("down-h16-gptq",.03)):
  b=np.asarray(vals[ctl]);lr=np.log(c/b);ci=bca_mean_interval(lr,lr[idx].mean(1));imp=1-gm(c)/gm(b);wins=int((c<b).sum());comps[ctl]={"relative_improvement":imp,"wins":wins,"mean_log_ratio_ci95_bca":ci};passed &= imp>=thr and wins>=12 and ci[1]<0
 out={"schema":"glm53-combined-identity-k4-validation-analysis.v1","plan":{"path":str(a.plan),"sha256":ps},"inputs":ins,"experts":ex,"geometric_mean_nmse":{k:gm(v) for k,v in vals.items()},"comparisons":comps,"decision":"pass-build-full-layer-combined" if passed else "fail-do-not-build","protected_roles_opened":[]};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__":main()
