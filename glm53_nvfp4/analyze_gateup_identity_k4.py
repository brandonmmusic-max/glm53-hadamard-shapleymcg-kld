"""Select a gate/up identity-K4 ridge."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from .shard_index import sha256_file
def gm(v): return math.exp(sum(math.log(x) for x in v)/len(v))
def main():
 p=argparse.ArgumentParser(); p.add_argument("--plan",type=Path,required=True); p.add_argument("--input",type=Path,action="append",required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 ps=sha256_file(a.plan); rows=[]; inputs=[]
 for path in a.input:
  raw=json.loads(path.read_text())
  if raw["plan"]["sha256"]!=ps or raw["protected_roles_opened"]: raise ValueError("bad input")
  rows+=raw["rows"]; inputs.append({"path":str(path),"sha256":sha256_file(path)})
 controls={r["expert"]:r["full_expert_nmse"] for r in rows if r["arm"]=="gptq"}; cand={}
 for r in rows:
  if r["arm"]=="identity-k4": cand.setdefault(str(r["ridge_ratio"]),{})[r["expert"]]=r["full_expert_nmse"]
 experts=sorted(controls); cg=gm([controls[e] for e in experts]); grid={}
 for ridge,v in cand.items():
  g=gm([v[e] for e in experts]); grid[ridge]={"geometric_mean_nmse":g,"relative_improvement":1-g/cg,"wins":sum(v[e]<controls[e] for e in experts)}
 selected=min(grid,key=lambda x:grid[x]["geometric_mean_nmse"]); passed=grid[selected]["relative_improvement"]>=.03 and grid[selected]["wins"]>=6
 out={"schema":"glm53-gateup-identity-k4-tuning-analysis.v1","plan":{"path":str(a.plan),"sha256":ps},"inputs":inputs,"experts":experts,"control_geometric_mean_nmse":cg,"grid":grid,"selected_ridge_ratio":float(selected),"decision":"pass-validate" if passed else "fail-stop-gateup-endpoint","protected_roles_opened":[]}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__": main()
