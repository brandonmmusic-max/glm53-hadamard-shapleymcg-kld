from pathlib import Path
import json, hashlib,itertools
import numpy as np
from scipy.special import xlogy
P=Path(__file__).resolve().parent
roots={'MX':Path('/media/brandonmusic/nvme1n1p3/trellismx-reference-kld-20260909'),'TR3':Path('/media/brandonmusic/nvme1n1p3/glm53-tr3-kld-20260909/attempt2')}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(P/n).write_text(json.dumps(x,indent=2)+'\n')
save('plan.json',{'status':'Exploratory post-hoc, already-opened CF32; no model execution','mask':'prediction rows 1..2046; equal 32 window means','metrics':['retained full teacher-to-student KL mean and token quantiles','top1 agreement, actual-token top1 hit rate','mean NLL and perplexity','actual-token probability absolute error','six student-pair top1 agreements','all ten teacher/student pairs: actual-next-token versus rest KL both directions and JS','student cache changes: teacher top1 retained/lost/gained'],'limits':'Two-bucket KL is a lower bound on full KL, not its estimate. One actual next token is not an answer correctness label. No independent repetitions, no new qualification, no multiplicity-adjusted claims.'})
D={};receipts=[];wins=None;inventory=[]
for model,root in roots.items():
 ws=json.loads((root/'verified-inputs.json').read_text())['windows']
 if wins is None:wins=ws
 assert ws==wins
 for cache in ['fp8','nvfp4']:
  arr=[]
  for w in ws:
   p=root/cache/'scores'/f"{w['id']}.npz";r=json.loads(p.with_suffix('.json').read_text())
   assert sha(p)==r['score_sha256'] and r['teacher_sha256']==w['teacher_sha256'] and r['token_sha256']==w['token_values_sha256'] and r['true_decode_rows']==2046
   with np.load(p) as z:a={k:np.asarray(z[k][1:],dtype=np.float64 if z[k].dtype.kind=='f' else z[k].dtype) for k in z.files}
   assert all(v.shape==(2046,) for v in a.values());arr.append(a)
   receipts.append({'path':str(p),'sha256':r['score_sha256']})
  D[model+'_'+cache]={k:np.stack([a[k] for a in arr]) for k in arr[0]}
  inventory.extend({'path':str(p),'bytes':p.stat().st_size} for p in (root/cache).rglob('*') if p.is_file() and ('logit' in p.name or p.suffix in ['.f32','.npy']))
base=next(iter(D.values()))
for a in D.values():
 for k in ['teacher_top1','teacher_top1_p','teacher_entropy','realized_token','teacher_logp_realized']:assert np.array_equal(a[k],base[k]),k
out={'rows':65472,'windows':32,'arms':{},'pairs':{},'cache_changes':{},'inputs':receipts,'raw_inventory':inventory,'script_sha256':sha(Path(__file__)),'plan_sha256':sha(P/'plan.json')}
probs={'BF16':np.exp(base['teacher_logp_realized'])}
def summary(v):return {'mean':float(v.mean()),'per_window':v.mean(axis=1).tolist()}
def kl(p,q):return xlogy(p,p)-xlogy(p,q)+xlogy(1-p,1-p)-xlogy(1-p,1-q)
# Verify binary KL implementation on known controls.
assert np.allclose(kl(np.array([.2,.5]),np.array([.2,.5])),0)
assert np.allclose(kl(np.array([.5]),np.array([.25])),.5*np.log(4/3))
for name,a in D.items():
 p=np.exp(a['student_logp_realized']);assert np.all((p>0)&(p<1));probs[name]=p
 agree=a['teacher_top1']==a['student_top1'];actual=a['student_top1']==a['realized_token']
 out['arms'][name]={'full_teacher_KL':summary(a['kld']),'KL_token_quantiles':dict(zip(['p50','p90','p95','p99','max'],map(float,np.quantile(a['kld'],[.5,.9,.95,.99,1])))),'teacher_top1_agreement':summary(agree),'actual_token_top1_hit':summary(actual),'NLL':summary(-a['student_logp_realized']),'perplexity':float(np.exp(-a['student_logp_realized'].mean())),'actual_token_probability_MAE_vs_teacher':summary(abs(p-probs['BF16'])),'high_confidence_teacher_matches':int((agree & (a['teacher_top1_p']>=.9)).sum())}
for n,m in itertools.combinations(probs,2):
 p,q=probs[n],probs[m];mid=(p+q)/2
 v={'KL_A_to_B_two_bucket':summary(kl(p,q)),'KL_B_to_A_two_bucket':summary(kl(q,p)),'JS_two_bucket':summary((kl(p,mid)+kl(q,mid))/2),'actual_token_probability_MAE':summary(abs(p-q))}
 if n in D and m in D:v['top1_agreement']=summary(D[n]['student_top1']==D[m]['student_top1'])
 if n=='BF16':assert np.all(kl(p,q)<=D[m]['kld']+1e-6)
 out['pairs'][n+' vs '+m]=v
for model in roots:
 a,b=D[model+'_fp8'],D[model+'_nvfp4'];am=a['student_top1']==a['teacher_top1'];bm=b['student_top1']==b['teacher_top1']
 out['cache_changes'][model]={'top1_flips':int((a['student_top1']!=b['student_top1']).sum()),'teacher_match_lost':int((am&~bm).sum()),'teacher_match_gained':int((~am&bm).sum()),'full_teacher_KL_increase':summary(b['kld']-a['kld'])}
out['teacher']={'actual_token_top1_hit':float((base['teacher_top1']==base['realized_token']).mean()),'perplexity':float(np.exp(-base['teacher_logp_realized'].mean()))}
save('results.json',out)
print(json.dumps({'arms':{k:{m:(v['mean'] if isinstance(v,dict) and 'mean' in v else v) for m,v in a.items()} for k,a in out['arms'].items()},'pairs':{k:{m:v['mean'] for m,v in a.items()} for k,a in out['pairs'].items()},'cache_changes':out['cache_changes'],'raw_inventory':inventory},indent=2))
