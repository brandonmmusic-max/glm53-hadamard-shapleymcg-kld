from pathlib import Path
import json,hashlib,math
import numpy as np
from scipy.stats import bootstrap
P=Path(__file__).resolve().parent
ROOTS={'trellismx':Path('/media/brandonmusic/nvme1n1p3/trellismx-reference-kld-20260909'),'tr3':Path('/media/brandonmusic/nvme1n1p3/glm53-tr3-kld-20260909/attempt2')}
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
plan={'type':'Post-hoc descriptive analysis of retained already-opened conditional-fit scores; not new qualification','windows':32,'rows_per_window':2046,'row_mask':'Exclude prediction row0; use1..2046','metrics':['top1 equality with teacher','cross-model top1 equality','teacher-confidence subsets pmax>=0.5 and pmax>=0.9','original-text mean negative log likelihood and perplexity','mean student NLL minus teacher NLL'],'aggregation':'Equal window mean for all-row metrics; pooled eligible row count for confidence subsets; confidence thresholds selected before this script inspected score values','uncertainty':'Paired window BCa95 intervals for TrellisMX-minus-TR3 top1 agreement and excess NLL,20000resamples,seed20260902; not server-run variability','limits':['Conditional teacher-forced histories, not free generation','Runtime/EP/activation/nonrouted-weight confounds retained','Teacher is BF16 and can itself be incorrect','No top-k ranks/probabilities retained; no fabricated top-k results']}
save(P/'analysis-plan.json',plan)
inputs={k:json.loads((r/'verified-inputs.json').read_text()) for k,r in ROOTS.items()};assert inputs['trellismx']['windows']==inputs['tr3']['windows']
windows=inputs['tr3']['windows'];assert len(windows)==32
out={'plan_sha256':sha(P/'analysis-plan.json'),'script_sha256':sha(Path(__file__)),'windows':32,'rows':32*2046,'arms':{},'paired':{},'input_receipts':[]}
for cache in ['fp8','nvfp4']:
 data={};per={}
 for model,root in ROOTS.items():
  arrays=[];per[model]=[]
  for w in windows:
   f=root/cache/'scores'/f"{w['id']}.npz";receipt=json.loads(f.with_suffix('.json').read_text());assert sha(f)==receipt['score_sha256'];assert receipt['true_decode_rows']==2046
   assert receipt['teacher_sha256']==w['teacher_sha256'] and receipt['token_sha256']==w['token_values_sha256']
   with np.load(f,allow_pickle=False) as z:a={k:z[k][1:].copy() for k in z.files}
   assert all(v.shape==(2046,) for v in a.values())
   agree=a['teacher_top1']==a['student_top1'];teacher_nll=-a['teacher_logp_realized'];student_nll=-a['student_logp_realized']
   per[model].append({'window':w['id'],'top1_agreement':float(agree.mean()),'teacher_nll':float(teacher_nll.mean()),'student_nll':float(student_nll.mean()),'excess_nll':float((student_nll-teacher_nll).mean())})
   arrays.append(a);out['input_receipts'].append({'model':model,'cache':cache,'window':w['id'],'score_sha256':receipt['score_sha256']})
  allrows={key:np.concatenate([a[key] for a in arrays]) for key in arrays[0]};data[model]=allrows
  confidence={}
  for threshold in [0.5,0.9]:
   mask=allrows['teacher_top1_p']>=threshold;n=int(mask.sum());matches=int((allrows['teacher_top1'][mask]==allrows['student_top1'][mask]).sum())
   confidence[str(threshold)]={'eligible_rows':n,'agreement_count':matches,'agreement':matches/n if n else None}
  nll=float(np.mean([r['student_nll'] for r in per[model]]));tnll=float(np.mean([r['teacher_nll'] for r in per[model]]))
  out['arms'][model+'_'+cache]={'teacher_top1_agreement':float(np.mean([r['top1_agreement'] for r in per[model]])),'agreement_count':int((allrows['teacher_top1']==allrows['student_top1']).sum()),'rows':len(allrows['kld']),'teacher_confidence_subsets':confidence,'student_nll':nll,'teacher_nll':tnll,'excess_nll':float(np.mean([r['excess_nll'] for r in per[model]])),'student_perplexity':math.exp(nll),'teacher_perplexity':math.exp(tnll),'per_window':per[model]}
 for k in ['teacher_top1','teacher_top1_p','teacher_logp_realized','realized_token','teacher_entropy']:assert np.array_equal(data['trellismx'][k],data['tr3'][k]),k
 a=data['trellismx'];b=data['tr3'];am=a['student_top1']==a['teacher_top1'];bm=b['student_top1']==b['teacher_top1']
 pair={'cross_model_top1_agreement':float((a['student_top1']==b['student_top1']).mean()),'teacher_wins_count':{'both_match':int((am&bm).sum()),'trellismx_only':int((am&~bm).sum()),'tr3_only':int((~am&bm).sum()),'neither_matches':int((~am&~bm).sum())}}
 for metric in ['top1_agreement','excess_nll']:
  delta=np.array([x[metric]-y[metric] for x,y in zip(per['trellismx'],per['tr3'])]);ci=bootstrap((delta,),np.mean,method='BCa',n_resamples=20000,random_state=20260902)
  pair['trellismx_minus_tr3_'+metric]={'mean':float(delta.mean()),'window_bca95':[float(ci.confidence_interval.low),float(ci.confidence_interval.high)]}
 out['paired'][cache]=pair
save(P/'results.json',out)
for name,row in out['arms'].items():print(name,json.dumps({k:v for k,v in row.items() if k!='per_window'}))
print('paired',json.dumps(out['paired']))
