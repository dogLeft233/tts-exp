from pathlib import Path
import json,numpy as np,torch
r=Path('/home/data2/wangjiajun/av-evaluation/jobs/talklikeyou_chinese20_rms24_20261030_attempt01');checks=[]
for model in ['synchformer','sparsesync']:
 for row in json.loads((r/model/'batch_scores.json').read_text()):
  if row['status']!='PASS':continue
  result=json.loads((r/model/row['id']/'deployment_validation.json').read_text())
  for shift,v in result['results'].items():
   z=np.asarray(v['logits'],dtype=np.float64);p=np.exp(z-z.max());p/=p.sum()
   cpu=torch.tensor([v['logits']],dtype=torch.float32).softmax(-1)[0].numpy()
   gpu=torch.tensor([v['logits']],dtype=torch.float32,device='cuda').softmax(-1)[0].cpu().numpy()
   expected=np.array(v['probabilities'])
   checks.append(dict(model=model,id=row['id'],shift=shift,zero_saved=v['p_zero'],fp64_zero_error=float(abs(p[10]-v['p_zero'])),cpu32_zero_error=float(abs(float(cpu[10])-v['p_zero'])),gpu32_zero_error=float(abs(float(gpu[10])-v['p_zero'])),gpu32_max_array_error=float(abs(gpu-expected).max()),fp64_max_array_error=float(abs(p-expected).max())))
worst=sorted(checks,key=lambda x:x['fp64_zero_error'],reverse=True)
result=dict(arrays=len(checks),max_fp64_zero_error=max(x['fp64_zero_error'] for x in checks),max_gpu32_zero_error=max(x['gpu32_zero_error'] for x in checks),max_gpu32_array_error=max(x['gpu32_max_array_error'] for x in checks),worst=worst[:5],checks=checks)
(r/'probability_precision_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
