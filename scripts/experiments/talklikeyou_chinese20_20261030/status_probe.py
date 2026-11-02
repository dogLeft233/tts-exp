from pathlib import Path
import json,subprocess,time
r=Path('/home/data2/wangjiajun/av-evaluation/jobs/talklikeyou_chinese20_rms24_20261030_attempt01')
x={'time':time.time()}
for n in ['generation_status.json','generation_complete.json','status.json','generation_failure.json','failure.json','finish_status.json','finish_failure.json']:
 p=r/n
 if p.exists():
  v=json.loads(p.read_text());x[n]={k:v[k] for k in ['status','stage','completed','actual_unique_forwards','error','updated'] if k in v}
x['completed_scorers']=[p.name.removesuffix('_complete.json') for p in (r/'stages').glob('*_complete.json')]
x['syncnet_scored']=len(list((r/'syncnet').glob('*_distances.npy')))
for stage in ['avs_u','mtd','vocalist','synchformer','sparsesync']:
 p=r/stage/'batch_scores.json'
 if p.exists():x[stage+'_items']=len(json.loads(p.read_text()))
x['cogenav_embeddings']=len(list((r/'cogenav/embeddings').glob('*.npz')))
p=r/'syncreward/scores.jsonl'
if p.exists():x['syncreward_items']=len(p.read_text().splitlines())
p=r/'logs/syncreward.log'
if p.exists():x['syncreward_completed_forwards']=sum(line.startswith('{"video_path":') for line in p.read_text(errors='replace').splitlines())
print(json.dumps(x))
