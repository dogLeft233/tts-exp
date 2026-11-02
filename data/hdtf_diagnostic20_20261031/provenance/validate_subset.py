from pathlib import Path
import json,hashlib,subprocess,wave,csv,html
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import imageio_ffmpeg
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'manifest.json').read_text());assert m['status']=='COMPLETE'
assert len(m['originals'])==20 and len(m['records'])==40
assert len({x['speaker'] for x in m['originals']})==20
ff=imageio_ffmpeg.get_ffmpeg_exe();rows=[]
source_timing=[]
excluded_sources=[]
selected_paths={x['original'] for x in m['originals']}
for p in (root/'original_videos').glob('*.mp4'):
 if str(p.relative_to(root)) not in selected_paths:
  excluded_sources.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
for x in m['originals']:
 assert hashlib.sha256((root/x['original']).read_bytes()).hexdigest()==x['sha256']
 span=x['official_time_line'].split()[1+x['official_clip_index']]
 def seconds(t):
  v=[float(z) for z in t.split(':')];return sum(z*60**i for i,z in enumerate(v[::-1]))
 a,b=span.split('-');delta=x['duration_s']-(seconds(b)-seconds(a))
 source_timing.append(dict(speaker=x['speaker'],mirror_minus_official_duration_s=delta))
 assert abs(delta)<.2,(x['speaker'],delta)
for x in m['records']:
 for y in x['files'].values():
  p=root/y['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==y['sha256']
 with wave.open(str(root/x['files']['audio']['path'])) as w:
  assert (w.getnchannels(),w.getframerate(),w.getnframes())==(1,16000,160000)
  a=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2');v=a.astype(float)/32768
  rms=float(np.sqrt(np.mean(v*v)));peak=float(abs(v).max());assert rms>1e-4
  sat=int(((a==-32768)|(a==32767)).sum())
 result=subprocess.run([ff,'-nostdin','-v','error','-i',str(root/x['files']['video']['path']),'-map','0:v:0','-f','null','-','-progress','pipe:1'],capture_output=True,check=True,text=True)
 frames=int([line.split('=')[1] for line in result.stdout.splitlines() if line.startswith('frame=')][-1]);assert frames==250,(x['id'],frames)
 rows.append(dict(id=x['id'],speaker=x['speaker'],duration_s=10,start_s=x['start_s'],frames=frames,rms_dbfs=20*np.log10(rms),peak=peak,saturated_samples=sat))
for person in {x['speaker'] for x in m['records']}:
 a,b=sorted([x for x in m['records'] if x['speaker']==person],key=lambda x:x['start_s']);assert b['start_s']>=a['start_s']+10
with (root/'samples.csv').open('w') as f:
 fields=['id','speaker','start_s','duration_s','image','audio','video','transcript_status'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for x in m['records']:
  writer.writerow({**{k:x[k] for k in fields if k in x},**{k:x['files'][k]['path'] for k in ['image','audio','video']}})
canvas=Image.new('RGB',(1000,8*170),'white');draw=ImageDraw.Draw(canvas)
for i,x in enumerate(m['records']):
 image=Image.open(root/x['files']['image']['path']).convert('RGB');image.thumbnail((195,140));px=(i%5)*200;py=(i//5)*170
 canvas.paste(image,(px+(200-image.width)//2,py));draw.text((px+3,py+142),x['speaker'][:24],fill='black');draw.text((px+3,py+155),x['id'].split('_')[-1],fill='black')
canvas.save(root/'contact_sheet.jpg',quality=90)
items=[]
for x in m['records']:
 files=x['files'];items.append(f'<article><h3>{html.escape(x["id"])}</h3><img loading="lazy" src="{files["image"]["path"]}"><p>source start {x["start_s"]:.2f}s · 10 seconds · transcript pending</p><audio controls preload="none" src="{files["audio"]["path"]}"></audio><p><a href="{files["video"]["path"]}">Natural video</a></p></article>')
(root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>HDTF 20 speakers / 40 clips</title><style>body{font-family:system-ui;max-width:1200px;margin:auto}main{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}img{width:100%;max-height:240px;object-fit:contain}h3{font-size:14px}audio{width:100%}</style><h1>HDTF diagnostic subset</h1><p>20 named speakers, 40 fixed 10-second windows. No ASR or model generation. These windows may cut words; prepare sentence boundaries before TTS.</p><main>'+''.join(items)+'</main>')
receipt=dict(status='PASS',speakers=20,clips=40,total_audio_seconds=400,images=40,source_videos=20,all_selected_media_sha_verified=True,all_clips_250_frames=True,all_wavs_mono16k_pcm16_10s=True,no_within_speaker_window_overlap=True,minimum_rms_dbfs=min(x['rms_dbfs'] for x in rows),maximum_rms_dbfs=max(x['rms_dbfs'] for x in rows),saturated_samples_total=sum(x['saturated_samples'] for x in rows),manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),official_source_duration_checks=source_timing,excluded_candidate_files=excluded_sources,rows=rows)
(root/'validation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='rows'},indent=2))
