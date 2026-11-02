"""Complete missing train-only ASR calls; cache only successful immutable responses."""
from pathlib import Path
import json,sys
from concurrent.futures import ThreadPoolExecutor
from prepare_audio import OUT,sha,save,transcribe
def main():
 key=sys.stdin.readline().strip();p=json.loads((OUT/'protocol.json').read_text())
 def work(row):
  d=OUT/'chunks'/f"{row['id']:03d}";assert sha(d/'natural.wav')==row['natural_sha256']
  r=d/'asr_response.json'
  if r.exists():
   v=json.loads(r.read_text());assert v['http_status']==200
   text=v['body']['output']['choices'][0]['message']['content'][0]['text'];cached=True
  else:text=transcribe(key,d/'natural.wav',r);cached=False
  assert text.strip();(d/'transcript.txt').write_text(text,encoding='utf-8')
  return dict(**row,text=text,asr_response_sha256=sha(r),transcript_sha256=sha(d/'transcript.txt'),reused_success=cached)
 with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(work,p['chunks']))
 save(OUT/'transcription_manifest.json',dict(passed=True,source_frames=[0,4200],duration_seconds=168,chunks=rows,model=p['asr_model'],input_normalization=False,training_jobs=0))
 print(json.dumps(dict(passed=True,chunks=len(rows),cached=sum(r['reused_success'] for r in rows))))
if __name__=='__main__':main()
