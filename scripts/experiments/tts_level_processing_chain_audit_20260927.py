"""Read-only bound current input chain, no API and no historical code inference."""
from pathlib import Path
import hashlib,json,subprocess,time
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_level_activity_resource_audit_20260927';CHAIN=ROOT/'runs/aishell1_qwen_mfa_linear_n100_20260816'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
p=read(ROOT/'runs/tts_native_level_generation_cross_20260927/protocol.json');meta=read(CHAIN/'01_tts_retry/tts_meta.json');cohort=read(CHAIN/'00_pairs/cohort.json');cc={v['sample_id']:v for v in cohort['records']};rows=[]
assert sha(CHAIN/'00_pairs/cohort.json')==meta['cohort_manifest_sha256']
for row in p['rows']:
 sid=row['id'];t=meta['results'][sid];n=cc[sid];record={'id':sid,'split':row['split'],'speaker':row['speaker'],'natural':{},'tts':{}}
 for a in ['N','T']:assert sha(row['audio'][a]['path'])==row['audio'][a]['sha256']
 assert n['audio_sha256']==row['audio']['N']['sha256']==t['reference_audio_sha256'];assert t['canonical_audio_sha256']==row['audio']['T']['sha256']==t['source_audio_sha256']
 nx,sr=sf.read(row['audio']['N']['path'],dtype='int16');tx,tr=sf.read(row['audio']['T']['path'],dtype='int16');assert sr==tr==16000
 for label,pcm in [('natural',nx),('tts',tx)]:record[label].update(current_PCM_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),samples=len(pcm),RMS=float(np.sqrt(np.mean((pcm.astype(float)/32768)**2))),peak=float(abs(pcm.astype(float)).max()/32768),int16_rail_samples=int(np.count_nonzero((pcm==-32768)|(pcm==32767))))
 sp=Path(n['source_audio_path']);record['natural'].update(source_path=str(sp),source_sha256_expected=n['source_audio_sha256'],source_exists=sp.exists(),source_kind=n['source_kind'])
 if sp.exists():
  assert sha(sp)==n['source_audio_sha256'];sx,ss=sf.read(sp,dtype='int16');record['natural'].update(source_sr=ss,source_subtype=sf.info(sp).subtype,source_PCM_exact=bool(ss==sr and np.array_equal(sx,nx)))
 pp=Path(t['provider_audio']);assert pp.exists() and sha(pp)==t['provider_audio_sha256'];px,ps=sf.read(pp,dtype='float64');assert ps==t['provider_sample_rate_hz']
 cmd=['ffmpeg','-v','error','-i',str(pp),'-ar','16000','-ac','1','-c:a','pcm_s16le','-f','s16le','pipe:1'];replay=np.frombuffer(subprocess.check_output(cmd),dtype='<i2')
 pr=float(np.sqrt(np.mean(px*px)));rr=record['tts']['RMS'];record['tts'].update(provider=t['provider'],model=t['model'],provider_path=str(pp),provider_sha256=sha(pp),provider_sr=ps,provider_subtype=sf.info(pp).subtype,provider_RMS=pr,provider_peak=float(abs(px).max()),provider_peak_matches_metadata=float(abs(px).max())==t['provider_peak'],provider_rail_samples=int(np.count_nonzero(abs(px)>=1)),provider_pcm16_grid=bool(np.array_equal(px*32768,np.round(px*32768))),plain_ffmpeg_replay_exact=bool(np.array_equal(replay,tx)),resampling_RMS_dB=float(20*np.log10(rr/pr)),request_payload_saved=False,historical_n100_generator_snapshot_bound=False)
 for a,pcm in [('N',nx),('T',tx)]:
  identity=ROOT/'runs/tts_native_level_generation_cross_20260927/audio'/sid/a/'identity.wav';ix,_=sf.read(identity,dtype='float64');assert np.array_equal(ix,pcm.astype(float)/32768)
 rows.append(record)
summary={'status':'PASS','pairs':len(rows),'bound_natural_and_canonical_hashes':200,'provider_hashes':100,'provider_rates':sorted({r['tts']['provider_sr'] for r in rows}),'provider_formats':sorted({r['tts']['provider_subtype'] for r in rows}),'natural_upstream_available':sum(r['natural']['source_exists'] for r in rows),'natural_upstream_exact':sum(r['natural'].get('source_PCM_exact',False) for r in rows),'plain_ffmpeg_replay_exact':sum(r['tts']['plain_ffmpeg_replay_exact'] for r in rows),'provider_peak_metadata_exact':sum(r['tts']['provider_peak_matches_metadata'] for r in rows),'provider_rail_samples':sum(r['tts']['provider_rail_samples'] for r in rows),'canonical_rail_samples':sum(r['tts']['int16_rail_samples'] for r in rows),'natural_rail_samples':sum(r['natural']['int16_rail_samples'] for r in rows),'resampling_RMS_dB_range':[min(r['tts']['resampling_RMS_dB'] for r in rows),max(r['tts']['resampling_RMS_dB'] for r in rows)],'identity_PCM_div32768_exact_arms':200,'no_API_requests':True,'new_waveform_files':0,'historical_provider_request_and_n100_generator_unverifiable':True,'current_code_only_context':'current dashscope provider request has no explicit volume/gain; sf.read float32 and stereo mean; code is not version-bound to historical100 requests so cannot establish historical or server-side absence'}
write(OUT/'processing_chain_rows.json',rows);write(OUT/'processing_chain_summary.json',summary)
files=[Path(__file__),CHAIN/'00_pairs/cohort.json',CHAIN/'01_tts_retry/tts_meta.json',OUT/'processing_chain_rows.json',OUT/'processing_chain_summary.json',ROOT/'scripts/tts/dashscope_vc.py',ROOT/'scripts/generate_aishell1_qwen_cloud_tts.py'];write(OUT/'processing_chain_seal.json',{'time':time.time(),'hashes':{str(f):sha(f) for f in files},'ffmpeg_version':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]});print(json.dumps(summary,indent=2))
