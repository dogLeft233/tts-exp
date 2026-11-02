"""Freeze identical CAL20 waveforms and matched-phone support before encoder results."""
from pathlib import Path
import hashlib,json,numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path('F:/Datasets/tts-exp/encoder_survey_20261018')
MODELS=[
 ('hubert_base','facebook/hubert-base-ls960','wave'),
 ('hubert_chinese','TencentGameMate/chinese-hubert-base','wave'),
 ('wav2vec2_base','facebook/wav2vec2-base','wave'),
 ('wavlm_base_plus','microsoft/wavlm-base-plus','wave'),
 ('data2vec_base','facebook/data2vec-audio-base','wave'),
 ('whisper_base','openai/whisper-base','whisper'),
 ('whisper_small','openai/whisper-small','whisper'),
 ('hubert_large','facebook/hubert-large-ll60k','wave'),
 ('xlsr53','facebook/wav2vec2-large-xlsr-53','wave'),
 ('wavlm_large','microsoft/wavlm-large','wave'),
 ('w2vbert2','facebook/w2v-bert-2.0','fbank'),
]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def main():
 inp=OUT/'input';inp.mkdir(parents=True,exist_ok=False)
 cal=ROOT/'runs/recovered_20260930/remote/calibration_20260930'
 worldpath=ROOT/'runs/recovered_20260930/remote/world_phone_transfer_20261001_v2/calibration_review_20261001/world_phone_transfer_calibration.json'
 world={r['id']:r for r in json.loads(worldpath.read_text(encoding='utf8'))['records']}
 summarypath=cal/'calibration_summary.json';sm=json.loads(summarypath.read_text(encoding='utf8'))
 rows=[]
 for pid,s in sorted(sm['clips'].items()):
  src=cal/'calibration_features'/(pid+'.npz')
  with np.load(src,allow_pickle=False) as d:
   arr={k:d[k] for k in ['natural_raw_float32','tts_raw_float32','natural_fixed_float32','tts_fixed_float32']}
  for arm in ['natural','tts']:
   assert digest(arr[arm+'_raw_float32'])==world[pid][arm]['raw_float32_pcm_sha256']
   assert digest(arr[arm+'_fixed_float32'])==world[pid][arm]['fixed_audio']['float32_pcm_sha256']
   assert np.isclose(np.sqrt(np.mean(arr[arm+'_fixed_float32'].astype(float)**2)),.0376838172675746,rtol=1e-7)
  arr['natural_plus3db']=(arr['natural_fixed_float32'].astype(float)*10**(3/20)).astype('float32')
  arr['natural_minus3db']=(arr['natural_fixed_float32'].astype(float)*10**(-3/20)).astype('float32')
  assert all(np.max(np.abs(x))<1 for x in arr.values())
  dst=inp/(pid+'.npz');np.savez(dst,**arr)
  phones=[]
  for i,p in enumerate(world[pid]['directions']['tts_to_natural']['mapping']['matched_phones']):
   assert p['recipient_normalized_label']==p['donor_normalized_label']
   if p['recipient_normalized_label'] in {'','sil','sp','spn','<unk>'}:continue
   phones.append({'index':i,'label':p['recipient_normalized_label'],'word':p['word_label'],
     'natural':p['recipient_interval_s'],'tts':p['donor_interval_s']})
  rows.append({'id':pid,'speaker':pid.split('_')[0],'input':str(dst),'sha256':sha(dst),
    'source_npz_sha256':sha(src),'phones':phones,
    'samples':{k:len(v) for k,v in arr.items()},'pcm_sha256':{k:digest(v) for k,v in arr.items()}})
 assert len(rows)==20 and len({r['speaker'] for r in rows})==10
 protocol={'status':'FROZEN_BEFORE_NEW_ENCODER_FORWARD','cohort':'CAL20 reused exploratory; 10 male speakers, 2 sentences each; not independent confirmation or contamination-clean claim',
 'models':[{'id':i,'repo':r,'kind':k} for i,r,k in MODELS],
 'conditions':['natural_raw_float32','tts_raw_float32','natural_fixed_float32','tts_fixed_float32','natural_plus3db','natural_minus3db'],
 'endpoints':{'TTS_RAW':[0,1],'TTS_FIXED':[2,3],'GAIN_P3':[2,4],'GAIN_M3_SIGNED':[5,2]},
 'design':'same exact PCM for all encoders; native preprocessing retained; no forced normalization bypass; whole utterance context; no TFG/SyncNet/training',
 'geometry':'unit per-frame; mean in exact matched phone interval then unit again; target-source; phone->sentence->speaker equal weight; cross-speaker and LOSO axes within each encoder/layer',
 'support':'common matched-phone intersection across all 11 new encoders, main full phone; center50 sensitivity; report missing phones explicitly; no interpolation',
 'layers':'all hidden states summarized; save per-token directions/radii for layers 0,1,2,middle,last, plus actual last-layer frames of first clip',
 'signal_gate':'per-endpoint mean unit-vector chord RMS>=1e-5 and >10*RAW repeat chord RMS, otherwise commonality marked BELOW_DIRECTION_RESOLUTION, no null-effect claim',
 'statistics':'descriptive reused cohort; 10 speaker units not 45 pair units; leave-one-speaker-out axes; report all layers, no best-layer selection; bootstrap intervals descriptive only',
 'clock':'wave models exact conv receptive-field center from config; Whisper centered STFT encoder stride20ms; Seamless fbank25ms/10ms stride2 stacked center17.5ms',
 'controls':'first clip RAW twice, exact PCM digests/length/RMS/peak; finite arrays; processor input response; actual output count and timestamp support; heldout and gain-vs-TTS separation',
 'resource':'local RTX4060 only, batch1/FP32/no TF32, max50% allocator; one model per subprocess; public weights removable after verified scientific readback; retain pinned URL/revision/SHA and all outputs',
 'source':{'world_sha256':sha(worldpath),'summary_sha256':sha(summarypath)},'rows':rows}
 (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({'clips':len(rows),'models':len(MODELS),'protocol_sha256':sha(OUT/'protocol.json')}))
if __name__=='__main__':main()
