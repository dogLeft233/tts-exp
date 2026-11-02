"""Create immutable audible inputs from the frozen Chinese20 canonical cohort."""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np

B = Path('/home/data2/wangjiajun/av-evaluation')
RUN = B/'jobs/flashhead_chinese20_rms24_20261030_attempt01'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)

def normalized_pcm(x, target=10**(-24/20)):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 1 and len(x) and np.isfinite(x).all()
    rms = float(np.sqrt(np.mean(x*x)))
    assert rms > 1e-8
    gain = target/rms
    y = x*gain
    if abs(y).max() > 10**(-1/20):
        raise ValueError('Target would exceed -1dBFS peak; do not clip or normalize arms differently')
    pcm = np.rint(y*32768).astype('<i2')
    return pcm, gain

def main():
    import soundfile as sf
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);a=p.parse_args()
    assert not a.run.exists(), 'Fresh run directory required'
    parent=B/'active_cohort.json'; cohort=json.loads(parent.read_text())
    rows=sorted(cohort['rows'],key=lambda r:(r['source'],r['arm']!='N',r['draw']))
    assert len(rows)==80 and len({r['source'] for r in rows})==20
    assert all(r['dataset']=='chinese20' for r in rows)
    assert sum(r['arm']=='N' for r in rows)==20
    a.run.mkdir(parents=True)
    inputs=[]
    for row in rows:
        src=Path(row['audio_path']);assert sha(src)==row['audio_sha256']
        info=sf.info(src);assert info.samplerate==16000 and info.channels==1 and info.subtype=='PCM_16'
        x,sr=sf.read(src,dtype='float64');pcm,gain=normalized_pcm(x)
        dest=a.run/'audio'/(row['id']+'.wav');dest.parent.mkdir(exist_ok=True)
        sf.write(dest,pcm,sr,subtype='PCM_16')
        decoded,_=sf.read(dest,dtype='int16');assert np.array_equal(decoded,pcm)
        y=decoded.astype(np.float64)/32768
        rms=float(np.sqrt(np.mean(y*y)));peak=float(abs(y).max())
        assert abs(20*math.log10(rms)+24)<.002 and peak<=10**(-1/20)
        inputs.append(dict(row,canonical_audio_path=str(src),canonical_audio_sha256=sha(src),
            audio_path=str(dest),audio_sha256=sha(dest),samples=len(x),duration_s=len(x)/sr,
            gain=gain,gain_db=20*math.log10(gain),rms=rms,rms_dbfs=20*math.log10(rms),peak=peak,
            strict_reference=sf.info(next(Path(r['audio_path']) for r in rows if r['source']==row['source'] and r['arm']=='N')).duration>=5))
    groups={}
    for r in inputs:groups.setdefault(r['audio_sha256'],[]).append(r['id'])
    protocol=dict(status='FROZEN_BEFORE_GENERATION',run_id=a.run.name,
        parent_manifest_sha256=sha(parent),selected_dataset='chinese20',sources=20,rows=80,
        arms={'N':20,'TTS':60},audio=dict(target_rms_dbfs=-24,peak_ceiling_dbfs=-1,
        method='single scalar gain per entire clip then round-to-nearest PCM16; no compression, trimming, resynthesis, time shift or silence padding; canonical files unchanged'),
        generation=dict(model='SoulX-FlashHead Pro',seed=42,audio_encode_mode='stream',face_path=cohort['face_path'],
        face_sha256=sha(cohort['face_path']),fps=25,size=[512,512],compile=False,
        reuse='Identical normalized WAV SHA, face and generation seed/settings permit exact video artifact copy with explicit alias receipt'),
        metrics=['SyncNet','CoGenAV','PEAVS','SyncReward','AVS_u','MTDVocaLiST','VocaLiST','Synchformer','SparseSync'],
        support='Native full clip; MTD/Vocal common full frames; classifiers center125 full frames; SyncReward official minimum; short media UNMEASURABLE, never silence-padded to qualify',
        peavs='all80 lexically sorted input filenames in single official batch64+16 invocation',
        statistics='Source-equal mean of3 TTS draws minus N; all20 plus primary same-reference12 and secondary extended-reference8. Complete-pair per metric, no imputation. Bootstrap10000 seed20261030, signflip100000 sensitivity, Holm9 per subset. Scores not perceptual truth.',
        resources='personal directories only; nohup; single GPU UUID; torch60%; CPU4; sequential GPU stages',
        duplicate_audio_groups=[v for v in groups.values() if len(v)>1],unique_audio_files=len(groups))
    write(a.run/'protocol.json',protocol);write(a.run/'inputs.json',{'rows':inputs})
    write(a.run/'preparation.json',dict(status='PASS',protocol_sha256=sha(a.run/'protocol.json'),
        verified_rows=80,unique_audio_files=len(groups),rms_dbfs_range=[min(r['rms_dbfs'] for r in inputs),max(r['rms_dbfs'] for r in inputs)],
        max_peak=max(r['peak'] for r in inputs),gain_db_range=[min(r['gain_db'] for r in inputs),max(r['gain_db'] for r in inputs)]))
    print((a.run/'preparation.json').read_text())

if __name__=='__main__':main()
