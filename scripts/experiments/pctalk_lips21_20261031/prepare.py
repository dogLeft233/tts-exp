"""Freeze one existing successful TTS pair per speaker, without score selection."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

B = Path('/home/data2/wangjiajun/av-evaluation')
D = Path('/home/data2/wangjiajun/PCTalkDeploy')
SOURCE = Path('/home/data2/wangjiajun/PC-Talk')
DATA = B/'datasets/chinese_lips_test21_20261031'
CLOUD = B/'experiments/chinese_lips_cloud31_20261031_attempt01'
RUN = B/'jobs/pctalk_lips21_rms24_nine_20261031_attempt01'
GPU = 'GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8*1024**2): h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def native_frames(samples):
    return max(1, int(max(1, 1+samples//200-16)/80*25))

def select(records):
    chosen = {}
    for r in sorted(records, key=lambda x:x['id']): chosen.setdefault(r['speaker'], r)
    return [chosen[s] for s in sorted(chosen)]

def normalize(x, target=-24):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x*x)))
    assert x.ndim == 1 and rms > 0 and np.isfinite(x).all()
    gain = 10**(target/20)/rms
    y = x*gain
    assert abs(y).max() <= 10**(-1/20), 'Do not silently clip or change the frozen target'
    return np.rint(y*32768).astype('<i2'), gain

def main():
    import numpy as np
    import soundfile as sf
    assert not RUN.exists(), 'Fresh run required'
    records = [json.loads(p.read_text()) for p in sorted((CLOUD/'records').glob('*.json'))]
    chosen = select(records)
    assert len(records) == 104 and len(chosen) == 21
    manifest = json.loads((DATA/'manifest_release.json').read_text())
    natural = {r['id']:r for r in manifest if r['role']=='target'}
    assignments = DATA/'hdtf_gender_matched_20261031/assignments.json'
    faces = {r['audio_id']:r for r in json.loads(assignments.read_text())['items']}
    parent = B/'jobs/pctalk_chinese20_rms24_20261031_attempt01'
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=SOURCE, text=True).strip()
    assert commit == 'a0392744cbac857d96db1b99ceab70cf7b5f6c83'
    assert not subprocess.check_output(['git','diff','HEAD','--'], cwd=SOURCE)
    pins = {p:sha(SOURCE/p) for p in subprocess.check_output(['git','ls-files'],cwd=SOURCE,text=True).splitlines() if (SOURCE/p).is_file()}
    weights = SOURCE/'pctalk/checkpoints'; assets={}
    for line in (weights/'SHA256SUMS').read_text().splitlines():
        expected,name = line.split(maxsplit=1); name=name.lstrip('* ').removeprefix('pctalk/checkpoints/')
        assert sha(weights/name)==expected
        assets[name]=expected
    assert len(assets)==12
    RUN.mkdir(parents=True)
    rows=[]; selection=[]
    for rec in chosen:
        source=rec['id']; raw=natural[source]; face=faces[source]
        assert raw['speaker']==rec['speaker'] and raw['audio_16k_sha256']==rec['natural_sha256']
        image=assignments.parent/face['image_path']; assert sha(image)==face['image_sha256']
        target_image=RUN/'images'/image.name; target_image.parent.mkdir(exist_ok=True)
        shutil.copyfile(image,target_image)
        strict = raw['raw_saturated_samples']==0 and not rec['flags']
        selection.append(dict(source=source,speaker=rec['speaker'],strict_quality=strict,
            natural_saturated_samples=raw['raw_saturated_samples'],tts_flags=rec['flags'],
            reference_id=rec['reference_id'],image_person=face['hdtf_person'],image_sha256=sha(image)))
        for arm,original,expected in [('N',DATA/raw['audio_16k'],rec['natural_sha256']),('T',CLOUD/'tts16'/(source+'.wav'),rec['tts16_sha256'])]:
            assert sha(original)==expected
            x,sr=sf.read(original,dtype='float64'); assert sr==16000 and x.ndim==1
            y,gain=normalize(x); item=source+'_'+arm
            audio=RUN/'audio'/(item+'.wav'); audio.parent.mkdir(exist_ok=True)
            sf.write(audio,y,16000,subtype='PCM_16')
            reread,_=sf.read(audio,dtype='int16');assert np.array_equal(y,reread)
            rows.append(dict(id=item,source=source,speaker=rec['speaker'],arm=arm,draw=0 if arm=='N' else 1,
                strict_reference=True,strict_quality=strict,audio_path=str(audio),audio_sha256=sha(audio),
                original_audio_path=str(original),original_audio_sha256=expected,scalar_gain=gain,
                samples=len(y),duration_s=len(y)/16000,expected_frames=native_frames(len(y)),
                image_path=str(target_image),image_sha256=sha(target_image),face_index=0,
                natural_saturated_samples=raw['raw_saturated_samples'],tts_flags=rec['flags']))
    protocol=json.loads((parent/'protocol.json').read_text())
    protocol.update(run_id=RUN.name,scope='21 speakers, one existing successful N/T pair each; all nine native scorers',
        cohort='Chinese-LiPS official test audio; cross-identity gender-matched HDTF portraits',
        selection='Per speaker, lexicographically first successfully synthesized target ID. Fixed before generation/scoring; no quality/score selection.',
        source_manifest_sha256=sha(DATA/'manifest_release.json'),cloud_manifest_sha256=sha(CLOUD/'audio_manifest.json'),
        image_assignment_sha256=sha(assignments),audio=dict(sample_rate=16000,channels=1,format='PCM16',rms_dbfs=-24,
        gain='scalar only, separately recorded per item; <= -1dB peak gate; no retime/denoise/padding',score_audio='same full PCM as driving audio'),
        generation=dict(model='PC-Talk',source_commit=commit,weights_revision='e6220816118bd8714ff57dc83e5906122b4e9168',
            fps=25,seed=42,person_id=192,lip_scale=1.,emotion=[],emc_enabled=False,face_index=0,crop_scale=2.3,
            image_policy='One fixed assigned HDTF portrait per audio speaker; same in both arms; largest detected face index0. Unlike old3.png index1.',
            frame_policy='Official complete mel windows, preserve full PCM; no external padding, retime or offset correction'),
        resources=dict(gpu_uuid=GPU,torch_memory_fraction=.45,cpu_threads=4,execution='nohup sequential GPU, affinity0-3'),
        statistics='one pair per speaker; primary strict_quality17 and supplementary all21; 10k bootstrap and100k signflip seed20261031; Holm9 per subset',
        quality_policy='strict requires zero natural source saturation and no TTS provider peak flags; all21 retained; original damage not repaired by scalar gain',
        limitations=['Single TTS draw, dataset transcripts unverified by listening/ASR.',
        'Independent reference ID does not establish nonoverlap in underlying recording.',
        'Chinese-LiPS audio provenance differs from HDTF; target portraits are HDTF and may overlap training. No all-component leakage-free claim.',
        'Scores are evaluator evidence, not perceptual synchronization truth. PEAVS batch composition matters.'],
        duplicate_audio_groups=[])
    assert sum(s['strict_quality'] for s in selection)==17
    write(RUN/'protocol.json',protocol);write(RUN/'inputs.json',dict(rows=rows))
    write(RUN/'selection.json',dict(policy=protocol['selection'],available_pairs=104,missing_api=['024_29_F_TY_139'],selected=selection))
    write(RUN/'source_pins.json',dict(source_root=str(SOURCE),source=pins,weights=assets))
    write(RUN/'cogenav/scores.json',[]);write(RUN/'cogenav/unmeasurable_faces03.json',[])
    write(RUN/'preparation.json',dict(status='PASS',sources=21,media=42,strict_quality=17,
        official_asset_files_verified=len(assets),source_files_verified=len(pins)))
    print((RUN/'preparation.json').read_text())

if __name__=='__main__': main()
