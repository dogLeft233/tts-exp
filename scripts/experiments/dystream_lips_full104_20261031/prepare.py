"""Freeze the same209 PCM/image inputs; never reuse another generator's outputs."""
from pathlib import Path
import hashlib,json,shutil,subprocess

B=Path('/home/data2/wangjiajun/av-evaluation')
D=Path('/home/data2/wangjiajun/DyStreamDeploy')
SOURCE=Path('/home/data2/wangjiajun/DyStream')
PARENT=B/'jobs/pctalk_lips_full104_rms24_nine_20261031_attempt01'
RUN=B/'jobs/dystream_lips_full104_rms24_nine_20261031_attempt01'
GPU='GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while x:=f.read(8*1024**2):h.update(x)
    return h.hexdigest()

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

def native_frames(samples):
    # Official app prepends inpainting prefix P; inference emits floor(N/640)-1
    # new frames using window=P+2,stride1, then removes P. No tail extension.
    assert samples>=1280
    return samples//640-1

def main():
    assert not RUN.exists(),'Never replace an existing experiment'
    parent=json.loads((PARENT/'inputs.json').read_text())['rows']
    assert len(parent)==209 and len({r['audio_sha256'] for r in parent})==209
    assert (D/'acceptance03.exit').read_text().strip()=='0'
    assets=json.loads((D/'delivery/asset_verification.json').read_text())
    assert assets['passed']
    pins={}
    for line in (D/'delivery/source-files.sha256').read_text().splitlines():
        h,rel=line.split(maxsplit=1);p=SOURCE/rel.strip();assert sha(p)==h;pins[str(p)]=h
    for rec in assets['files']:
        p=Path(rec['path'])
        if p.suffix=='.pyc':continue
        assert p.stat().st_size==rec['size'] and sha(p)==rec['sha256'],str(p)
        pins[str(p)]=rec['sha256']
    for line in (D/'delivery/final_files.sha256').read_text().splitlines():
        h,rel=line.split(maxsplit=1);p=D/rel.strip();assert sha(p)==h;pins[str(p)]=h
    reuse=json.loads((D/'delivery/wav2vec_reuse.json').read_text())
    # Bind the complete actual offline HF cache, including linked files.
    for p in sorted((D/'cache/huggingface').rglob('*')):
        if p.is_file() and '.locks' not in p.parts:pins[str(p)]=sha(p)
    frozen=(D/'delivery/environment.freeze.txt').read_text()
    actual=subprocess.check_output([str(D/'env/bin/python'),'-m','pip','freeze'],text=True)
    assert set(actual.splitlines())==set(frozen.splitlines())
    RUN.mkdir(parents=True);copies=[]
    def copy(a,b):
        a=Path(a);b=Path(b);b.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(a,b);h=sha(a);assert sha(b)==h
        copies.append(dict(parent_path=str(a),path=str(b.relative_to(RUN)),sha256=h))
    rows=[]
    for old in parent:
        for k,h in [('audio_path','audio_sha256'),('image_path','image_sha256'),('original_audio_path','original_audio_sha256')]:assert sha(old[k])==old[h]
        row=dict(old)
        for k,folder in [('audio_path','audio'),('image_path','images')]:
            dest=RUN/folder/Path(old[k]).name
            if not dest.exists():copy(old[k],dest)
            row[k]=str(dest)
        copy(PARENT/'original_audio'/(old['id']+'.wav'),RUN/'original_audio'/(old['id']+'.wav'))
        row['expected_frames']=native_frames(row['samples'])
        row['pc_talk_face_index_provenance']=row.pop('face_index')
        rows.append(row)
    for name in ['inputs.json','protocol.json','selection.json']:
        copy(PARENT/name,RUN/'parent_input_snapshot'/name)
    copy(PARENT/'scorer_pins.json',RUN/'scorer_pins.json')
    copy(D/'delivery/environment.freeze.txt',RUN/'dystream_environment_baseline.txt')
    copy(D/'gpu_infer.py',RUN/'deployment_snapshot/gpu_infer.py')
    copy(D/'infer.sh',RUN/'deployment_snapshot/infer.sh')
    write(RUN/'source_pins.json',dict(files=pins,source_commit='f587541ccaa77f092d03487d7d89502a67f0a084',
        asset_revision='9ad5b1d3b0ef7aece4b9855d972c1f819f04dfbb'))
    oldp=json.loads((PARENT/'protocol.json').read_text())
    protocol={k:oldp[k] for k in ['audio','metrics','support','statistics','quality_policy','complete_speaker_policy','references','image_assignment_sha256','source_manifest_sha256','cloud_manifest_sha256']}
    protocol.update(status='FROZEN_BEFORE_FORWARD',run_id=RUN.name,model='DyStream',parent_inputs=str(PARENT),
        rows=209,sources=105,speakers=21,paired_sources=104,unpaired_sources=['024_29_F_TY_139'],
        arms=dict(N=105,TTS=104),unique_audio_files=209,
        generation=dict(source_commit='f587541ccaa77f092d03487d7d89502a67f0a084',asset_revision='9ad5b1d3b0ef7aece4b9855d972c1f819f04dfbb',
            entry='Verified official app.run_inference speaker-only, unchanged renderer adapter; CPU MediaPipe',
            seed=222,steps=5,cfg_audio=.5,cfg_listener=.5,cfg_anchor=0.,cfg_all=1.,listener='native zero tensor',
            fps=25,resolution=[512,512],frames='floor(samples/640)-1; official inpainting prefix/removal unchanged',
            audio_frontend='Official librosa16k + native zero prefix and Wav2Vec2 processor; capture exact PCM tensor and actual do_normalize value',
            soundtrack='Preserve native AAC MP4; separate scoring MP4 stream-copy video + original full16kPCM encoded losslessly ALAC; zero PTS, no shift/stretch/padding'),
        resources=dict(gpu_uuid=GPU,cpu_threads=4,affinity=[0,1,2,3],torch_fraction=.55),
        reuse='Input PCM/original WAV/images only. Zero generator outputs or scorer results reused across models.',
        peavs='208 paired media lexical batches64+64+64+16, unpaired natural separate batch1',
        dependency='PC-Talk server orchestrator_complete, successful independent audits, terminal receipt. Failed/stalled parent never triggers inference.',
        limitations=['Evaluator evidence, not perceptual truth or complete leakage exclusion.',
            'Native renderer uses aligned512 face rather than PC-Talk source-size pasteback; native prefix/tail/audio processing differ.',
            'Historical PC-Talk pilot_seen labels retained only for matched sensitivity; no DyStream score has been inspected.',
            'Same21 speakers with repeated utterances; speaker-weighted inference. No independent participant replication.',
            'Full PCM remains intact; visual tail coverage follows native generator, no posthoc AV offset correction.',
            'Reference source interval nonoverlap and training/test overlap remain unverified.'])
    write(RUN/'protocol.json',protocol);write(RUN/'inputs.json',dict(rows=rows))
    write(RUN/'selection.json',json.loads((PARENT/'selection.json').read_text()))
    write(RUN/'input_copy_manifest.json',dict(status='PASS',files=copies))
    write(RUN/'experiment_source_manifest.json',dict(files={str(p):sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ['.py','.sh']}))
    write(RUN/'preparation.json',dict(status='PASS',media=209,paired_sources=104,speakers=21,strict_pairs=84,strict_speakers=17,
        generator_outputs_reused=0,scorer_outputs_reused=0,source_asset_pins=len(pins),copied_files=len(copies)))
    print('PREPARATION_PASS',209,104,len(pins))

if __name__=='__main__':main()
