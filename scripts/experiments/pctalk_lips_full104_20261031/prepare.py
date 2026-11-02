"""Freeze all105 N/104 T inputs and byte-verified pilot reuse, without stale metadata."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

B=Path('/home/data2/wangjiajun/av-evaluation')
D=Path('/home/data2/wangjiajun/PCTalkDeploy')
SOURCE=Path('/home/data2/wangjiajun/PC-Talk')
DATA=B/'datasets/chinese_lips_test21_20261031'
CLOUD=B/'experiments/chinese_lips_cloud31_20261031_attempt01'
PARENT=B/'jobs/pctalk_lips21_rms24_nine_20261031_attempt01'
RUN=B/'jobs/pctalk_lips_full104_rms24_nine_20261031_attempt01'
GPU='GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while block:=f.read(8*1024**2):h.update(block)
    return h.hexdigest()

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

def native_frames(n):return max(1,int(max(1,1+n//200-16)/80*25))

def normalize(x):
    import numpy as np
    x=np.asarray(x,dtype=np.float64);assert x.ndim==1 and np.isfinite(x).all()
    rms=float(np.sqrt(np.mean(x*x)));assert rms>0
    gain=10**(-24/20)/rms;y=x*gain
    assert abs(y).max()<=10**(-1/20)
    return np.rint(y*32768).astype('<i2'),gain

def main():
    import numpy as np
    import soundfile as sf
    assert not RUN.exists(),'Never overwrite a previous experiment'
    assert json.loads((PARENT/'terminal_receipt.json').read_text())['all_recorded_experiment_processes_exited']
    manifest=json.loads((DATA/'manifest_release.json').read_text())
    natural=sorted([r for r in manifest if r['role']=='target'],key=lambda r:r['id'])
    refs={r['speaker']:r for r in manifest if r['role']=='reference'}
    records={r['id']:r for r in [json.loads(p.read_text()) for p in (CLOUD/'records').glob('*.json')]}
    assert len(natural)==105 and len(records)==104 and len(refs)==21
    assert {r['id'] for r in natural}-set(records)=={'024_29_F_TY_139'}
    assignment=DATA/'hdtf_gender_matched_20261031/assignments.json'
    face_map={r['audio_id']:r for r in json.loads(assignment.read_text())['items']}
    preflight=json.loads((PARENT/'face_preflight.json').read_text())
    faces={r['image_sha256']:r for r in preflight['portraits']}
    parent_rows={r['id']:r for r in json.loads((PARENT/'scoring_inputs.json').read_text())['rows']}
    assert len(parent_rows)==42
    pp=json.loads((PARENT/'protocol.json').read_text())
    assert all(pp['generation'][k]==v for k,v in dict(person_id=192,lip_scale=1.,emc_enabled=False,seed=42,crop_scale=2.3,fps=25).items())
    pins=json.loads((PARENT/'source_pins.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==pp['generation']['source_commit']
    assert not subprocess.check_output(['git','diff','HEAD','--'],cwd=SOURCE)
    for p,h in pins['source'].items():assert sha(SOURCE/p)==h,p
    for p,h in pins['weights'].items():assert sha(SOURCE/'pctalk/checkpoints'/p)==h,p
    scorerpins=json.loads((PARENT/'scorer_pins.json').read_text())
    for p,h in scorerpins['files'].items():assert sha(p)==h,p
    RUN.mkdir(parents=True);rows=[];selection=[];copied=[]
    def copy(a,b):
        a=Path(a);b=Path(b);b.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(a,b);assert sha(a)==sha(b)
        copied.append(dict(parent_path=str(a),path=str(b.relative_to(RUN)),sha256=sha(b)))
    def tree(a,b):
        a=Path(a)
        for p in sorted(a.rglob('*')):
            if p.is_file() and not p.is_symlink():copy(p,Path(b)/p.relative_to(a))
    for raw in natural:
        sid=raw['id'];rec=records.get(sid);face=face_map[sid]
        image=assignment.parent/face['image_path'];assert sha(image)==face['image_sha256']
        image_out=RUN/'images'/image.name
        if not image_out.exists():copy(image,image_out)
        face_index=faces[face['image_sha256']]['selected_index']
        paired=rec is not None
        strict=bool(paired and raw['raw_saturated_samples']==0 and not rec['flags'])
        if rec:
            assert raw['speaker']==rec['speaker'] and raw['text']==rec['text']
            assert rec.get('synthesis_text',rec['text'])==rec['text']
            assert raw['audio_16k_sha256']==rec['natural_sha256']
            assert rec['reference_id']==refs[raw['speaker']]['id']
        selection.append(dict(source=sid,speaker=raw['speaker'],paired_available=paired,strict_quality=strict,
            pilot_seen=sid+'_N' in parent_rows,natural_saturated_samples=raw['raw_saturated_samples'],
            tts_flags=rec['flags'] if rec else ['UPSTREAM_TTS_ENGINE411_MISSING']))
        originals=[('N',DATA/raw['audio_16k'],raw['audio_16k_sha256'])]
        if paired:originals.append(('T',CLOUD/'tts16'/(sid+'.wav'),rec['tts16_sha256']))
        for arm,original,expected in originals:
            assert sha(original)==expected
            x,sr=sf.read(original,dtype='float64');assert sr==16000
            pcm,gain=normalize(x);item=sid+'_'+arm
            audio=RUN/'audio'/(item+'.wav');audio.parent.mkdir(exist_ok=True)
            sf.write(audio,pcm,16000,subtype='PCM_16')
            assert np.array_equal(sf.read(audio,dtype='int16')[0],pcm)
            copy(original,RUN/'original_audio'/(item+'.wav'))
            row=dict(id=item,source=sid,speaker=raw['speaker'],arm=arm,draw=0 if arm=='N' else 1,
                paired_available=paired,strict_quality=strict,pilot_seen=sid+'_N' in parent_rows,
                reference_design='separate_id_utterance',reference_id=refs[raw['speaker']]['id'],
                audio_path=str(audio),audio_sha256=sha(audio),original_audio_path=str(original),
                original_audio_sha256=expected,scalar_gain=gain,samples=len(pcm),duration_s=len(pcm)/16000,
                expected_frames=native_frames(len(pcm)),image_path=str(image_out),image_sha256=sha(image_out),
                face_index=face_index,natural_saturated_samples=raw['raw_saturated_samples'],
                tts_flags=rec['flags'] if rec else ['UPSTREAM_TTS_ENGINE411_MISSING'])
            if item in parent_rows:
                old=parent_rows[item]
                for key in ['audio_sha256','image_sha256','samples','expected_frames','face_index']:
                    assert row[key]==old[key],(item,key)
                assert sha(old['video_path'])==old['video_sha256']
                tree(PARENT/'generation'/item,RUN/'generation'/item)
                copy(old['video_path'],RUN/'videos'/(item+'.mp4'))
                copy(Path(old['video_path']).with_suffix('.jpg'),RUN/'videos'/(item+'.jpg'))
                for model in ['avs_u','mtd','vocalist','synchformer','sparsesync']:
                    assert (PARENT/model/item/'deployment_validation.json').exists()
                    tree(PARENT/model/item,RUN/model/item)
                for suffix in ['.json','_distances.npy']:copy(PARENT/'syncnet'/(item+suffix),RUN/'syncnet'/(item+suffix))
                for folder in ['pywork','pycrop','pyavi']:
                    assert (PARENT/'syncnet/work'/folder/item).is_dir()
                    tree(PARENT/'syncnet/work'/folder/item,RUN/'syncnet/work'/folder/item)
                for name in ['syncnet_'+item+'.log','syncnet_pipeline_'+item+'.log']:
                    copy(PARENT/'logs'/name,RUN/'logs'/name)
            rows.append(row)
    assert len(rows)==209 and len({r['audio_sha256'] for r in rows})==209
    assert sum(r['paired_available'] and r['arm']=='N' for r in rows)==104
    assert sum(r['strict_quality'] and r['arm']=='N' for r in rows)==84
    assert len({r['speaker'] for r in rows if r['strict_quality']})==17
    for folder in ['embeddings','crops']:tree(PARENT/'cogenav'/folder,RUN/'cogenav'/folder)
    copy(PARENT/'cogenav/protocol.json',RUN/'cogenav/reuse_protocol_original.json')
    copy(PARENT/'cogenav/scores.json',RUN/'cogenav/scores.json')
    copy(PARENT/'cogenav/unmeasurable_faces03.json',RUN/'cogenav/unmeasurable_faces03.json')
    copy(PARENT/'syncreward/scores.jsonl',RUN/'syncreward/reused_scores.jsonl')
    for name in ['protocol.json','source_pins.json','scorer_pins.json','scorer_runtime_before.json','scorer_runtime_after.json','terminal_receipt.json','delivery_receipt.json','selection.json']:
        copy(PARENT/name,RUN/'parent_checkpoint'/name)
    for name in ['source_pins.json','scorer_pins.json']:copy(PARENT/name,RUN/name)
    for i,p in enumerate([DATA/'manifest_release.json',assignment,CLOUD/'protocol.json',CLOUD/'verification.json',CLOUD/'audio_manifest.json']+sorted((CLOUD/'records').glob('*.json'))):
        copy(p,RUN/'upstream_metadata'/f'{i:03d}_{p.name}')
    new_pre=dict(preflight,reused_from=str(PARENT),policy='Same21 portrait SHA and native face indices; no repeat face-detector inference')
    new_pre['portraits']=[dict(r,image_path=str(RUN/'images'/Path(r['image_path']).name)) for r in preflight['portraits']]
    write(RUN/'face_preflight.json',new_pre)
    protocol=dict(status='FROZEN_BEFORE_NEW_FORWARD',run_id=RUN.name,selected_dataset='chinese_lips_test21_20261031',
        parent_run=str(PARENT),parent_protocol_sha256=sha(PARENT/'protocol.json'),sources=105,speakers=21,rows=209,
        paired_sources=104,unpaired_sources=['024_29_F_TY_139'],arms=dict(N=105,TTS=104),
        audio=pp['audio'],generation=pp['generation'],metrics=pp['metrics'],support=pp['support'],
        selection='All105 natural targets and all104 existing successful TTS, no score filtering; retain105th unpaired N descriptively',
        references='One separate-ID reference per speaker; source recording interval nonoverlap unverified',
        source_manifest_sha256=sha(DATA/'manifest_release.json'),cloud_manifest_sha256=sha(CLOUD/'audio_manifest.json'),
        image_assignment_sha256=sha(assignment),resources=pp['resources'],unique_audio_files=209,
        peavs='Recompute208 paired media, lexical batches64+64+64+16; unpaired N separate batch1, excluded from paired means',
        statistics='Predeclared per-speaker mean of all planned complete utterance pairs; primary strict84pairs/17speakers, all104pairs/21speakers; new-only strict67/17 and all83/21 sensitivity.10k speaker bootstrap/100k signflip seed20261031;Holm9 separately per stratum.',
        quality_policy='Pair strict iff zero natural source saturation and no provider peak-at-limit flag; no score-based removal',
        complete_speaker_policy='For each metric/stratum, require both arms of every predeclared eligible pair for that speaker. No imputation or threshold relaxation.',
        reuse='42 pilot videos and eight per-clip endpoints reused only with exact input/portrait/weights/settings SHA; CoGenAV allows old input-manifest-only protocol hash with proof. PEAVS recomputed.',
        limitations=pp['limitations']+['Full104 includes pilot21 already inspected; new83 are new utterances from the same21 speakers, not independent participants.','Repeated utterances are aggregated per speaker, not104 independent people.'])
    write(RUN/'protocol.json',protocol);write(RUN/'inputs.json',dict(rows=rows))
    write(RUN/'selection.json',dict(selected=selection,planned_pairs=104,planned_media=209,strict_pairs=84,strict_speakers=17))
    write(RUN/'reuse_manifest.json',dict(status='PASS',parent_run=str(PARENT),reused_ids=sorted(parent_rows),files=copied,
        cogenav_original_protocol_sha256=sha(PARENT/'cogenav/protocol.json'),new_generation=167,new_syncnet=167,
        peavs_reused_scores=0,pilot_score_endpoints_reused=8*42))
    write(RUN/'preparation.json',dict(status='PASS',media=209,speakers=21,paired_sources=104,strict_pairs=84,strict_speakers=17,
        inherited_media=42,new_generation=167,missing_tts=['024_29_F_TY_139'],copied_files_verified=len(copied),official_source_files=len(pins['source']),official_weights=len(pins['weights'])))
    print((RUN/'preparation.json').read_text())

if __name__=='__main__':main()
