"""Create a new continuation run from byte-verified completed PC-Talk outputs."""
from pathlib import Path
import hashlib
import json
import shutil

B=Path('/home/data2/wangjiajun/av-evaluation')
PARENT=B/'jobs/pctalk_chinese20_rms24_20261031_attempt01'
RUN=B/'jobs/pctalk_chinese20_rms24_nine_20261031_attempt01'
GPU='GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8*1024**2):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def support(rows):
    eligible=[r['id'] for r in rows if min(r['samples']//640,r['actual_video_frames'])>=125]
    complete=[s for s in sorted({r['source'] for r in rows}) if all(r['id'] in eligible for r in rows if r['source']==s)]
    return dict(status='FROZEN_BEFORE_EIGHT_NEW_SCORERS',classifier_eligible_media=len(eligible),
        classifier_eligible_ids=eligible,classifier_complete_sources=complete,
        face_support_threshold=.95,face_support_failure='UNMEASURABLE; no threshold relaxation or replacement',
        policy='Complete source needs one N and all three T; no imputation. Aliases must match audio+video SHA.')

def main():
    assert not RUN.exists(),'Fresh continuation required'
    assert json.loads((PARENT/'terminal_receipt.json').read_text())['all_recorded_experiment_processes_exited']
    original=json.loads((PARENT/'scoring_inputs.json').read_text())
    assert len(original['rows'])==80 and len(original['unique_rows'])==41
    RUN.mkdir(parents=True)
    inherited=[]
    for row in original['rows']:
        assert sha(row['audio_path'])==row['audio_sha256']
        assert sha(row['video_path'])==row['video_sha256']
        for field,folder in [('audio_path','audio'),('video_path','videos')]:
            old=Path(row[field]);new=RUN/folder/old.name;new.parent.mkdir(exist_ok=True)
            shutil.copyfile(old,new);assert sha(old)==sha(new)
            row['parent_'+field]=str(old);row[field]=str(new)
        row['actual_video_frames']=json.loads((PARENT/'generation'/row['id']/'receipt.json').read_text())['frames']
        inherited.append(dict(id=row['id'],audio_sha256=row['audio_sha256'],video_sha256=row['video_sha256']))
    shutil.copytree(PARENT/'generation',RUN/'generation')
    for p in (PARENT/'syncnet').iterdir():
        if p.is_file() and p.suffix in ['.json','.npy']:
            target=RUN/'syncnet'/p.name;target.parent.mkdir(exist_ok=True)
            shutil.copyfile(p,target);assert sha(p)==sha(target)
    shutil.copyfile(PARENT/'generation_complete.json',RUN/'generation_complete.json')
    write(RUN/'inputs.json',dict(rows=original['rows']))
    protocol=json.loads((PARENT/'protocol.json').read_text())
    protocol.update(run_id=RUN.name,parent_run=str(PARENT),scope='Eight additional scorers; parent generation and SyncNet unchanged',
        parent_protocol_sha256=sha(PARENT/'protocol.json'),reuse='All80 generation outputs and41 SyncNet scores reused byte-for-byte; no generation/SyncNet model forward',
        new_scorers=['CoGenAV Eq1','AVS_u frame cosine','MTDVocaLiST K5','VocaLiST K5','Synchformer P0','SparseSync P0','SyncReward BF16 encode_batch1','PEAVS batch64+16'],
        inference_qualification='UNMEASURABLE at unchanged native support gates; preserve all failures, no imputation',
        statistics='Three existing T draws averaged per source; primary same_reference12, supplementary extended8/all20;10k bootstrap/100k signflip seed20261030;Holm9 per stratum including reused SyncNet')
    write(RUN/'protocol.json',protocol)
    write(RUN/'predeclared_support.json',support(original['rows']))
    write(RUN/'stages/syncnet_complete.json',dict(status='COMPLETE',reused_from=str(PARENT),scores_sha256=sha(RUN/'syncnet/scores.json'),no_model_forward=True))
    write(RUN/'cogenav/scores.json',[])
    write(RUN/'cogenav/unmeasurable_faces03.json',[])
    for name in ['protocol.json','inputs.json','scoring_inputs.json','summary.json','score_audit.json','source_pins.json','scorer_pins.json','official_internal_codec_diagnostic.json','delivery_receipt.json','terminal_receipt.json']:
        dest=RUN/'parent_checkpoint'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(PARENT/name,dest)
    write(RUN/'reuse_receipt.json',dict(status='PASS',parent=str(PARENT),media=inherited,
        parent_scores_sha256=sha(PARENT/'syncnet/scores.json'),copied_scores_sha256=sha(RUN/'syncnet/scores.json'),
        generated_new_videos=0,new_syncnet_forwards=0,scientific_scope='Only eight newly authorized scorers'))
    print((RUN/'predeclared_support.json').read_text())

if __name__=='__main__':main()
