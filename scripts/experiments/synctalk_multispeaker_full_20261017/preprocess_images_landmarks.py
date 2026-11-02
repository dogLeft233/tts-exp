"""Official JPEG/FAN/BiSeNet preprocessing for the fixed supervision support."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

def require_primary_face(boxes):
    if boxes is None or len(boxes)==0:raise ValueError('no face')
    areas=[max(0,float(b[2]-b[0]))*max(0,float(b[3]-b[1])) for b in boxes]
    if areas[0]<=0 or float(boxes[0][4])<.9 or max(areas)>areas[0]:
        raise ValueError('official first face is not a confident dominant face')
    for b,area in zip(boxes[1:],areas[1:]):
        if float(b[4])>=.9 or area>=areas[0]*.1:raise ValueError('competing face requires review')
    return len(boxes)-1


def run(root,identity,reuse_image_root=None):
    import cv2,numpy as np,face_alignment
    official=root/'official';material=root/'material'/identity
    manifest=json.loads((material/'split_manifest.json').read_text())
    out=root/'preprocessed'/identity
    assert not out.exists(),'never replace executed preprocessing'
    out.mkdir(parents=True)
    receipt=dict(identity=identity,stage='official_jpeg_fan_bisenet',splits=[],started=time.time(),
        pose_qualified=False,training_admitted=False)
    execution_source=Path(__file__).read_bytes();source_sha=hashlib.sha256(execution_source).hexdigest()
    source_folder=root/'executed_sources'/source_sha;source_folder.mkdir(parents=True,exist_ok=True)
    archived=source_folder/Path(__file__).name
    if archived.exists():assert archived.read_bytes()==execution_source
    else:archived.write_bytes(execution_source)
    import torch
    receipt['provenance']=dict(execution_source_sha256=source_sha,
        material_receipt_sha256=hashlib.sha256((material/'material_receipt.json').read_bytes()).hexdigest(),
        split_manifest_sha256=hashlib.sha256((material/'split_manifest.json').read_bytes()).hexdigest(),
        official_parser_sha256=hashlib.sha256((official/'data_utils/face_parsing/test.py').read_bytes()).hexdigest(),
        software=dict(torch=torch.__version__,face_alignment=face_alignment.__version__,opencv=cv2.__version__,numpy=np.__version__))
    def save():(out/'stage1_receipt.json').write_text(json.dumps(receipt,indent=2))
    save()
    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False)
    for s in manifest['splits']:
        folder=out/s['name'];images=folder/'ori_imgs';images.mkdir(parents=True)
        for name in ['parsing','gt_imgs','torso_imgs','face_mask','flow_result']:(folder/name).mkdir()
        start=s['source_start_frame'];stop=start+s['frames']
        reused=reuse_image_root/s['name']/'ori_imgs' if reuse_image_root else None
        if reused and len(list(reused.glob('*.jpg')))==s['frames']:
            for p in reused.glob('*.jpg'):os.link(p,images/p.name)
        else:
            subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(material/(s['recording']+'.mkv')),
                '-an','-vf',f'trim=start_frame={start}:end_frame={stop},setpts=PTS-STARTPTS',
                '-qmin','1','-q:v','1','-start_number','0',str(images/'%d.jpg')],check=True)
        assert len(list(images.glob('*.jpg')))==s['frames']
        stats=dict(name=s['name'],frames=s['frames'],landmarks_processed=0,primary_face_frames=0,
            minor_secondary_detections=[],review_frames=[],
            source_recording=s['recording'],source_start_frame=start,failed_frames=[],minimum_interocular_px=None)
        receipt['splits'].append(stats);save()
        resume=0
        old_receipt=reuse_image_root/'stage1_receipt.json' if reuse_image_root else None
        if old_receipt and old_receipt.exists():
            old=json.loads(old_receipt.read_text())
            prior=next((x for x in old['splits'] if x['name']==s['name']),None)
            if prior:
                resume=prior['landmarks_processed']
                for i in range(resume):
                    p=reused/f'{i}.lms';lands=np.loadtxt(p)
                    assert lands.shape==(68,2) and np.isfinite(lands).all()
                    os.link(p,images/p.name)
                stats.update(landmarks_processed=resume,primary_face_frames=resume,
                    minor_secondary_detections=prior['minor_secondary_detections'],
                    review_frames=prior.get('review_frames',[]),
                    minimum_interocular_px=prior['minimum_interocular_px'],
                    reused_landmarks=resume,reused_receipt_sha256=hashlib.sha256(old_receipt.read_bytes()).hexdigest())
                save()
        for i in range(resume,s['frames']):
            image=cv2.cvtColor(cv2.imread(str(images/f'{i}.jpg')),cv2.COLOR_BGR2RGB)
            preds,_,boxes=fa.get_landmarks(image,return_bboxes=True)
            try:secondary=require_primary_face(boxes)
            except ValueError as error:
                issue=dict(frame=i,error=str(error),boxes=[] if boxes is None else [b.tolist() for b in boxes])
                if str(error)=='competing face requires review':
                    stats['review_frames'].append(issue);secondary=len(boxes)-1
                else:
                    stats['failed_frames'].append(issue);save()
                    raise RuntimeError(f'{identity}/{s["name"]}/{i}: {error}')
            if secondary:stats['minor_secondary_detections'].append(dict(frame=i,boxes=[b.tolist() for b in boxes[1:]]))
            lands=preds[0].reshape(-1,2)
            assert lands.shape==(68,2) and np.isfinite(lands).all()
            assert (lands[48:68]>=0).all() and (lands[48:68]<512).all(),'mouth outside image'
            eye=float(np.linalg.norm(lands[36:42].mean(0)-lands[42:48].mean(0)))
            stats['minimum_interocular_px']=eye if stats['minimum_interocular_px'] is None else min(eye,stats['minimum_interocular_px'])
            np.savetxt(images/f'{i}.lms',lands,'%f')
            stats['primary_face_frames']+=1;stats['landmarks_processed']+=1
            if (i+1)%250==0:
                save();print(json.dumps(dict(identity=identity,split=s['name'],phase='landmarks',frames=i+1)),flush=True)
        save()
    del fa
    import torch
    torch.cuda.empty_cache()
    for s in manifest['splits']:
        folder=out/s['name'];images=folder/'ori_imgs'
        subprocess.run([sys.executable,'data_utils/face_parsing/test.py',f'--respath={folder/"parsing"}',f'--imgpath={images}'],cwd=official,check=True)
        assert len(list((folder/'parsing').glob('[0-9]*.png')))==2*s['frames']
        for i in range(s['frames']):
            # Official parser emits parsing/N_face.png; official flow loader expects face_mask/N.png.
            shutil.copy2(folder/'parsing'/f'{i}_face.png',folder/'face_mask'/f'{i}.png')
        next(x for x in receipt['splits'] if x['name']==s['name'])['parsing_complete']=True;save()
    receipt['passed']=True;receipt['completed']=time.time()
    receipt['face_coverage_qualified']=not any(x['review_frames'] for x in receipt['splits'])
    receipt['review_required']=not receipt['face_coverage_qualified'];save()
    print(json.dumps(dict(identity=identity,stage1_passed=True,training_admitted=False)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    p.add_argument('--reuse-image-root',type=Path)
    a=p.parse_args();run(a.root,a.identity,a.reuse_image_root)
