"""Select largest actual native face before any video/scorer inference."""
import os
import shutil
import sys
from prepare import RUN, SOURCE, sha, write
import json

def main():
    os.chdir(SOURCE);sys.path.insert(0,str(SOURCE))
    import cv2
    import numpy as np
    from pctalk.liveportrait.config.crop_config import CropConfig
    from pctalk.liveportrait.utils.face_analysis_diy import FaceAnalysisDIY
    cfg=CropConfig()
    detector=FaceAnalysisDIY(name='buffalo_l',root=cfg.insightface_root,providers=['CPUExecutionProvider'])
    detector.prepare(ctx_id=-1,det_size=(512,512),det_thresh=cfg.det_thresh)
    rows=json.loads((RUN/'inputs.json').read_text())['rows'];by_image={};tiles=[]
    assert not (RUN/'generation').exists()
    for row in rows:
        path=row['image_path']
        if path in by_image:continue
        image=cv2.imread(path);assert image is not None
        faces=detector.get(image,direction=cfg.direction,flag_do_landmark_2d_106=True,max_face_num=cfg.max_face_num)
        assert faces, path
        areas=[float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) for f in faces]
        selected=int(np.argmax(areas));f=faces[selected]
        assert areas[selected]/(image.shape[0]*image.shape[1])>.015
        record=dict(image_path=path,image_sha256=sha(path),selected_index=selected,
            faces=[dict(bbox=f.bbox.tolist(),score=float(f.det_score)) for f in faces],
            sort_direction=cfg.direction,threshold=cfg.det_thresh,rule='Largest native bbox area; image-only pre-generation selection')
        by_image[path]=record
        x1,y1,x2,y2=map(int,f.bbox);cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
        tile=cv2.resize(image,(256,256));cv2.putText(tile,row['speaker']+' idx'+str(selected),(5,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1)
        tiles.append(tile)
    shutil.copyfile(RUN/'inputs.json',RUN/'inputs_before_face_preflight.json')
    shutil.copyfile(RUN/'protocol.json',RUN/'protocol_before_face_preflight.json')
    for row in rows:row['face_index']=by_image[row['image_path']]['selected_index']
    write(RUN/'inputs.json',dict(rows=rows))
    protocol=json.loads((RUN/'protocol.json').read_text())
    protocol['generation'].update(face_index='per-row native left-right index of largest detected bbox',
        image_policy='Gender-matched fixed HDTF portrait per speaker; same both arms; pre-generation largest native bbox chosen without audio/scores')
    write(RUN/'protocol.json',protocol)
    while len(tiles)%5:tiles.append(np.zeros((256,256,3),dtype=np.uint8))
    sheet=np.concatenate([np.concatenate(tiles[i:i+5],axis=1) for i in range(0,len(tiles),5)],axis=0)
    cv2.imwrite(str(RUN/'face_preflight.jpg'),sheet)
    write(RUN/'face_preflight.json',dict(status='PASS',portraits=list(by_image.values()),video_forwards=0,scorer_forwards=0))
    print('FACE_PREFLIGHT_PASS',len(by_image),[r['selected_index'] for r in by_image.values()])

if __name__=='__main__':main()
