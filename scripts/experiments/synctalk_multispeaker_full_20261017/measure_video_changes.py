"""Exact decoded pixel changes within fixed identity/seed; no perceptual-quality claim."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
from cohort_binding import identity_root,verify_member_pins

def differences(first,second):
    assert first.dtype==second.dtype==np.uint8 and first.shape==second.shape and first.ndim==3 and first.shape[2]==3 and min(first.shape[:2])>0
    delta=first.astype(np.int16)-second.astype(np.int16)
    return dict(absolute_sum=int(np.abs(delta).sum(dtype=np.int64)),squared_sum=int((delta.astype(np.int32)**2).sum(dtype=np.int64)),
        changed_pixels=int(np.any(delta!=0,axis=2).sum()),channels=int(delta.size),pixels=int(delta.shape[0]*delta.shape[1]))

def run(identity,seed):
    inputs=identity_root(identity,ROOT,require_admission=True);verify_member_pins(ROOT,identity)
    import cv2
    assert identity in ['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers'] and seed in [42,2026]
    folder=ROOT/'renders'/identity/f'seed{seed}';keys=[(a,b) for a in [-6,0,6] for b in [-6,0,6]]
    caps={};receipts={}
    for key in keys:
        out=folder/f'train_{key[0]:+d}'/f'driver_{key[1]:+d}'
        receipts[key]=json.loads((out/'render_receipt.json').read_text());assert receipts[key]['passed']
        caps[key]=cv2.VideoCapture(str(out/'original_pcm.mkv'));assert caps[key].isOpened()
    pairs=[('training',gain,driver,(gain,driver),(0,driver)) for gain in [-6,6] for driver in [-6,0,6]]
    pairs += [('driver',gain,train,(train,gain),(train,0)) for gain in [-6,6] for train in [-6,0,6]]
    totals={index:dict(full={},mouth={},changed_frames=0) for index in range(len(pairs))}
    for i in range(1500):
        images={}
        for key in keys:
            ok,bgr=caps[key].read();assert ok;rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==receipts[key]['frames'][i]['rgb_sha256'];images[key]=rgb
        landmarks=np.loadtxt(inputs/'prepared'/identity/'test/ori_imgs'/f'{i}.lms')[48:68]
        lo=np.maximum(np.floor(landmarks.min(0)).astype(int)-12,0);hi=np.minimum(np.ceil(landmarks.max(0)).astype(int)+13,512)
        for index,(_,_,_,first,second) in enumerate(pairs):
            row=totals[index];full=differences(images[first],images[second]);mouth=differences(images[first][lo[1]:hi[1],lo[0]:hi[0]],images[second][lo[1]:hi[1],lo[0]:hi[0]])
            row['changed_frames']+=int(full['changed_pixels']>0)
            for name,values in [('full',full),('mouth',mouth)]:
                for key,value in values.items():row[name][key]=row[name].get(key,0)+value
    for cap in caps.values():assert not cap.read()[0];cap.release()
    rows=[]
    for index,(axis,gain,fixed,first,second) in enumerate(pairs):
        row=dict(axis=axis,changed_gain_db=gain,fixed_other_gain_db=fixed,first_condition=first,baseline_condition=second,frames=1500,**totals[index])
        for support in ['full','mouth']:
            values=row[support];values.update(mean_absolute_difference_0_1=values['absolute_sum']/values['channels']/255,
                mean_squared_difference_0_1=values['squared_sum']/values['channels']/255**2,changed_pixel_fraction=values['changed_pixels']/values['pixels'])
        rows.append(row)
    result=dict(identity=identity,seed=seed,passed=True,pairs=rows,all1500_frames_per_condition_verified=True,
        interpretation='exact appearance/pixel change, including mouth region; does not establish motion accuracy or perceived synchronization')
    (folder/'pixel_changes.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(identity=identity,seed=seed,passed=True,pairs=len(rows))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.identity,a.seed)
