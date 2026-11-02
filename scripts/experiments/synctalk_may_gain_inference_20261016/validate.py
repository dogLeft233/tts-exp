"""Independently replay saved distances, support policy, pixels and source PCM."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np


def assert_pcm_equal(actual,expected):
    assert actual.dtype==expected.dtype and actual.shape==expected.shape, 'PCM length/dtype mismatch'
    assert np.array_equal(actual,expected), 'PCM content changed'


def score_curve(matrix,guard):
    selected=matrix[guard:len(matrix)-guard] if guard else matrix
    curve=selected.astype(np.float64).mean(0)
    return {'sync_c':float(np.median(curve)-curve.min()),'sync_d':float(curve.min()),
            'distance_d0':float(curve[15]),'av_offset':int(15-curve.argmin())}


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    import cv2
    import soundfile as sf
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root;render=root/'render_attempt01';score=root/'score_attempt02'
    protocol=json.loads((root/'protocol.json').read_text())
    receipt=json.loads((render/'render_receipt.json').read_text());scores=json.loads((score/'scores.json').read_text())
    original,sr=sf.read(root/'data/May/aud.wav',dtype='int16');assert sr==16000
    expected=original[3532800:3724800,0].copy()
    pcm,_=sf.read(render/'original_score_audio.wav',dtype='int16');assert_pcm_equal(pcm,expected)
    wave=original[:,0].astype(np.float64)/32768
    af=np.load(score/'source_audio_features.npy');assert af.shape[0]==295
    padded=np.pad(af,((15,15),(0,0)))
    matrix_max=0.;metric_max=0.;frames_checked=0;arm_checks=[]
    for arm in receipt['arms']:
        name=arm['arm'];row=next(x for x in scores['arms'] if x['arm']==name)
        driver,_=sf.read(render/name/'data/aud.wav',dtype='float64')
        assert np.array_equal(driver,wave*10**(arm['gain_db']/20))
        assert np.max(np.abs(driver))<1
        container=render/name/'original_pcm.mkv'
        assert digest(container)==arm['original_pcm_container_sha256']
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(container),'-map','0:a:0',
                                     '-f','s16le','-acodec','pcm_s16le','pipe:1'])
        assert_pcm_equal(np.frombuffer(raw,dtype=np.int16),expected)
        cap=cv2.VideoCapture(str(container));assert cap.get(cv2.CAP_PROP_FPS)==25
        count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==arm['frames'][count]['rgb_sha256']
            count+=1
        cap.release();assert count==300;frames_checked+=count
        v=np.load(score/(name+'_visual_features.npy'));d=np.load(score/(name+'_distances.npy'))
        assert d.dtype==np.float32 and d.shape==(295,31)
        rebuilt=[]
        for j in range(31):
            delta=v.astype(np.float64)-padded[j:j+295].astype(np.float64)+1e-6
            rebuilt.append(np.sqrt(np.sum(delta*delta,axis=1)))
        rebuilt=np.stack(rebuilt,1)
        matrix_error=float(np.max(np.abs(rebuilt-d)));assert matrix_error<2e-5
        matrix_max=max(matrix_max,matrix_error)
        for scope,guard in [('full',0),('guard20',20)]:
            computed=score_curve(d,guard)
            for key in ['sync_c','sync_d','distance_d0']:
                error=abs(computed[key]-row[scope][key]);assert error<1e-5
                metric_max=max(metric_max,error)
            assert computed['av_offset']==row[scope]['av_offset']
        arm_checks.append(dict(arm=name,frames=count,original_pcm_exact=True,pure_gain_exact=True,
                              matrix_max_error=matrix_error))
    assert scores['zero_repeat_pixels_exact'] and scores['zero_repeat_score_max_abs']==0
    assert np.array_equal(np.load(score/'gain_0_distances.npy'),np.load(score/'gain_0_repeat_distances.npy'))
    assert next(x for x in scores['arms'] if x['arm']=='gain_0_repeat')['pixels']['full_max_abs']==0
    result={'status':'PASS','frames_replayed':frames_checked,'distance_cells_replayed':4*295*31,
            'matrix_float64_replay_max_abs':matrix_max,'metric_float64_replay_max_abs':metric_max,
            'zero_repeat_exact':True,'arms':arm_checks,'scores_sha256':digest(score/'scores.json')}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
