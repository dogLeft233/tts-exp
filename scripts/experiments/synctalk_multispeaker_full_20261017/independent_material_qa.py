"""Re-read source/canonical media and audit gain, clocks, support and duplicates."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def duplicate_audio(a,b,sample_rate=2000):
    import numpy as np
    from scipy.signal import fftconvolve
    width=4*sample_rate
    energy=np.r_[0,np.cumsum(b*b)]
    norms=np.sqrt(np.maximum(energy[width:]-energy[:-width],0))
    best=dict(absolute_correlation=0.0,probe_start_seconds=None,match_start_seconds=None)
    for start in range(0,len(a)-width+1,width):
        probe=a[start:start+width].copy();probe-=probe.mean()
        if np.sqrt(np.mean(probe*probe))<10**(-40/20):continue
        corr=fftconvolve(b,probe[::-1],mode='valid')
        values=np.abs(corr)/(np.linalg.norm(probe)*np.maximum(norms,1e-12))
        at=int(np.argmax(values));value=float(values[at])
        if value>best['absolute_correlation']:
            best=dict(absolute_correlation=value,probe_start_seconds=start/sample_rate,match_start_seconds=at/sample_rate)
    best['flag']=best['absolute_correlation']>=.90
    best['method']='non-silent 4s waveform probes every 4s, resampled 2kHz, full target lag search, abs normalized correlation >=0.90'
    return best

def verify(root,identity):
    import cv2,numpy as np,soundfile as sf
    from scipy.signal import resample_poly
    folder=root/'material'/identity
    receipt=json.loads((folder/'material_receipt.json').read_text())
    split=json.loads((folder/'split_manifest.json').read_text())
    assert receipt['media_preparation_passed']
    records=[];signals={};rgb={}
    for row in receipt['recordings']:
        role=row['role'];canonical=folder/row['path'];source=root/'source'/row['source']['file']
        assert sha(source)==row['source']['sha256'] and sha(canonical)==row['sha256']
        pcm,sr=sf.read(folder/f'{role}_original.wav',dtype='int16');assert sr==16000 and pcm.ndim==1
        assert hashlib.sha256(pcm.tobytes()).hexdigest()==row['pcm_sha256']
        direct=subprocess.check_output(['ffmpeg','-v','error','-i',str(source),'-t',str(row['source']['seconds']),
            '-vn','-af','pan=mono|c0=c0','-ar','16000','-acodec','pcm_s16le','-f','s16le','pipe:1'])
        assert direct==pcm.tobytes(),'source PCM changed during preparation'
        for gain in row['gain_drivers']:
            path=folder/gain['path'];assert sha(path)==gain['sha256']
            samples,rate=sf.read(path,dtype='float64');assert rate==16000
            assert np.array_equal(samples,pcm.astype(np.float64)/32768*10**(gain['gain_db']/20))
        expected=(folder/f'{role}_rgb_sha256.bin').read_bytes();assert len(expected)==row['frames']*32
        cap=cv2.VideoCapture(str(canonical));index=0;digest=hashlib.sha256()
        while True:
            ok,image=cap.read()
            if not ok:break
            assert image.shape==(512,512,3)
            block=hashlib.sha256(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).tobytes()).digest()
            assert block==expected[index*32:(index+1)*32],(role,index)
            digest.update(block);index+=1
        cap.release();assert index==row['frames'] and digest.hexdigest()==row['frame_rgb_sequence_sha256']
        meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_entries',
            'frame=best_effort_timestamp_time','-select_streams','v:0','-of','json',str(canonical)]))
        pts=np.array([float(x['best_effort_timestamp_time']) for x in meta['frames']])
        assert len(pts)==index and np.max(np.abs(pts-np.arange(index)/25))<1e-6
        assert len(pcm)==index*640
        signals[role]=pcm.astype(np.float64)/32768
        rgb[role]={expected[i:i+32] for i in range(0,len(expected),32)}
        records.append(dict(role=role,frames=index,pcm_exact_source=True,gains_exact=True,rgb_exact=True,pts_exact=True))
    for s in split['splits']:
        start=s['source_start_frame'];stop=start+s['frames'];cstart=s['audio_context_start_frame'];cstop=cstart+s['audio_context_frames']
        assert cstart<=start-25 and cstop>=stop+25
        assert 0<=cstart and cstop*640<=len(signals[s['recording']])
    train,val,test=split['splits'];assert train['name']=='train' and val['name']=='val' and test['name']=='test'
    assert train['audio_context_start_frame']+train['audio_context_frames']+50<=val['audio_context_start_frame']
    assert receipt['train_validation_recording']!=receipt['test_recording']
    duplicate=duplicate_audio(resample_poly(signals['main'],1,8),resample_poly(signals['test'],1,8))
    exact_shared=len(rgb['main']&rgb['test']);assert not duplicate['flag'] and exact_shared==0
    clips=[]
    for c in split['eval_clips']:
        x=signals['test'][c['source_start_frame']*640:(c['source_start_frame']+c['frames'])*640]
        assert len(x)==160000
        clips.append(dict(id=c['id'],rms=float(np.sqrt(np.mean(x*x))),peak=float(np.abs(x).max()),
            silence_fraction_20ms=float(np.mean(np.sqrt(np.mean(x.reshape(-1,320)**2,axis=1))<10**(-50/20)))))
    result=dict(identity=identity,passed=True,recordings=records,audio_duplicate_check=duplicate,
        exact_shared_rgb_frames=exact_shared,fixed_test_clip_diagnostics=clips,
        source_recording_identity_visual_review='separate source contact sheet and official original IDs',
        face_landmark_coverage_qualified=False,official_pose_qualified=False,training_admitted=False)
    (folder/'independent_material_qa.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();verify(a.root,a.identity)
