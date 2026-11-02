"""Prepare natural per-recording canonical media and explicit held-out manifests."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def validate_selection(identities):
    if len(identities)!=4 or len({x['identity'] for x in identities})!=4:raise ValueError('require four distinct identities')
    for row in identities:
        if row['main']['original_video_id']==row['test']['original_video_id']:raise ValueError('test recording overlaps train/val')
        if not 360<=row['main']['seconds']+row['test']['seconds']<=480:raise ValueError('material outside 6-8 minutes')
        if row['main']['seconds']<278 or row['test']['seconds']<64:raise ValueError('recording too short for split and context')


def prepare(root,identity):
    import cv2
    import numpy as np
    import soundfile as sf
    selection=json.loads((root/'selection.json').read_text());validate_selection(selection['identities'])
    selected=next(x for x in selection['identities'] if x['identity']==identity)
    folder=root/'material'/identity
    assert not folder.exists(), 'never overwrite prepared identity'
    folder.mkdir(parents=True)
    receipt=dict(identity=identity,recordings=[],supervision=dict(train_frames=6000,val_frames=750,test_frames=1500),
        train_validation_recording=selected['main']['original_video_id'],test_recording=selected['test']['original_video_id'],
        raw_data_only=True,official_3dmm_preprocessing_qualified=False,training_admitted=False)
    for role in ['main','test']:
        row=selected[role];source=root/'source'/row['file']
        assert sha(source)==row['sha256']
        canonical=folder/(role+'.mkv')
        print(json.dumps(dict(identity=identity,phase='normalizing',role=role,seconds=row['seconds'])),flush=True)
        subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-threads','4','-i',str(source),'-t',str(row['seconds']),
            '-map','0:v:0','-map','0:a:0','-vf','scale=512:512:flags=bicubic','-c:v','ffv1','-level','3',
            '-coder','1','-context','1','-g','1','-pix_fmt','yuv420p','-af','pan=mono|c0=c0',
            '-ar','16000','-c:a','pcm_s16le','-threads','4',str(canonical)],check=True)
        wav=folder/(role+'_original.wav')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(canonical),'-map','0:a:0','-c:a','copy',str(wav)],check=True)
        pcm,sr=sf.read(wav,dtype='int16');assert sr==16000 and pcm.shape==(row['seconds']*16000,)
        gains=[]
        for gain in [-6,0,6]:
            driver=pcm.astype(np.float64)/32768*10**(gain/20)
            path=folder/f'{role}_driver_{gain:+d}.wav'
            sf.write(path,driver,sr,subtype='DOUBLE')
            assert np.array_equal(sf.read(path,dtype='float64')[0],driver)
            gains.append(dict(gain_db=gain,path=path.name,sha256=sha(path),peak=float(np.abs(driver).max())))
        cap=cv2.VideoCapture(str(canonical));count=0;digest=hashlib.sha256();gray_prev=None;scene_flags=[]
        with (folder/(role+'_rgb_sha256.bin')).open('xb') as hashes:
            while True:
                ok,image=cap.read()
                if not ok:break
                assert image.shape==(512,512,3)
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                value=hashlib.sha256(rgb.tobytes()).digest();hashes.write(value);digest.update(value)
                gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(32,32)).astype(np.float64)
                if gray_prev is not None:
                    change=float(np.abs(gray-gray_prev).mean())
                    if change>20:scene_flags.append(dict(frame=count,change=change))
                gray_prev=gray;count+=1
        cap.release();assert count==row['seconds']*25
        frame_pts=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','frame=best_effort_timestamp_time','-of','json',str(canonical)]))['frames']
        pts=np.array([float(f['best_effort_timestamp_time']) for f in frame_pts])
        assert len(pts)==count and np.max(np.abs(pts-np.arange(count)/25))<1e-6
        signal=pcm.astype(np.float64)/32768
        samples=dict(peak=float(np.abs(signal).max()),rms=float(np.sqrt(np.mean(signal**2))),
            clipped_samples=int(np.count_nonzero(np.abs(pcm.astype(np.int32))>=32767)),
            silence_fraction_20ms=float(np.mean(np.sqrt(np.mean(signal.reshape(-1,320)**2,axis=1))<10**(-50/20))))
        receipt['recordings'].append(dict(role=role,source=row,path=canonical.name,bytes=canonical.stat().st_size,
            sha256=sha(canonical),original_wav_sha256=sha(wav),pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
            frames=count,audio_samples=len(pcm),frame_rgb_sequence_sha256=digest.hexdigest(),
            pts_max_error=float(np.max(np.abs(pts-np.arange(count)/25))),audio_stats=samples,
            gain_drivers=gains,scene_flags=scene_flags))
        save(folder/'material_receipt.json',receipt)
        print(json.dumps(dict(identity=identity,phase='normalized',role=role,frames=count,bytes=canonical.stat().st_size,scene_flags=len(scene_flags))),flush=True)
    manifest=dict(identity=identity,fps=25,height=512,width=512,material_seconds=420,
        geometry_preprocessing='PENDING official BFM09/face tracking and cross-recording identity coordinate gate',
        splits=[dict(name='train',recording='main',source_start_frame=50,frames=6000,audio_context_start_frame=25,audio_context_frames=6050),
            dict(name='val',recording='main',source_start_frame=6150,frames=750,audio_context_start_frame=6125,audio_context_frames=800),
            dict(name='test',recording='test',source_start_frame=50,frames=1500,audio_context_start_frame=25,audio_context_frames=1550)],
        eval_clips=[dict(id=f'test_{i:02d}',recording='test',source_start_frame=50+i*250,frames=250,samples=160000) for i in range(6)])
    save(folder/'split_manifest.json',manifest)
    receipt['media_preparation_passed']=True;save(folder/'material_receipt.json',receipt)
    print(json.dumps(dict(identity=identity,media_preparation_passed=True,training_admitted=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    args=p.parse_args();prepare(args.root,args.identity)
