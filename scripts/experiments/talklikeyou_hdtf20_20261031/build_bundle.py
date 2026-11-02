"""Local-only media bundle: preserve raw inputs; derive RMS27 PCM and GT support."""
import argparse,json,shutil,subprocess,tarfile
from pathlib import Path
import numpy as np,soundfile as sf
from prepare import native_frames,sha,write

def normalize(x):
    x=np.asarray(x,dtype=np.float64);assert x.ndim==1 and np.isfinite(x).all()
    rms=float(np.sqrt(np.mean(x*x)));assert rms>0
    gain=10**(-27/20)/rms;y=x*gain
    assert abs(y).max()<=10**(-1/20),'RMS27 would exceed frozen peak ceiling; do not silently lower gain or clip'
    pcm=np.rint(y*32768).astype(np.int16)
    assert np.max(abs(pcm.astype(np.int32)))<32767
    return pcm,dict(rms_before=rms,gain=gain,rms_after=float(np.sqrt(np.mean((pcm.astype(float)/32768)**2))),peak_after=float(abs(pcm.astype(float)/32768).max()))

def main():
    import imageio_ffmpeg
    p=argparse.ArgumentParser();p.add_argument('--cloud',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cloud=a.cloud.resolve();out=a.output.resolve();assert not out.exists()
    done=json.loads((cloud/'cloud_complete.json').read_text());assert done['status']=='PASS'
    records=json.loads((cloud/'audio_manifest.json').read_text())['rows'];assert len(records)==40
    out.mkdir(parents=True);rows=[];real=[]
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for r in records:
        sid=r['id'];face=out/'images'/(sid+'.png');face.parent.mkdir(exist_ok=True);shutil.copyfile(r['image'],face);assert sha(face)==r['image_sha256']
        for arm,key in [('N','natural'),('T','tts16')]:
            raw=Path(r[key]);assert sha(raw)==r['natural_sha256' if arm=='N' else 'tts16_sha256']
            canonical=out/'canonical'/(sid+'_'+arm+'.wav');canonical.parent.mkdir(exist_ok=True);shutil.copyfile(raw,canonical)
            x,sr=sf.read(canonical,dtype='float64');assert sr==16000
            pcm,stats=normalize(x);audio=out/'audio'/(sid+'_'+arm+'.wav');audio.parent.mkdir(exist_ok=True);sf.write(audio,pcm,16000,subtype='PCM_16')
            n=native_frames(len(pcm))
            rows.append(dict(id=sid+'_'+arm,source=sid,speaker=r['speaker'],arm=arm,draw=0 if arm=='N' else 1,strict_reference=True,
                text=r['text'],tts_qc_flag=r['qc_flag'],tts_asr_wer=r['word_error_rate'],
                duration_s=len(pcm)/16000,samples=len(pcm),expected_frames=n,unsupported_tail_samples=len(pcm)-n*640,
                audio_path=str(audio.relative_to(out)),audio_sha256=sha(audio),canonical_audio_path=str(canonical.relative_to(out)),canonical_audio_sha256=sha(canonical),
                face_path=str(face.relative_to(out)),face_sha256=sha(face),level=stats))
        nr=rows[-2];gt=out/'real_matched'/(sid+'_G.mp4');gt.parent.mkdir(exist_ok=True)
        assert sha(r['real_video'])==r['real_video_sha256']
        # Copy the first expected natural TFG frames, zero origin, untouched video speed.
        subprocess.run([ffmpeg,'-nostdin','-v','error','-n','-i',r['real_video'],'-i',str(out/nr['audio_path']),
            '-map','0:v:0','-map','1:a:0','-vf',f"trim=end_frame={nr['expected_frames']},setpts=N/(25*TB)",'-r','25','-fps_mode','cfr',
            '-c:v','libx264','-crf','18','-preset','fast','-c:a','alac',str(gt)],check=True)
        real.append(dict(id=sid+'_G',source=sid,speaker=r['speaker'],arm='G',draw=0,duration_s=10,samples=160000,
            audio_path=nr['audio_path'],video_path=str(gt.relative_to(out)),audio_sha256=nr['audio_sha256'],video_sha256=sha(gt),
            actual_video_frames=nr['expected_frames'],expected_frames=nr['expected_frames'],real_original_sha256=r['real_video_sha256'],
            support_policy='native real video prefix matched to natural TFG frame count, full RMS27 N audio retained'))
    for name in ['protocol.json','cloud_complete.json','transcripts.json','audio_manifest.json','normalization_amendment_rms27.json']:shutil.copyfile(cloud/name,out/('cloud_'+name if name=='protocol.json' else name))
    for folder in ['responses','requests','records','provider_audio','reference24']:
        shutil.copytree(cloud/folder,out/'cloud_evidence'/folder)
    files=[dict(path=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.rglob('*')) if f.is_file()]
    write(out/'bundle_manifest.json',dict(rows=rows,real_rows=real,files=files,protocol=dict(sources=40,speakers=20,rows=80,
        selected_dataset='HDTF diagnostic20',arms=dict(N=40,TTS=40),audio=dict(target_rms_dbfs=-27,peak_ceiling_dbfs=-1,method='single scalar gain and round-to-nearest PCM16'),
        metrics=['SyncNet','CoGenAV','PEAVS','SyncReward','AVS_u','MTDVocaLiST','VocaLiST','Synchformer','SparseSync'],
        statistics='two clips averaged within each arm and speaker; paired20; bootstrap10000/signflip100000 seed20261030; Holm9',
        cloud=dict(asr='qwen-audio-3.1-asr-flash',tts='qwen-audio-3.1-tts-plus',draws_per_clip=1,self_reference=True),
        strict_unseen=False,official_test_split=False,boundary_policy='fixed ten-second windows may cut words/sentences; no manual transcript verification',
        real_baseline='SyncNet only; forty original-video prefixes matched to N TFG support; not TTS+real-video mismatch scoring')))
    archive=out.with_suffix('.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        for f in sorted(out.rglob('*')):
            if f.is_file():tf.add(f,arcname=str(f.relative_to(out)))
    print(json.dumps(dict(status='PASS',rows=len(rows),speakers=20,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive))))
if __name__=='__main__':main()
