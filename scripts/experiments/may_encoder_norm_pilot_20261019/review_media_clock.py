"""Independent ffprobe packet-clock readback; CPU only, no video regeneration."""
from pathlib import Path
import sys,json,subprocess,hashlib
root=Path(sys.argv[1]);rows=[]
for path in sorted((root/'renders').rglob('original_pcm.mkv')):
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)],text=True))
 video=next(s for s in probe['streams'] if s['codec_type']=='video');audio=next(s for s in probe['streams'] if s['codec_type']=='audio')
 assert video['codec_name']=='ffv1' and video['avg_frame_rate']=='25/1' and video['pix_fmt']=='bgr0'
 assert audio['codec_name']=='pcm_s16le' and audio['sample_rate']=='16000' and audio['channels']==1
 assert float(video.get('start_time',0))==float(audio.get('start_time',0))==0
 frames=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)],text=True))['frames']
 times=[float(f['best_effort_timestamp_time']) for f in frames];assert len(times)==250
 err=max(abs(t-i/25) for i,t in enumerate(times));assert err<1e-9
 rows.append(dict(path=path.relative_to(root).as_posix(),codec='FFV1/BGR0+PCM16',frame_times_seconds=times,max_clock_error_seconds=err,frames=250,sample_rate=16000))
assert len(rows)==48
result=dict(passed=True,videos=48,frames=12000,rows=rows,new_gpu_calls=0,new_video_encodes=0)
(root/'media_clock_review.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=True,videos=48,frames=12000,max_clock_error=max(r['max_clock_error_seconds'] for r in rows))))
