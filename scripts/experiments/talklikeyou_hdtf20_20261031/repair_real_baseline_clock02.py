"""Local GT-only CFR remux repair; no change to any frozen TFG input or output."""
from pathlib import Path
import json,subprocess,tarfile,hashlib
import imageio_ffmpeg,numpy as np,soundfile as sf
from prepare import sha,write

def main():
    root=Path('runs/hdtf20_cloud31_20261031_attempt01').resolve();bundle=root/'server_bundle_rms27'
    original=json.loads((root/'audio_manifest.json').read_text())['rows']
    manifest=json.loads((bundle/'bundle_manifest.json').read_text());lookup={r['source']:r for r in manifest['real_rows']}
    out=root/'real_baseline_clock02';out.mkdir(exist_ok=False);rows=[];ff=imageio_ffmpeg.get_ffmpeg_exe()
    for row in original:
        sid=row['id'];p=out/(sid+'_G.mp4');ref=lookup[sid];audio=bundle/ref['audio_path']
        subprocess.run([ff,'-nostdin','-v','error','-n','-i',row['real_video'],'-i',str(audio),'-map','0:v:0','-map','1:a:0',
            '-vf','trim=end_frame=245,setpts=N/(25*TB)','-r','25','-fps_mode','cfr','-c:v','libx264','-crf','18','-preset','fast','-c:a','alac',str(p)],check=True)
        decoded=subprocess.run([ff,'-nostdin','-v','error','-i',str(p),'-map','0:v:0','-fps_mode','passthrough','-f','null','/dev/null',
            '-map','0:a:0','-f','s16le','pipe:1','-progress','pipe:2'],capture_output=True,check=True)
        frames=int([x.split('=')[1] for x in decoded.stderr.decode().splitlines() if x.startswith('frame=')][-1])
        expected=sf.read(audio,dtype='int16')[0].astype('<i2').tobytes();assert frames==245 and decoded.stdout==expected
        rows.append(dict(id=sid+'_G',new_relative_path='real_matched_cfr02/'+p.name,new_sha256=sha(p),old_sha256=ref['video_sha256'],frames=frames,pcm_exact=True,samples=160000))
    write(out/'real_baseline_clock02.json',dict(status='PASS',rows=rows,initial_failure='first old GT reported245 container frames but244 decoded; explicit CFR yields245 decoded',
        scope='Only GT diagnostic media; all80 TFG images/audio and every generated output unchanged; repair before any scoring',
        method='First245 actual source frames at original25fps, explicit N/(25*TB) and CFR25, no audio time edits'))
    with tarfile.open(out.with_suffix('.tar.gz'),'w:gz',compresslevel=3) as tf:
        for p in out.iterdir():tf.add(p,arcname=p.name)
    print(json.dumps(dict(status='PASS',real_videos=40,archive=str(out.with_suffix('.tar.gz')),sha256=sha(out.with_suffix('.tar.gz')))))
if __name__=='__main__':main()
