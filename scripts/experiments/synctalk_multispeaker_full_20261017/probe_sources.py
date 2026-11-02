"""Read-only media inventory and visual contact sheets for natural source eligibility."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def main(root):
    rows=[];tiles=[]
    for path in sorted((root/'source').glob('*.mp4')):
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
        video=next(s for s in info['streams'] if s['codec_type']=='video')
        audio=next(s for s in info['streams'] if s['codec_type']=='audio')
        duration=float(video.get('duration',info['format']['duration']))
        rows.append(dict(name=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            duration=duration,frames=video.get('nb_frames'),width=video['width'],height=video['height'],
            fps=video['avg_frame_rate'],video_start=video.get('start_time'),audio_start=audio.get('start_time'),
            audio_rate=audio['sample_rate'],audio_channels=audio['channels'],audio_duration=audio.get('duration')))
        cap=cv2.VideoCapture(str(path));samples=[]
        for fraction in [.05,.35,.65,.95]:
            cap.set(cv2.CAP_PROP_POS_MSEC,duration*fraction*1000);ok,image=cap.read();assert ok,path
            image=cv2.resize(image,(240,240));samples.append(image)
        cap.release()
        tile=np.zeros((270,960,3),np.uint8)
        tile[:240]=np.concatenate(samples,axis=1)
        cv2.putText(tile,path.name,(8,260),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile)
    (root/'source_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
    cv2.imwrite(str(root/'source_contact_sheet.jpg'),np.concatenate(tiles))
    print(json.dumps(rows,indent=2),flush=True)
    print(json.dumps(dict(dependencies={name:importlib.util.find_spec(name) is not None for name in
        ['face_alignment','mediapipe','pytorch3d','lpips','sklearn','torchvision']})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
