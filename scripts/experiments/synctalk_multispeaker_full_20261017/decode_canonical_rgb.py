"""Emit material-QA-authoritative RGB bytes from the original OpenCV runtime."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
a=p.parse_args();assert a.identity in ['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers']
material=a.root/'material'/a.identity
expected=(material/'test_rgb_sha256.bin').read_bytes()
cap=cv2.VideoCapture(str(material/'test.mkv'));assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==25
print(json.dumps(dict(opencv=cv2.__version__,python=sys.executable,hash_authority='material full-frame RGB digest')),file=sys.stderr,flush=True)
for i in range(1550):
    ok,bgr=cap.read();assert ok
    if i>=50:
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()
        assert hashlib.sha256(rgb).digest()==expected[i*32:(i+1)*32],f'frame {i} authority mismatch'
        sys.stdout.buffer.write(rgb)
cap.release();sys.stdout.buffer.flush()
