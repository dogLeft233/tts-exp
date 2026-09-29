"""Independently bind every calibration decode to the sealed generated pixels."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tts_level_visual_calibration_20260927'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    p = read(OUT / 'protocol.json')
    assert sha(OUT / 'protocol.json') == read(OUT / 'seal.json')['protocol_sha256']
    assert sha(p['model']) == p['model_sha256']
    records = [r for r in p['rows'] if r['split'] == 'calibration']
    counts = {'clips': 0, 'frames': 0, 'feature_files': 0, 'full_repeat_files': 0}
    for row in records:
        receipt = read(OUT / 'receipts' / (row['id'] + '.json'))
        assert receipt['id'] == row['id'] and receipt['speaker'] == row['speaker']
        for key, video in row['cells'].items():
            assert sha(video['metadata']) == video['metadata_sha256']
            assert sha(video['path']) == video['sha256']
            raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '1', '-i', video['path'], '-map', '0:v:0', '-an', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-threads', '1', '-'])
            assert hashlib.sha256(raw).hexdigest() == video['pixel_sha256']
            frames = np.frombuffer(raw, np.uint8).reshape(-1, 224, 224, 3)
            assert len(frames) == video['frames']
            qc = receipt['cells'][key]['qc']
            if qc['status'] == 'complete':
                rgb_hash = hashlib.sha256(np.ascontiguousarray(frames[..., ::-1]).tobytes()).hexdigest()
                assert rgb_hash == qc['decoded_RGB_sha256']
            counts['clips'] += 1
            counts['frames'] += len(frames)
            for feature in receipt['cells'][key]['features'].values():
                assert sha(feature['path']) == feature['sha256']
                counts['feature_files'] += 1
                if 'full_path' in feature:
                    assert sha(feature['full_path']) == feature['full_sha256']
                    counts['full_repeat_files'] += 1
        print('PIXELS_VERIFIED', row['id'], flush=True)
    assert counts['clips'] == 104 and counts['full_repeat_files'] == 4
    output = {'status': 'PASS', 'protocol_sha256': sha(OUT / 'protocol.json'), 'checker_sha256': sha(__file__), 'counts': counts,
              'meaning': 'every FFV1 decoded BGR frame equals sealed generated pixels; independent RGB channel conversion equals torchvision decoded input hash; no audio or SyncNet scores read'}
    (OUT / 'independent_pixels.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
