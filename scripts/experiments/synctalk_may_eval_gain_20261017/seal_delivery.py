"""Seal scoring-only outputs and qualify the parent-input immutability."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main(root):
    assert json.loads((root/'independent_qa.json').read_text())['passed']
    assert not (root/'remote_delivery_manifest.json').exists()
    metadata=dict(audio_clip_encode_passes=24,audio_network_batch_forwards=312,
        interpretation='run.log audio_forward_calls counts complete clip encode passes; each contains 13 forward_aud batches',
        new_training_runs=0,new_generated_video_frames=0,new_visual_encoder_forwards=0)
    (root/'execution_counts.json').write_text(json.dumps(metadata,indent=2)+'\n')
    files=[]
    for path in sorted(root.rglob('*')):
        if path.is_file() and 'mplcache' not in path.relative_to(root).parts:
            files.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    result=dict(passed=True,files=files,total_files=len(files),total_bytes=sum(f['bytes'] for f in files))
    (root/'remote_delivery_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
