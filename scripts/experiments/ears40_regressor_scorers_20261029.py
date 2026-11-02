"""EARS40 PEAVS official batch64 and SyncReward official BF16 adapters.

PEAVS controls have their own within-batch native baseline. The primary native
batch64 scores are never compared with shifted media in a different batch.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/home/wjj/tts-audio/tts-exp')
NAME = 'ears40_w2l_nine_scorers_20261029_attempt01'
RUN = ROOT/'runs'/NAME
DEPLOY = Path('/home/wjj/av-eval-20261026')
SCRATCH = Path('/dev/shm')/NAME/'regressors'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def shift_pcm(pcm, samples):
    import numpy as np
    out = np.zeros_like(pcm)
    if samples > 0:
        out[samples:] = pcm[:-samples]
    elif samples < 0:
        out[:samples] = pcm[-samples:]
    else:
        out[:] = pcm
    return out


def call(args, log, cwd=None):
    import importlib.util
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    spec = importlib.util.find_spec('torch')
    if spec is not None:
        site = Path(spec.origin).parent.parent
        libs = sorted((site/'nvidia').glob('*/lib'))
        env['LD_LIBRARY_PATH'] = ':'.join([*(str(p) for p in libs), env.get('LD_LIBRARY_PATH', '')])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        subprocess.run(list(map(str, args)), stdout=stream, stderr=subprocess.STDOUT,
                       cwd=cwd, env=env, check=True, timeout=14400)


def validate_rows(rows):
    assert len(rows) == 80 and len({r['id'] for r in rows}) == 80
    sources = {r['sample_id'] for r in rows}
    assert len(sources) == 40
    for source in sources:
        pair = [r for r in rows if r['sample_id'] == source]
        assert len(pair) == 2 and {r['arm'] for r in pair} == {'N', 'T'}


def control_media(rows):
    import numpy as np
    import soundfile as sf
    folder = SCRATCH/'shifted_media'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rows:
        if row['arm'] != 'N':
            continue
        pcm, sr = sf.read(row['audio_path'], dtype='int16')
        assert sr == 16000 and pcm.ndim == 1
        for samples, label in [(-6400, 'minus400ms'), (6400, 'plus400ms')]:
            dest = folder/(row['id']+'_'+label+'.mp4')
            audio = folder/(row['id']+'_'+label+'.wav')
            expected = shift_pcm(pcm, samples).astype('<i2')
            if not dest.exists():
                sf.write(audio, expected, sr, subtype='PCM_16')
                call(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', row['video_path'], '-i', audio,
                      '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'alac',
                      '-map_metadata', '-1', dest], RUN/'regressor_media_logs'/(dest.stem+'.log'))
            actual = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(dest),
                                              '-map', '0:a:0', '-f', 's16le', '-'])
            assert actual == expected.tobytes()
            # Stream copy is also checked at decoded pixel/frame level.
            hashes = []
            for path in [row['video_path'], dest]:
                raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
                      '-map', '0:v:0', '-an', '-f', 'framemd5', '-']).decode()
                hashes.append([line.split(',')[-1].strip() for line in raw.splitlines()
                               if not line.startswith('#')])
            assert hashes[0] == hashes[1]
            manifest.append(dict(id=row['id'], source=row['sample_id'], shift_samples=samples,
                                  video_path=str(dest), video_sha256=sha(dest), pcm_exact=True,
                                  video_decoded_exact=True, pcm_sha256=hashlib.sha256(actual).hexdigest()))
    write(RUN/'regressor_controls_media.json', manifest)
    assert len(manifest) == 80 and len({r['video_path'] for r in manifest}) == 80
    return manifest


def source_control_batches(naturals, shifted):
    """Each <=63-item batch contains whole 3-item source families."""
    lookup = {(r['id'], r['shift_samples']): r['video_path'] for r in shifted}
    batches = []
    for start in range(0, len(naturals), 21):
        batch = []
        for row in naturals[start:start+21]:
            batch.extend([row['video_path'], lookup[(row['id'], -6400)], lookup[(row['id'], 6400)]])
        batches.append(batch)
    return batches


def peavs_values(raw):
    return {Path(r['filename']).stem: float(r['score']) for r in raw['scores']}


def copy_features(work, out):
    source = work/'av_features_extraction/output_feats'
    target = out/'features'
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted(source.rglob('*')):
        if path.is_file():
            dest = target/path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                assert sha(dest) == sha(path)
            else:
                shutil.copyfile(path, dest)
            manifest.append(dict(file=str(dest.relative_to(out)), bytes=dest.stat().st_size,
                                 sha256=sha(dest)))
    write(out/'feature_manifest.json', manifest)


def peavs_batch(videos, label, out):
    receipt = out/(label+'.raw.json')
    if receipt.exists():
        return read(receipt)
    raw = SCRATCH/'peavs'/(label+'.json')
    raw.parent.mkdir(parents=True, exist_ok=True)
    wrapper = DEPLOY/'score_peavs_official64_20261027.py'
    call([sys.executable, wrapper, *videos, '--output', raw, '--metric_batch_size', '64'],
         out/(label+'.log'))
    result = read(raw)
    work = Path(result['work_dir'])
    copy_features(work, out/label)
    # Upstream sorts feature basenames, not absolute input directories.
    sorted_videos = sorted(map(str, videos), key=lambda p: Path(p).stem)
    result.update(sorted_batch_composition=[sorted_videos[i:i+64]
                                           for i in range(0, len(videos), 64)],
                  wrapper_sha256=sha(wrapper), native_recipe='official metric.py batch_size64 unchanged')
    write(receipt, result)
    return result


def peavs():
    rows = read(RUN/'inputs.json')
    validate_rows(rows)
    out = RUN/'peavs'
    out.mkdir(parents=True, exist_ok=True)
    shifted = control_media(rows)
    raw = peavs_batch([r['video_path'] for r in rows], 'native', out)
    values = peavs_values(raw)
    scores = {r['id']: dict(id=r['id'], source=r['sample_id'], arm=r['arm'],
                           score=values[r['id']], metric_batch_size=64) for r in rows}
    write(out/'scores.json', scores)
    # Rerun only metric inference over the already-extracted native features.
    # This sensitivity addresses upstream length-mask batch-composition effects.
    sensitivity = out/'native_batch1.json'
    if not sensitivity.exists():
        work = Path(raw['work_dir'])
        code = (work/'metric.py').read_text().replace(
            "    os.system('rm -rf ./av_features_extraction/output_feats/')",
            '    pass # retain evidence')
        call([sys.executable, '-c', code, '--batch_size', '1'], out/'native_batch1.log', cwd=work)
        records = list(csv.DictReader((work/'results/results.csv').open()))
        vals = {Path(r['filename']).stem: float(r['score']) for r in records}
        assert set(vals) == set(scores)
        write(sensitivity, dict(scores=vals, metric_batch_size=1, feature_extraction_reused=True,
                                interpretation='sensitivity; never replaces primary official batch64'))
    naturals = [r for r in rows if r['arm']=='N']
    controls = {}
    batches = source_control_batches(naturals, shifted)
    for bi, videos in enumerate(batches):
        values = peavs_values(peavs_batch(videos, 'controls'+str(bi), out))
        for r in naturals[bi*21:bi*21+21]:
            key = r['id']
            baseline = values[key]
            controls[key] = dict(id=key, source=r['sample_id'], native_baseline=baseline,
                baseline_same_batch=True, batch_label='controls'+str(bi), shifts={
                    str(samples): dict(score=values[key+'_'+label], decreased=values[key+'_'+label] < baseline)
                    for samples, label in [(-6400, 'minus400ms'), (6400, 'plus400ms')]})
        write(out/'controls.json', controls)
        print(json.dumps(dict(model='peavs', native=len(scores), natural_control_pairs=len(controls))), flush=True)
    finalize(out, scores, controls)


def syncreward():
    rows = read(RUN/'inputs.json')
    validate_rows(rows)
    out = RUN/'syncreward'
    out.mkdir(parents=True, exist_ok=True)
    shifted = control_media(rows)
    raw_path = out/'all_scores.raw.jsonl'
    wrapper = DEPLOY/'score_syncreward_official_20261027.py'
    if not raw_path.exists():
        call([sys.executable, wrapper, *[r['video_path'] for r in rows],
              *[r['video_path'] for r in shifted], '--output', raw_path, '--precision', 'bf16'],
             out/'all_scores.log')
    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    lookup = {Path(r['video_path']).stem: r for r in records}
    assert len(lookup) == len(rows)+len(shifted)
    scores = {r['id']: dict(id=r['id'], source=r['sample_id'], arm=r['arm'],
                           score=float(lookup[r['id']]['reward']), precision='bf16') for r in rows}
    controls = {}
    for r in rows:
        if r['arm'] != 'N':
            continue
        key = r['id']
        baseline = scores[key]['score']
        controls[key] = dict(id=key, source=r['sample_id'], native_baseline=baseline, shifts={
            str(samples): dict(score=float(lookup[key+'_'+label]['reward']),
                               decreased=float(lookup[key+'_'+label]['reward']) < baseline)
            for samples, label in [(-6400, 'minus400ms'), (6400, 'plus400ms')]})
    write(out/'scores.json', scores)
    write(out/'controls.json', controls)
    write(out/'model_recipe.json', dict(wrapper=str(wrapper), wrapper_sha256=sha(wrapper), precision='bf16',
          strict_stage2=True, encode_batch_size=1, inputs=len(lookup)))
    finalize(out, scores, controls)


def finalize(out, scores, controls):
    assert len(scores) == 80 and len(controls) == 40
    assert len({r['source'] for r in scores.values()}) == 40
    checks = [s['decreased'] for r in controls.values() for s in r['shifts'].values()]
    assert len(checks) == 80
    write(out/'control_summary.json', dict(total=len(checks), decreased=sum(checks),
          primary_control_pass=sum(checks) >= .8*len(checks),
          recipe='actual full PCM +/-400ms, fixed copied video; no score-tuned endpoint'))
    write(out/'status.json', dict(status='COMPLETE', native_scored=len(scores), natural_control_pairs=len(controls)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['peavs', 'syncreward'])
    args = parser.parse_args()
    globals()[args.model]()
