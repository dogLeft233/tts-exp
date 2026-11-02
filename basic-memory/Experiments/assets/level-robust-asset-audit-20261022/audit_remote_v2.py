"""Read-only asset checks; no generation, scoring, model loading or training."""
from pathlib import Path
import array
import hashlib
import json
import math
import shutil
import subprocess
import sys

ROOT = Path('/home/wjj/tts-audio/tts-exp')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def file_info(path, hash_body=False):
    p = Path(path)
    d = {'path': str(p), 'exists': p.is_file()}
    if p.is_file():
        d['bytes'] = p.stat().st_size
        if hash_body:
            d['sha256'] = sha(p)
    return d

def probe(path):
    cp = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration,size:stream=codec_type,codec_name,width,height,sample_rate,channels,avg_frame_rate,start_time,duration',
        '-of', 'json', str(path)], capture_output=True, text=True, timeout=15)
    return {'rc': cp.returncode, 'data': json.loads(cp.stdout) if cp.returncode == 0 else cp.stderr[-300:]}

out = {'scope': 'metadata, targeted hashes and waveform statistics only; no model inference', 'repo': str(ROOT)}
manifest = ROOT / 'data/dataset_samples/video_manifest_250.json'
rs = json.loads(manifest.read_text())['records']
out['manifest'] = file_info(manifest, True)
lrs = sorted([r for r in rs if r['dataset'] == 'lrs3'], key=lambda r: (Path(r['video_local_path']).parent.name, r['stem']))
eligible = []
seen = set()
for r in lrs:
    p = ROOT / r['video_local_path']
    pr = probe(p)
    ss = pr['data'].get('streams', []) if pr['rc'] == 0 else []
    vv = [s for s in ss if s['codec_type'] == 'video']
    aa = [s for s in ss if s['codec_type'] == 'audio']
    if vv and aa and float(vv[0].get('duration', 0)) >= 6 and float(aa[0].get('duration', 0)) >= 6:
        source = p.parent.name
        if source not in seen:
            eligible.append({'source_group': source, 'stem': r['stem'], 'path': str(p), 'probe': pr['data']})
            seen.add(source)
out['lrs3_six_second_source_groups'] = len(eligible)
out['candidates'] = []
for r in eligible:
    p = Path(r['path'])
    cp = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(p), '-ss', '1', '-t', '5', '-map', '0:a:0',
        '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'], capture_output=True, timeout=20)
    d = dict(r)
    d.update(file_info(p, True))
    d['selection'] = 'all sorted source groups with AV duration >=6s; waveform-only preflight, no mouth/score selection'
    d['target_seconds'] = [1, 6]
    d['reference_candidate_seconds'] = 0
    d['speaker_identity_verified'] = False
    d['already_used_in_multiset'] = True
    if cp.returncode == 0:
        a = array.array('f'); a.frombytes(cp.stdout)
        if sys.byteorder != 'little': a.byteswap()
        n = len(a); mu = sum(a)/n; rms = math.sqrt(sum(v*v for v in a)/n)
        std = math.sqrt(sum((v-mu)**2 for v in a)/n)
        pk = max(abs(v) for v in a)
        d['audio_preflight'] = {'samples': n, 'expected_samples': 80000, 'mean': mu, 'rms': rms, 'peak': pk,
            'peak_plus6db': pk*10**(6/20), 'finite': all(math.isfinite(v) for v in a),
            'rms_target_peaks': {str(k):pk*(0.0376838172675746*k)/rms for k in [0.5,1,2]},
            'demean_target_peak':max(abs(v-mu) for v in a)*0.0376838172675746/std}
    else:
        d['audio_preflight'] = {'error': cp.stderr.decode(errors='replace')[-300:]}
    out['candidates'].append(d)

grid = ROOT / 'runs/grid_fixed24_acquisition_20260927/manifest.json'
g = json.loads(grid.read_text())
out['grid'] = {'manifest':file_info(grid,True),'eval_visual_analysis_authorized':g['eval_visual_analysis_authorized'],
    'rows':[{'label':r['speaker_label'],'split':r['split'],'video':file_info(ROOT/r['mpg_path']),
        'audio':file_info(ROOT/r['pcm_path']),'declared_duration':r['pcm_duration_seconds']} for r in g['rows']],
    'note':'only stat on eval files; no eval body, waveform or visual analysis'}
lm = ROOT / 'checkpoints/mediapipe/face_landmarker.task'
out['landmarker'] = file_info(lm, True)
out['runtime_commands'] = {'ffmpeg':shutil.which('ffmpeg'),'ffprobe':shutil.which('ffprobe')}
out['disk_home'] = dict(zip(['total','used','free'],shutil.disk_usage(ROOT)))
print(json.dumps(out,ensure_ascii=False))
