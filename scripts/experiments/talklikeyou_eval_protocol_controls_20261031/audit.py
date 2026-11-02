"""Independent saved-artifact checks; no new network forward."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
from scipy.io import wavfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=None)
    a = ap.parse_args()
    root = a.run
    read = lambda p: json.loads(Path(p).read_text())
    protocol = read(root/'protocol.json')
    expected = read(root/'summary.json')
    ffmpeg = '/home/data2/wangjiajun/av-evaluation/tools/bin/ffmpeg'
    checks = []
    for row in protocol['unique_rows']:
        key = row['id']
        folder = root/'items'/key
        results = read(folder/'complete.json')['cells']
        rec = dict(id=key)
        paths = [folder/(m+'_25')/'work' for m in ['original_mp4', 'canonical_pcm']]
        hashes = []
        crop_hashes = []
        for work in paths:
            def video_hash(path):
                return subprocess.check_output([ffmpeg, '-nostdin', '-v', 'error', '-i', str(path),
                    '-map', '0:v:0', '-pix_fmt', 'bgr24', '-f', 'hash', '-hash', 'sha256', '-']).decode().strip()
            hashes.append(video_hash(work/'pyavi'/key/'video.avi'))
            crops = sorted((work/'pycrop'/key).glob('0*.avi'))
            crop_hashes.append(video_hash(crops[0]) if len(crops) == 1 else None)
        rec['intermediate_pixels_equal_between_audio_modes'] = hashes[0] == hashes[1]
        rec['crop_pixels_equal_between_audio_modes'] = crop_hashes[0] == crop_hashes[1] and crop_hashes[0] is not None
        for name in ['faces.pckl', 'tracks.pckl', 'scene.pckl']:
            rec[name+'_bytes_equal_between_audio_modes'] = (
                (paths[0]/'pywork'/key/name).read_bytes() == (paths[1]/'pywork'/key/name).read_bytes())
        for mode in ['original_mp4', 'canonical_pcm']:
            target = folder/(mode+'_25')
            sr_a, actual = wavfile.read(target/'work/pytmp'/key/'audio.wav')
            sr_b, diagnostic = wavfile.read(target/'crop_decoded.wav')
            assert sr_a == sr_b == 16000
            rec[mode+'_actual_network_pcm_matches_crop_decoder'] = bool(np.array_equal(actual, diagnostic))
            assert rec[mode+'_actual_network_pcm_matches_crop_decoder']
        errors = []
        for cell, scalar in results.items():
            if scalar['status'] != 'PASS':
                continue
            matrix = np.load(folder/cell/'distances.npy').astype(np.float64)
            curve = np.sum(matrix, axis=0)/matrix.shape[0]
            d = float(np.min(curve))
            c = float(sorted(curve)[15]-d)
            assert abs(c-scalar['C']) < 1e-5 and abs(d-scalar['D']) < 1e-5
            assert scalar['offset'] == 15-int(np.argmin(curve))
            errors.append(abs(c-scalar['C']))
        rec['fp64_C_max_error'] = max(errors)
        checks.append(rec)
    all_results = {r['id']: read(root/'items'/r['id']/'complete.json')['cells'] for r in protocol['unique_rows']}
    for alias, original in protocol['aliases'].items():
        all_results[alias] = all_results[original]
    # Independently reconstruct common sources and all source-level deltas.
    common = []
    for source in sorted({r['source'] for r in protocol['rows']}):
        group = [r for r in protocol['rows'] if r['source'] == source]
        if not all(all_results[r['id']][c]['status'] == 'PASS' for r in group for c in expected['coverage']):
            continue
        common.append(source)
    assert common == expected['common_four_cell_source_ids']
    numeric_errors = []
    inference_checks = []

    def check_inference(differences, target, label):
        d = np.asarray(differences, dtype=np.float64)
        rng = np.random.default_rng(20261031)
        samples = rng.choice(d, size=(10000, len(d)), replace=True)
        ci = np.percentile(np.mean(samples, axis=1), [2.5, 97.5])
        # Enumerate the signed-sum distribution by recursion rather than bit masks.
        sums = np.zeros(1)
        for value in d:
            sums = np.concatenate([sums+value, sums-value])
        p = float(np.mean(np.abs(sums/len(d)) >= abs(d.mean())-1e-14))
        assert np.allclose(ci, target['ci95'], atol=1e-12, rtol=0), label
        assert p == target['exact_signflip_p'], (label, p, target['exact_signflip_p'])
        inference_checks.append(label)
    for stratum, select in [('all', lambda r: True), ('strict_reference', lambda r: r['strict_reference']),
                            ('extended_reference', lambda r: not r['strict_reference'])]:
        eligible = [s for s in common if select(next(r for r in protocol['rows'] if r['source'] == s))]
        if not eligible:
            continue
        for cell in expected['coverage']:
            differences = []
            for source in eligible:
                group = [r for r in protocol['rows'] if r['source'] == source]
                natural = [all_results[r['id']][cell]['C'] for r in group if r['arm']=='N']
                tts = [all_results[r['id']][cell]['C'] for r in group if r['arm']!='N']
                differences.append(sum(tts)/len(tts)-natural[0])
            actual = sum(differences)/len(differences)
            delta = expected['strata'][stratum]['cells'][cell]['C']['mean']
            numeric_errors.append(abs(actual-delta))
            assert abs(actual-delta) < 1e-12
            check_inference(differences, expected['strata'][stratum]['cells'][cell]['C'], stratum+':'+cell)
        group = [r for r in read(root/'paired_common_four_cells.json') if r['source'] in eligible]
        contrasts = dict(threshold_original=('original_mp4_100','original_mp4_25'),
            threshold_pcm=('canonical_pcm_100','canonical_pcm_25'),
            audio_original_minus_pcm_at100=('original_mp4_100','canonical_pcm_100'))
        pvalues = []
        for name, (left, right) in contrasts.items():
            differences = [r['cells'][left]['C']['delta']-r['cells'][right]['C']['delta'] for r in group]
            target = expected['strata'][stratum]['protocol_effects'][name]
            check_inference(differences, target, stratum+':'+name)
            pvalues.append((name, target['exact_signflip_p']))
        previous = 0.
        for index, (name, p) in enumerate(sorted(pvalues, key=lambda x: x[1])):
            previous = min(1., max(previous, p*(len(pvalues)-index)))
            assert previous == expected['strata'][stratum]['protocol_effects'][name]['holm3_p']
    stages = [read(p) for p in (root/'stages').glob('*.json')]
    assert all(s['returncode']==0 for s in stages)
    result = dict(status='PASS', no_new_forward=True, unique_items=len(checks), stage_count=len(stages),
        all_stage_exit0=True, common_source_ids=common, independent_source_delta_max_error=max(numeric_errors),
        bootstrap_exact_signflip_checks=inference_checks,
        visual_invariance={k: sum(r[k] for r in checks) for k in checks[0] if k not in ['id', 'fp64_C_max_error']},
        fp64_C_max_error=max(r['fp64_C_max_error'] for r in checks), checks=checks,
        provenance='Initial producer11201 completed27 items then resource-gated before item28 scoring; revision02 resumes only incomplete stages, waiting for GPU availability. Auxiliary PID12360 was refused before a forward. See resume_gate_preflight.json and terminal_receipt.json.')
    (a.output or root/'independent_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
