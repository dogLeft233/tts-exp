"""Minimise upstream PEAVS padding dependence and rescore native80.

Equal-length truncation below is diagnostic only, never a replacement score.
Native rescoring executes the unchanged upstream metric at default batch64.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

import peavs_official_parity_20261027 as parity


def features_from_raw(folder):
    mapping = {}
    for raw in folder.glob('batch*.raw.json'):
        data = parity.read(raw)
        features = Path(data['work_dir']) / 'av_features_extraction/output_feats'
        for row in data['scores']:
            name = Path(row['filename']).stem
            mapping[name] = dict(audio=features / 'vggish' / (name + '_vggish.npy'),
                                 video=features / 'i3d_feats' / (name + '_i3d.npy'))
    return mapping


def main(out):
    torch.set_num_threads(4)
    sys.path.insert(0, str(parity.SOURCE))
    import utils
    import net
    args = SimpleNamespace(device='cpu', seed=0, output_dim=1, attn_dropout=.1, relu_dropout=.1,
        embed_dropout=.25, res_dropout=.1, out_dropout=.2, layers=3, num_heads=8, attn_mask=True)
    utils.set_deterministic(0)
    model = net.av_sync_Model.SyncMetric(args)
    model.load_state_dict(torch.load(parity.SOURCE / 'metric_weights/final_model.pt', map_location='cpu', weights_only=True), strict=True)
    model.eval()
    mapping = features_from_raw(parity.OLD / 'peavs')
    target = 'native__BAC009S0008W0121_N_0'
    companion = 'native__BAC009S0002W0122_N_0'
    arrays = {k: [np.load(mapping[k]['audio']), np.load(mapping[k]['video'])] for k in [target, companion]}

    def score(names, equalize=False):
        audio, video = [], []
        for name in names:
            a, v = arrays[name]
            if equalize:
                a, v = a[:min(len(a), len(v))], v[:min(len(a), len(v))]
            audio.append(a)
            video.append(v)
        data = utils.AudVidSet(audio, video)
        batch = utils.collate_fn_padd([data[i] for i in range(len(names))])
        with torch.inference_mode():
            values = torch.clamp(model(batch[0], batch[1], batch[2], batch[3]) * 4 + 1, 1, 5)
        return float(values[0])

    alone = score([target])
    mixed = score([target, companion])
    trimmed_alone = score([target], True)
    trimmed_mixed = score([target, companion], True)
    assert abs(alone-mixed) > .1, 'Minimal padding dependence no longer reproduced'
    assert abs(trimmed_alone-trimmed_mixed) < 1e-5, 'Feature-length diagnostic did not isolate padding'
    minimal = dict(target=target, audio_shape=list(arrays[target][0].shape),
        video_shape=list(arrays[target][1].shape), original_alone=alone, original_with_companion=mixed,
        batch_dependency=abs(alone-mixed), equalized_diagnostic_alone=trimmed_alone,
        equalized_diagnostic_with_companion=trimmed_mixed, diagnostic_not_official_score=True,
        upstream_duration_bug='AudVidSet.__getitem__: cur_dur_v = len(cur_aud)',
        mechanism='When audio has an extra feature, the same video zero padding can be marked valid in a mixed batch')
    (out / 'peavs_minimal_batch_dependency.json').write_text(json.dumps(minimal, indent=2))
    print('MINIMAL', json.dumps(minimal), flush=True)
    cloud = parity.ROOT / 'runs/cloud_av_preference20_20261027_attempt05'
    paired = parity.read(cloud / 'paired_scores.json')
    selected = {k:v for k,v in mapping.items() if k.startswith('native__') and k.endswith('_N_0')}
    selected.update(features_from_raw(cloud / 'peavs'))
    assert len(selected) == 80
    target_dir = out / 'peavs_native80'
    target_dir.mkdir(exist_ok=True)
    scores = parity.score_original(target_dir, 64, selected)
    assert set(scores) == set(selected)
    pairs = []
    for row in paired:
        nk = 'native__' + row['source'] + '_N_0'
        pairs.append(dict(id=row['id'], source=row['source'], strict_reference=row['strict_reference'],
            natural=scores[nk], tts=scores[row['id']], delta=scores[row['id']]-scores[nk],
            previous_delta=row['delta']['peavs'], delta_syncC=row['delta']['SyncC']))
    subsets = {}
    for name, rows in [('all20', pairs), ('same_reference12', [r for r in pairs if r['strict_reference']])]:
        sources = sorted({r['source'] for r in rows})
        mean = lambda key: float(np.mean([np.mean([r[key] for r in rows if r['source']==s]) for s in sources]))
        subsets[name] = dict(natural=mean('natural'), tts=mean('tts'), delta=mean('delta'),
            pairs=len(rows), tts_higher=sum(r['delta']>0 for r in rows),
            conflicts_vs_syncC=sum(r['delta']*r['delta_syncC']<0 for r in rows),
            preference_flips_vs_batch1=sum((r['delta']>0)!=(r['previous_delta']>0) for r in rows))
    report = dict(status='DEFAULT64_NATIVE_RECOMPUTED_WITH_UPSTREAM_PADDING_BUG_RETAINED',
        minimal=minimal, official_batch_size=64, sorted_input_names=sorted(selected),
        maximum_score_change_vs_previous=max(abs(scores[r['id']]-r['T']['peavs']) for r in paired),
        subsets=subsets, pairs=pairs)
    (out / 'peavs_native_default64_reanalysis.json').write_text(json.dumps(report, indent=2))
    print('NATIVE64', json.dumps({k:v for k,v in report.items() if k not in ['pairs','sorted_input_names']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
