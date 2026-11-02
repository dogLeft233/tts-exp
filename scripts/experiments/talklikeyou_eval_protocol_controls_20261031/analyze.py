"""Source-paired analysis on fixed, jointly measurable factorial supports."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

CELLS = ['original_mp4_25', 'original_mp4_100', 'canonical_pcm_25', 'canonical_pcm_100']


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def stats(values):
    d = np.asarray(values, dtype=float)
    assert d.ndim == 1 and len(d) and np.isfinite(d).all()
    rng = np.random.default_rng(20261031)
    ci = np.quantile(d[rng.integers(0, len(d), (10000, len(d)))].mean(1), [.025, .975])
    exceed = 0
    assert len(d) <= 20
    for begin in range(0, 2**len(d), 8192):
        patterns = np.arange(begin, min(begin+8192, 2**len(d)), dtype=np.uint64)
        signs = ((patterns[:, None] >> np.arange(len(d), dtype=np.uint64)) & 1).astype(float)*2-1
        exceed += int(np.sum(abs(signs@d/len(d)) >= abs(d.mean())-1e-14))
    return dict(n_sources=len(d), mean=float(d.mean()), ci95=ci.tolist(),
        positive=int(np.sum(d > 0)), negative=int(np.sum(d < 0)), ties=int(np.sum(d == 0)),
        exact_signflip_p=exceed/(2**len(d)))


def holm(ps):
    p = np.asarray(ps)
    order = np.argsort(p, kind='stable')
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(p[order]*np.arange(len(p), 0, -1)))
    return out.tolist()


def make_pairs(rows, lookup, required):
    pairs, missing = [], []
    for source in sorted({r['source'] for r in rows}):
        group = [r for r in rows if r['source'] == source]
        natural = [r for r in group if r['arm'] == 'N']
        tts = [r for r in group if r['arm'] != 'N']
        assert len(natural) == 1 and len(tts) == 3
        absent = [f"{r['id']}:{c}" for r in group for c in required
                  if lookup[r['id']]['cells'][c]['status'] != 'PASS']
        if absent:
            missing.append(dict(source=source, unmeasurable=absent))
            continue
        cells = {}
        for cell in required:
            endpoint = {}
            for metric in ['C', 'D', 'D0', 'offset', 'B']:
                n = lookup[natural[0]['id']]['cells'][cell][metric]
                t = float(np.mean([lookup[r['id']]['cells'][cell][metric] for r in tts]))
                endpoint[metric] = dict(N=n, T=t, delta=t-n)
            cells[cell] = endpoint
        pairs.append(dict(source=source, speaker=natural[0]['speaker'],
                          strict_reference=natural[0]['strict_reference'], cells=cells))
    return pairs, missing


def summarize_pairs(pairs, cells):
    output = {}
    for cell in cells:
        output[cell] = {}
        for metric in ['C', 'D', 'D0', 'offset', 'B']:
            records = [r['cells'][cell][metric] for r in pairs]
            output[cell][metric] = dict(stats([r['delta'] for r in records]),
                N_mean=float(np.mean([r['N'] for r in records])),
                T_mean=float(np.mean([r['T'] for r in records])))
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    root = args.run
    assert read(root/'status.json')['status'] == 'FORWARDS_COMPLETE'
    protocol = read(root/'protocol.json')
    results = read(root/'results.json')
    lookup = {r['id']: r for r in results}
    for alias, original in protocol['aliases'].items():
        lookup[alias] = lookup[original]
    common, excluded = make_pairs(protocol['rows'], lookup, CELLS)
    audio_pairs, audio_excluded = make_pairs(protocol['rows'], lookup,
                                            ['original_mp4_25', 'canonical_pcm_25'])
    contrasts = {
        'threshold_original': ('original_mp4_100', 'original_mp4_25'),
        'threshold_pcm': ('canonical_pcm_100', 'canonical_pcm_25'),
        'audio_original_minus_pcm_at100': ('original_mp4_100', 'canonical_pcm_100'),
    }
    summary = dict(status='COMPLETE', common_four_cell_sources=len(common),
        common_four_cell_source_ids=[r['source'] for r in common], excluded_sources=excluded,
        common25_audio_sources=len(audio_pairs), common25_audio_excluded=audio_excluded,
        strata={}, audio25_strata={}, transport={}, precision_audit={}, coverage={})
    for label, select in [('all', lambda r: True), ('strict_reference', lambda r: r['strict_reference']),
                          ('extended_reference', lambda r: not r['strict_reference'])]:
        group = [r for r in common if select(r)]
        if group:
            effects = {name: stats([r['cells'][a]['C']['delta']-r['cells'][b]['C']['delta']
                                    for r in group]) for name, (a, b) in contrasts.items()}
            for (name, result), adjusted in zip(effects.items(), holm([r['exact_signflip_p'] for r in effects.values()])):
                result['holm3_p'] = adjusted
            summary['strata'][label] = dict(n_sources=len(group),
                cells=summarize_pairs(group, CELLS), protocol_effects=effects)
        group = [r for r in audio_pairs if select(r)]
        if group:
            effect = stats([r['cells']['original_mp4_25']['C']['delta']-
                            r['cells']['canonical_pcm_25']['C']['delta'] for r in group])
            summary['audio25_strata'][label] = dict(n_sources=len(group),
                cells=summarize_pairs(group, ['original_mp4_25', 'canonical_pcm_25']),
                audio_effect=effect)
    baseline = {r['id']: r for r in read(Path(protocol['parent'])/'syncnet/scores.json')}
    errors, bridge, threshold_errors, reused = [], [], [], []
    for row in protocol['unique_rows']:
        key = row['id']
        fresh = lookup[key]['cells']['canonical_pcm_25']
        if fresh['status'] == 'PASS':
            bridge.append(dict(id=key, C_difference=fresh['C']-baseline[key]['C'],
                               D_difference=fresh['D']-baseline[key]['D']))
        for cell in CELLS:
            rec = lookup[key]['cells'][cell]
            if rec['status'] != 'PASS':
                continue
            z = np.load(root/'items'/key/cell/'distances.npy').astype(np.float64)
            curve = z.mean(0)
            c = np.median(curve)-curve.min()
            errors.append(max(abs(c-rec['C']), abs(curve.min()-rec['D']), abs(curve[15]-rec['D0'])))
            if rec['forward_reused_from']:
                prior = rec['forward_reused_from']
                a = root/'items'/key/cell/'work/pycrop'/key/'00000.avi'
                b = root/'items'/key/prior/'work/pycrop'/key/'00000.avi'
                assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
                assert np.array_equal(np.load(root/'items'/key/prior/'distances.npy'), z.astype(np.float32))
                reused.append(f'{key}:{cell}')
        for mode in ['original_mp4', 'canonical_pcm']:
            a, b = [lookup[key]['cells'][mode+'_'+str(t)] for t in [25, 100]]
            if a['status'] == b['status'] == 'PASS':
                threshold_errors.append(max(abs(a[k]-b[k]) for k in ['C', 'D', 'D0', 'offset']))
    assert max(errors) < 1e-5
    write(root/'baseline_bridge.json', bridge)
    summary['precision_audit'] = dict(fp64_scalar_max_abs_error=max(errors),
        baseline_bridge_max_abs_C=max(abs(r['C_difference']) for r in bridge),
        threshold_scalar_max_abs_difference=max(threshold_errors),
        crop_identical_forward_reuse=len(reused), crop_reuse_ids=reused)
    for mode in ['original_mp4', 'canonical_pcm']:
        records = [read(root/'items'/r['id']/(mode+'_25')/'transport.json') for r in protocol['unique_rows']]
        codec_counts = {}
        for record in records:
            for s in record['avi_probe']['streams']:
                key = s['codec_type']+':'+s['codec_name']
                codec_counts[key] = codec_counts.get(key, 0)+1
        transport = dict(codec_counts=codec_counts, n_unique_clips=len(records), comparisons={})
        for comparison in ['canonical_to_input', 'input_to_intermediate', 'canonical_to_intermediate', 'canonical_to_official_audio', 'diagnostic_vs_official_audio']:
            records2 = [r[comparison] for r in records]
            ranges = {}
            for field in ['lag_samples', 'lag_ms', 'aligned_cosine', 'aligned_gain', 'zero_lag_rmse_pcm_units', 'aligned_snr_db', 'length_difference_samples']:
                values = [r[field] for r in records2 if r[field] is not None]
                ranges[field] = dict(min=float(min(values)), median=float(np.median(values)), max=float(max(values))) if values else None
            transport['comparisons'][comparison] = dict(exact_equal=sum(r['exact_equal'] for r in records2),
                boundary_hits=sum(r['lag_hit_boundary'] for r in records2), ranges=ranges)
        summary['transport'][mode] = transport
    for cell in CELLS:
        summary['coverage'][cell] = dict(unique_pass=sum(r['cells'][cell]['status']=='PASS' for r in results),
                                        logical_pass=sum(lookup[r['id']]['cells'][cell]['status']=='PASS' for r in protocol['rows']))
    write(root/'paired_common_four_cells.json', common)
    write(root/'paired_audio25.json', audio_pairs)
    write(root/'summary.json', summary)
    rows = []
    for pair in common:
        for cell in CELLS:
            rows.append(dict(source=pair['source'], strict_reference=pair['strict_reference'], cell=cell,
                             **pair['cells'][cell]['C']))
    with (root/'paired_common_four_cells.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'strict_reference', 'cell', 'N', 'T', 'delta'])
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 中文TalkLikeYou：固定视频的SyncNet门槛与音轨对照', '',
        '✅ 完成。原80逻辑条目／41唯一生成，未重生成视频或TTS。Sync-C差只解释评价器结果。', '',
        f'四格共同完整来源：{len(common)}/20；原12同句参考分层单列。四格都可测才纳入，未按分数筛样。',
        '每来源三次既有TTS逻辑draw先平均，再减自然分数；重复draw不视为独立样本。', '',
        '100/25门槛效应定义为ΔC100−ΔC25；音轨效应定义为ΔC原MP4−ΔC规范PCM。',
        '95%区间为来源bootstrap10000；双侧exact signflip依赖对称/可交换假设，非随机处理试验。',
        '三个协议效应每分层单独Holm3；单格TTS−自然的边际检验不替代原报告Holm9。', '']
    for label, group in summary['strata'].items():
        lines += [f'## 四格共同支持：{label}（{group["n_sources"]}来源）', '',
                  '|条件|自然C|TTS C|T−N|95% CI|T较高|', '|---|---:|---:|---:|---|---:|']
        for cell in CELLS:
            v = group['cells'][cell]['C']
            lines.append(f'|{cell}|{v["N_mean"]:.3f}|{v["T_mean"]:.3f}|{v["mean"]:+.3f}|[{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]|{v["positive"]}/{v["n_sources"]}|')
        lines += ['', '|协议效应|ΔΔC|95% CI|Holm3 p|', '|---|---:|---|---:|']
        for name, v in group['protocol_effects'].items():
            lines.append(f'|{name}|{v["mean"]:+.6f}|[{v["ci95"][0]:+.6f}, {v["ci95"][1]:+.6f}]|{v["holm3_p"]:.6f}|')
        lines.append('')
    lines += ['## 音轨对照：共同25门槛支持（补充）', '']
    for label, group in summary['audio25_strata'].items():
        a, b = [group['cells'][c]['C'] for c in ['original_mp4_25', 'canonical_pcm_25']]
        e = group['audio_effect']
        lines.append(f'- {label}，n={group["n_sources"]}：原MP4 ΔC{a["mean"]:+.3f}，规范PCM{b["mean"]:+.3f}；效应{e["mean"]:+.6f}，95%[{e["ci95"][0]:+.6f},{e["ci95"][1]:+.6f}]，p={e["exact_signflip_p"]:.6f}。')
    lines += ['', '## 中间AVI与审计', '', '完整codec、波形延迟/相关/误差范围见summary.json transport；逐片见items/*/*/transport.json。诊断对齐从未用于评分。', '',
        json.dumps(summary['precision_audit'], ensure_ascii=False), '',
        '原始MP4及规范WAV只读绑定到父run的SHA；官方源码与权重完整SHA在protocol.json。保留所有官方中间AVI、track、检测及31lag矩阵。',
        '原producer11201完成27条后在第28条评分前因共享GPU资源门停下；revision02等待资源并复用已核验完成的pipeline。一次辅助重复启动在fresh-run断言处拒绝；失败与修订凭据保留，科学单元退出状态见stages与完成凭据。',
        '原HDTF暂停实验未启动。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k in ['common_four_cell_sources', 'coverage', 'precision_audit']}, indent=2))


if __name__ == '__main__':
    main()
