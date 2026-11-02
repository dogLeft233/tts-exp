"""Exploratory read-only outlier/ASR description; never changes the main cohort."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'runs/qwen_nrt_expansion40_20261022_attempt01'
ROLES = ('T1', 'T2', 'T3')


def read(path):
    return json.loads(path.read_text())


def main():
    analysis = read(RUN / 'analysis.json')
    protocol = read(RUN / 'protocol.json')
    new_asr = read(RUN / 'asr/results.json')['records']
    old_asr = read(ROOT / 'runs/qwen_tts_decline_audit_20261022_attempt01/asr_attempt02/results.json')['records']
    asr = {(r['sample_id'], r['role']): r for r in old_asr + new_asr}
    samples = {r['sample_id']: r for r in protocol['samples']}
    reliability = {(r['sample_id'], r['role']): r for r in analysis['all120_draws_reliability']}
    vad = read(RUN / 'asr/vad_results.json')['records']
    old_vad_path = ROOT / 'runs/qwen_tts_decline_audit_20261022_attempt01/cap_vad_sensitivity/results.json'
    if old_vad_path.exists():
        vad += read(old_vad_path)['records']
    vad = {(r['sample_id'], r['role']): r for r in vad}
    records = []
    for record in analysis['records']:
        sid = record['sample_id']
        row = dict(sample_id=sid, cohort=record['cohort'], transcript=samples[sid]['transcript'], policies={})
        for policy, cells in record['policies'].items():
            draws = []
            for role in ROLES:
                content = asr[sid, role]
                item = dict(role=role, C=cells[role]['C'], TN=cells[role]['C']-cells['N']['C'],
                            TR=cells[role]['C']-cells['R']['C'], reliability=reliability[sid, role],
                            raw_ASR=content['raw'], CER=content['CER'])
                if (sid, role) in vad:
                    item['exploratory_VAD_ASR'] = vad[sid, role]
                draws.append(item)
            t = float(np.mean([d['C'] for d in draws]))
            row['policies'][policy] = dict(N=cells['N']['C'], R=cells['R']['C'], T=t,
                                          TN=t-cells['N']['C'], TR=t-cells['R']['C'], draws=draws)
        row['N_ASR'] = asr[sid, 'N']['raw']
        row['R_ASR'] = asr[sid, 'R']['raw']
        records.append(row)
    summaries = {}
    lines = ['# 扩样后的下降样本与影响量（探索性描述）', '',
             '不删除任何主分析样本；极端排名、ASR和留一法均为描述性诊断。ASR不是人工真值，数字表记可能造成假差异。', '',
             '| 队列/支持/比较 | 均值 | 提高人数 | 提高draw | 留一后均值范围 | 单人最大均值改变量 |',
             '|---|---:|---:|---:|---|---:|']
    for cohort in ('new30', 'all40', 'pilot10'):
        rows = [r for r in records if cohort == 'all40' or r['cohort'] == cohort]
        summaries[cohort] = {}
        for policy in ('common', 'native_full', 'equal_count_spread'):
            summaries[cohort][policy] = {}
            for key, contrast in [('TN', 'T-N'), ('TR', 'T-R')]:
                x = np.array([r['policies'][policy][key] for r in rows], dtype=float)
                mean = float(x.mean())
                loo = np.array([(float(x.sum())-v)/(len(x)-1) for v in x])
                expected = analysis['cohorts'][cohort]['summaries'][policy]['C']['contrasts'][contrast]
                assert abs(mean-expected['mean']) < 1e-12 and int((x > 0).sum()) == expected['positive']
                draws = [d[key] for r in rows for d in r['policies'][policy]['draws']]
                ordered = sorted(rows, key=lambda r: r['policies'][policy][key])
                result = dict(n_speakers=len(x), n_draws=len(draws), mean=mean, positive_speakers=int((x>0).sum()),
                              positive_draws=sum(d>0 for d in draws), leave_one_out_min=float(loo.min()),
                              leave_one_out_max=float(loo.max()), max_absolute_single_person_mean_change=float(abs(loo-mean).max()),
                              negative_contribution_sum=float(x[x<0].sum()), positive_contribution_sum=float(x[x>0].sum()),
                              largest_five_decreases=[r['sample_id'] for r in ordered[:5]],
                              largest_five_increases=[r['sample_id'] for r in ordered[-5:][::-1]])
                summaries[cohort][policy][key] = result
                lines.append(f"| {cohort}/{policy}/{contrast} | {mean:+.3f} | {result['positive_speakers']}/{len(x)} | {result['positive_draws']}/{len(draws)} | [{loo.min():+.3f}, {loo.max():+.3f}] | {result['max_absolute_single_person_mean_change']:.3f} |")
    lines += ['', '## 新30人共同窗口下降最多与提高最多的各5人', '',
              '只按预定主要支持的T−N排序，正负两端同时展示。高N基线与T−N的负相关含数学耦合，不能据此证明天花板或均值回归。']
    for title, key in [('下降最多', 'largest_five_decreases'), ('提高最多', 'largest_five_increases')]:
        lines += ['', '### ' + title, '']
        for sid in summaries['new30']['common']['TN'][key]:
            row = next(r for r in records if r['sample_id'] == sid)
            c = row['policies']['common']
            lines += [f"- {sid}：T−N {c['TN']:+.3f}，T−R {c['TR']:+.3f}；N/R/T={c['N']:.3f}/{c['R']:.3f}/{c['T']:.3f}。", f"  语料文字：{row['transcript']}", f"  N转写：{row['N_ASR']}；R转写：{row['R_ASR']}"]
            for d in c['draws']:
                r = d['reliability']
                lines.append(f"  - {d['role']}：T−N {d['TN']:+.3f}；EOS={r['EOS']}，时长比={r['duration_ratio']:.3f}，CER={d['CER']:.3f}；转写：{d['raw_ASR']}")
    lines += ['', '本诊断不把低分直接等同于生成差，也不把准确ASR等同于韵律或自然度正常；与更强TTS模型的比较仍未进行。']
    out = dict(status='EXPLORATORY_DESCRIPTION_WITH_PRIMARY_ARITHMETIC_CROSSCHECK',
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), summaries=summaries, records=records,
               no_primary_sample_removed=True, human_listening_performed=False)
    with (RUN / 'decline_summary.json').open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    with (RUN / 'decline_summary.md').open('x') as f:
        f.write('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
