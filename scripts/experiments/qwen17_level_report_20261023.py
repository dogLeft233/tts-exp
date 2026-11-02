"""Render saved statistics, without rerunning a model or selecting results."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def number(x, signed=False):
    return 'NA' if x is None else (f'{x:+.3f}' if signed else f'{x:.3f}')


def interval(block, key='ci95'):
    return 'NA' if block[key] is None else '[' + ', '.join(number(x, True) for x in block[key]) + ']'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run
    a, p = read(run / 'analysis.json'), read(run / 'protocol.json')
    audit = read(run / 'independent_audit.json')
    audioaudit = read(run / 'audio_independent_audit.json')
    sensitivity = read(run / 'reliability_sensitivity.json')
    if a['status'] != 'ANALYZED_COMPLETE_COVERAGE' or audit['status'] != 'PASS' or audioaudit['status'] != 'PASS':
        raise ValueError('Final report requires complete evidence and independent audit PASS')
    labels = {'N': '自然语音', 'L06': '本地 Qwen3-TTS 0.6B', 'L17': '本地 Qwen3-TTS 1.7B', 'CLOUD': '云端 Qwen Audio 3.1 Plus'}
    lines = ['# 同一10人上的 Qwen 1.7B 与电平对照', '', '## Material Passport', '',
             '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run + validate',
             '- Origin Date: 2026-10-23', '- Verification Status: ANALYZED；实际模型执行完成，保存波形、评分证据和统计独立CPU复核；未重新随机生成作独立复现。',
             '- Version Label: ' + run.name, '', '## 设计与范围', '',
             '沿用云端先导固定的10位说话人、同一自然参考录音与目标文本。该小队列此前已经分析，不能当成新的盲法确认样本。每个TTS每人预定三次合成，先在人内平均，再以10位说话人为配对统计单位；自然组每人一条。', '',
             '仅新增1.7B的30次完整self-ICL合成。使用独立1.7B环境、官方完整Base权重与同一codec，LM与codec均FP32，eager请求；沿用0.6B的原说话人seed和解码设置（temperature0.9、top_k50、top_p1、max_new_tokens512、non_streaming_mode=False）。云端、0.6B和自然原始音频及已完成的统一电平评分均复用，没有再次调用云端。', '',
             '两个音量条件来自每条同一原始波形：先ffmpeg转16kHz单声道FLOAT，raw以单位增益写PCM16，norm整段RMS调到0.0376838172675746后写PCM16。不裁剪时间、拉伸、补静音、动态限幅或改变RMS目标；仅下述8点PCM16格式饱和例外。云端内部策略仍不可见。“未归一化”指我们不额外调整下游驱动音量，不代表模型内部没有音量处理。TTS参考音频输入保持既有版本不变。', '',
             '执行偏离：attempt01的1.7B合成30/30完成、actual exit0，随后音频门控actual exit1，尚无新视频或SyncNet评分。全部100份原始波形只发现S0016的旧0.6B draw3超出PCM16范围：40.88秒、654080采样中8点，峰值约1.05174；全部norm版本均不超限。attempt02在任何新视频和评分前，将这8点作为明示的PCM16格式转换例外冻结；仅允许原SHA及指定8个索引的标准端点饱和，其余不放宽，所有原波形和失败记录保留。饱和本身的整段RMS变化约−0.000186dB，主结果仍披露该例外；另报排除整个S0016的严格9人结果。没有重新抽取1.7B。', '',
             '两组都重新生成对应口型，并用对应音轨评分。统一Wav2Lip GAN权重、3.png静态脸、batch4、nosmooth、25fps。官方完整run_pipeline（min_track25）→run_syncnet沿用前次实现、权重、AVI/MP3转码与延迟，取第一条打印track，所有track与距离保存；原画面/PTS/输入PCM remux一致性均核验。', '',
             '共200个比较单元：复用70个已归一化自然/0.6B/云端结果；新增1.7B归一化30个，以及四组raw100个，共130个新视频和官方评分。未训练。', '',
             '主比较提前固定为norm下1.7B−自然、1.7B−0.6B、1.7B−云端三项Sync-C；配对bootstrap10000次、PCG64 seed20261023，报告Bonferroni调整的98.333%百分位区间。其余raw、音量效应、交互、Sync-D/offset及资格敏感性均为描述性95%区间。区间跨零不证明等效。', '',
             '## 各组结果', '', '| 组 | 未归一化 Sync-C | 归一化 Sync-C | 归一化−未归一化（95%区间） |', '|---|---:|---:|---|']
    for model, label in labels.items():
        raw = a['by_level']['C']['raw']['groups'][model]
        norm = a['by_level']['C']['norm']['groups'][model]
        effect = a['norm_minus_raw']['C'][model]
        lines.append(f"| {label} | {number(raw['mean'])} | {number(norm['mean'])} | {number(effect['mean'], True)} {interval(effect)} |")
    lines += ['', '## 1.7B的预定主比较（归一化后）', '', '| 比较 | ΔSync-C | 98.333%区间 | 提高人数 |', '|---|---:|---|---:|']
    for label in ['L17-N', 'L17-L06', 'L17-CLOUD']:
        b = a['by_level']['C']['norm']['contrasts'][label]
        lines.append(f"| {label} | {number(b['mean'], True)} | {interval(b, 'ci98_333')} | {b['positive']}/{b['n']} |")
    lines += ['', '## TTS相对自然语音：归一化前后（描述性）', '',
              '| 比较 | 未归一化ΔC（95%区间） | 归一化ΔC（95%区间） | 分差变化：归一化−未归一化（95%区间） |', '|---|---|---|---|']
    for name in ['L06-N', 'L17-N', 'CLOUD-N', 'L17-L06', 'L17-CLOUD', 'CLOUD-L06']:
        raw = a['by_level']['C']['raw']['contrasts'][name]
        norm = a['by_level']['C']['norm']['contrasts'][name]
        change = a['contrast_norm_minus_raw']['C'][name]
        lines.append(f"| {name} | {number(raw['mean'], True)} {interval(raw)} | {number(norm['mean'], True)} {interval(norm)} | {number(change['mean'], True)} {interval(change)} |")
    lines += ['', '## 生成可靠性与敏感性分析', '',
              '主分析保留全部draw；没有按EOS、时长、ASR、试听或Sync-C排除。补充分析沿用原40人资格规则：本地输出EOS=true且整段时长/自然录音时长在[0.5,2]，在人内平均合格draw；若一人无合格draw明确列为缺失。规则在本轮合成期间、任何新SyncNet评分前另行固定，已知旧0.6B中28/30合格。它是处理后条件分析，不代替全draw主结果。云端未暴露EOS，三次原调用全部保留，重复音频不重抽。', '',
              '| 本地模型 | 合格draw | EOS缺失 | 时长比范围 |', '|---|---:|---:|---|']
    for model in ['L06', 'L17']:
        rows = [r for r in sensitivity['qualifications'] if r['model'] == model]
        lines.append(f"| {labels[model]} | {sum(r['qualified'] for r in rows)}/30 | {sum(r['EOS'] is not True for r in rows)} | {min(r['duration_ratio'] for r in rows):.3f}–{max(r['duration_ratio'] for r in rows):.3f} |")
    lines += ['', '| 补充比较（合格本地draw，norm） | ΔC | 95%区间 | 来源数 |', '|---|---:|---|---:|']
    for name in ['L17-N', 'L17-L06', 'L17-CLOUD', 'L06-N']:
        b = sensitivity['contrasts']['norm'][name]
        lines.append(f"| {name} | {number(b['mean'], True)} | {interval(b)} | {b['n']} |")
    lines += ['', '严格9人补充：排除包含8点原始PCM16饱和的整个S0016，两个音量条件均排除此人，所有模型和剩余draw完整保留；不以分数挑选。', '',
              '| 比较 | 严格9人raw ΔC（95%区间） | 严格9人norm ΔC（95%区间） |', '|---|---|---|']
    for name in ['L17-N', 'L17-L06', 'L17-CLOUD', 'L06-N', 'CLOUD-N', 'CLOUD-L06']:
        raw, norm = [sensitivity['strict9']['contrasts'][level][name] for level in ['raw', 'norm']]
        lines.append(f"| {name} | {number(raw['mean'], True)} {interval(raw)} | {number(norm['mean'], True)} {interval(norm)} |")
    lines += ['', '## 逐人值与其他端点', '',
              '| 来源 | N raw/norm | 0.6B raw/norm | 1.7B raw/norm | 云端 raw/norm |', '|---|---|---|---|---|']
    for person in a['records']:
        values = [' / '.join(number(person['values'][level][model]['C']) for level in ['raw', 'norm']) for model in labels]
        lines.append('| ' + person['sample_id'] + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '| 组 | raw Sync-D | norm Sync-D | raw offset | norm offset |', '|---|---:|---:|---:|---:|']
    for model, label in labels.items():
        vals = [a['by_level'][metric][level]['groups'][model]['mean'] for metric in ['D', 'offset'] for level in ['raw', 'norm']]
        lines.append('| ' + label + ' | ' + ' | '.join(number(v) for v in vals) + ' |')
    lines += ['', '## 结论边界与核验', '',
              '本比较检验这批来源上换模型后SyncNet评分是否变化。1.7B与0.6B虽同属开源Qwen3-TTS，仍是不同checkpoint及模型配置的联合对照，不能独立归因参数量、训练数据或某个学习任务；云端Plus更不能视为开源1.7B的同一模型。', '',
              '归一化对照同时改变驱动音频和评分音轨，因此只识别这套下游流程中的整体音量效应，未分离生成端与评价端。Sync-C更高不是独立的人类同步或嘴型准确度证明。模型、语言和人物的普遍性仍未验证。', '',
              f"独立CPU音频审计：{audioaudit['canonical_cells_checked']}条PCM由100份原始波形重采样/增益逐样点回放PASS，含70个旧norm桥接；独立评分审计：{audit['score_cells_checked']}个单元、{audit['tracks_checked']}条track、{audit['summaries_checked']}项统计PASS，最大数值误差{audit['max_arithmetic_error']:.3g}。保存的图只呈现统计结果。", '',
              '统计解释11/11检查：来源等权/不混合亚组（Simpson）；人而非draw为n（Ecological）；固定有选择的10人队列（Berkson）；资格为处理后补充（Collider）；未作诊断准确率（Base rate）；已观察队列不是新盲复现（Regression to mean）；全部draw和失败覆盖（Survivorship）；三主比较校正、其他描述（Look elsewhere）；输入/分析与补充规则事先封存（Forking paths）；模型联合差异和整体电平效应不作单机制解释（Correlation/causation）；评分未反向挑输入（Reverse causality）。', '',
              '完整数值：analysis.json、person_means.csv、reliability_sensitivity.json。协议与证据：protocol.json、root_input_seal.json、comparison_index.json、audio_independent_audit.json、independent_audit.json及各阶段actual-exit回执。完整归档位置与SHA以archive_receipt.json为准；权重为外部已部署依赖，不声称包含于实验归档。', '']
    (run / 'report.md').write_text('\n'.join(lines), encoding='utf8')
    with (run / 'all_draw_scores.csv').open('x', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        audio_checks = {(r['sample_id'], r['model'], r['level'], r['draw_index']): r for r in audioaudit['checks']}
        writer.writerow(['sample_id', 'speaker_id', 'model', 'level', 'draw_index', 'reused',
                         'Sync-C', 'Sync-D', 'offset_frames', 'source_rms', 'output_rms', 'gain',
                         'audio_sha256', 'score_sha256'])
        for row in read(run / 'comparison_index.json')['records']:
            score = read(row['score_path'])
            audio = audio_checks[row['sample_id'], row['model'], row['level'], row['draw_index']]
            writer.writerow([row[k] for k in ['sample_id', 'speaker_id', 'model', 'level', 'draw_index', 'reused']] +
                            [score[k] for k in ['sync_c', 'sync_d', 'av_offset']] +
                            [audio[k] for k in ['source_rms', 'output_rms', 'gain']] +
                            [row['audio_sha256'], row['score_sha256']])
    plot(run, a)


def plot(run, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    models = ['N', 'L06', 'L17', 'CLOUD']
    colors = {'raw': '#c37436', 'norm': '#246b9d'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for level, shift in [('raw', -.12), ('norm', .12)]:
        for i, model in enumerate(models):
            b = a['by_level']['C'][level]['groups'][model]
            y = i + shift
            axes[0].scatter(b['values'], np.full(10, y), color=colors[level], s=16, alpha=.32)
            axes[0].errorbar(b['mean'], y, xerr=[[b['mean'] - b['ci95'][0]], [b['ci95'][1] - b['mean']]],
                             fmt='o', color=colors[level], capsize=3, label=level if i == 0 else None)
    axes[0].set_yticks(range(4), ['Natural', 'Qwen 0.6B', 'Qwen 1.7B', 'Cloud Plus'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Sync-C (higher is better)')
    axes[0].set_title('Same ten speakers: means and 95% intervals')
    axes[0].legend(frameon=False)
    pairs = ['L17-N', 'L17-L06', 'L17-CLOUD']
    for i, pair in enumerate(pairs):
        b = a['by_level']['C']['norm']['contrasts'][pair]
        lo, hi = b['ci98_333']
        axes[1].scatter(b['values'], np.full(10, i), s=20, alpha=.4, color='#246b9d')
        axes[1].errorbar(b['mean'], i, xerr=[[b['mean'] - lo], [hi - b['mean']]], fmt='D',
                         color='#143c5a', capsize=4)
    axes[1].axvline(0, color='#888888', linewidth=1, linestyle='--')
    axes[1].set_yticks(range(3), ['1.7B - Natural', '1.7B - 0.6B', '1.7B - Cloud'])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Paired Sync-C difference')
    axes[1].set_title('Normalized: three primary comparisons\n98.333% intervals (three-comparison adjustment)')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color='#dddddd', alpha=.5)
        ax.set_axisbelow(True)
    fig.savefig(run / 'score_comparison.png', dpi=180)
    fig.savefig(run / 'score_comparison.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
