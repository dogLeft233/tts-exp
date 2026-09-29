"""Read sealed summaries and render the fixed prototype trial report; no new scores."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/tts_fixed_generator_phone_prototype_20260927'
AUDIT = ROOT / 'runs/tts_fixed_generator_phone_prototype_independent_audit_20260927'


def read(path):
    path = Path(path)
    return json.loads(gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(stats):
    lo, hi = stats['ci99']
    return f"{stats['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    assert not (RUN / 'report.md').exists()
    summary = read(RUN / 'summary.json.gz')
    events = read(RUN / 'event_summary.json.gz')
    analysis = read(RUN / 'analysis.json')
    protocol = read(RUN / 'protocol.json')
    binding = read(RUN / 'independent_validation.json')
    assert binding['status'] == 'PASS'
    for path, digest in binding['receipts'].items():
        assert sha(path) == digest and read(path)['status'] == 'PASS'
    lines = [
        '# FIXED 生成器音素原型凸组合：结果', '',
        '2026-09-27。状态 concluded；先验、旧 baseline 精确桥接、现场身份/处理校准、LOSO 拟合及最终独立数值复核均通过。', '',
        '## 1. 主问题与判据', '',
        '固定原 FIXED 评价音频，在 Wav2Lip post-ReLU 512D 音频瓶颈以 λ=.5 混入 cal-only LOSO 的 N/T phone 或 global 原型，实际重新生成视频。'
        '主支持为旧共同 71 clips / 15 speakers；全 74 eval、两臂、四处理共 592 cells 全部保留。', '',
        '主判据须自然臂 phoneT−phoneN 与 phoneT−原 baseline 的 99CI 下界同时大于 0。'
        '仅胜 phoneN 表示替换损伤较小；仅胜 baseline 表示模板混合收益，不能称 TTS 特异收益。'
        '即使双门通过，名称也限定为“固定自然音轨下，生成路径的 Sync-C 收益”。', '',
    ]
    criteria = analysis['criteria']
    lines += ['| 冻结判据 | 通过 |', '|:--|:--|']
    for key, val in criteria.items():
        lines.append(f'| {key} | {val} |')
    natural_template = summary['raw/guard20/common71/C/phone/TminusN_template/N']
    natural_baseline = summary['raw/guard20/common71/C/response/phoneT/N']
    lines += ['', f"自然臂 phoneT−phoneN：{value(natural_template)}；phoneT−baseline：{value(natural_baseline)}。"]
    if not criteria['natural_generator_gain_both']:
        lines.append('冻结的自然生成收益双门未通过。这只约束本次 λ=.5 原型凸组合，不能据此排除来源均值中所有可迁移信息。')
    global_template = summary['raw/guard20/common71/C/global/TminusN_template/N']
    global_baseline = summary['raw/guard20/common71/C/response/globalT/N']
    lines.append(f"自然臂 globalT−globalN：{value(global_template)}；globalT−baseline：{value(global_baseline)}。两者必须并读，不能将相对受损对照的优势命名为生成收益。")
    lines += ['', '所有数值为 speaker 等权均值 [99% bootstrap CI]；20,000 次、seed 20260926，各比较无 FWER 校正。CI 条件于冻结 cal 原型。', '',
              '## 2. 主 raw guard20，共同 71 支持', '']
    prefix = 'raw/guard20/common71/'

    def table(title, rows, metrics, data=summary, pre=prefix):
        lines.extend([f'### {title}', '', '| 对比 | ' + ' | '.join(metrics) + ' |', '|:--|' + ':--|' * len(metrics)])
        for label, key in rows:
            lines.append('| ' + label + ' | ' + ' | '.join(value(data[pre + m + '/' + key]) for m in metrics) + ' |')
        lines.append('')

    conditions = ['baseline', 'phoneN', 'phoneT', 'globalN', 'globalT']
    table('各臂处理后值', [(c + '/' + a, f'cell/{c}/{a}') for c in conditions for a in 'NT'],
          ['C', 'B', 'D', 'C_anchor', 'D_anchor', 'search_uplift', 'best_lag'])
    table('相对原 baseline 的响应', [(c + '/' + a, f'response/{c}/{a}') for c in conditions[1:] for a in 'NT'],
          ['C', 'B', 'D', 'C_anchor', 'search_uplift'])
    table('模板来源与交互',
          [(kind + ' T模板−N模板/' + a, f'{kind}/TminusN_template/{a}') for kind in ['phone', 'global'] for a in 'NT'] +
          [(kind + ' source×template', f'{kind}/source_template_interaction') for kind in ['phone', 'global']] +
          [('phone−global/' + src + '模板/' + a, f'phone_minus_global/template{src}/{a}') for src in 'NT' for a in 'NT'] +
          [('phone/global模板差的交互/' + a, f'phone_global_template_interaction/{a}') for a in 'NT'],
          ['C', 'B', 'D', 'C_anchor', 'search_uplift'])
    table('原配 T−N 差及变化', [(c + ' gap', 'gap/' + c) for c in conditions] +
          [(c + ' gap change', 'gap_change/' + c) for c in conditions[1:]], ['C', 'C_anchor', 'search_uplift'])
    lines += ['标准 C 与固定 k3 的 C_anchor 分开解释。C−C_anchor 是 search uplift；'
              '若只有标准 C 改善，只能称最佳 lag 搜索口径的收益，不泛化为原时钟同步改善。', '',
              '## 3. 全 10 视图敏感性', '',
              'raw/unit 各自保留父定义，不跨几何计算份额。下表始终并列两个自然臂主对比；所有其他端点/对比的完整 CI 在 summary.json.gz。', '',
              '| 几何/支持 | C phoneT−phoneN/N | C phoneT−baseline/N | anchor phoneT−phoneN/N | anchor phoneT−baseline/N |',
              '|:--|:--|:--|:--|:--|']
    for geom in ['raw', 'unit']:
        for policy, support in [('guard20', 'common71'), ('valid', 'common71'), ('valid', 'all74'), ('guard0', 'common71'), ('guard0', 'all74')]:
            pre = f'{geom}/{policy}/{support}/'
            keys = [pre + m + '/' + k for m in ['C', 'C_anchor'] for k in ['phone/TminusN_template/N', 'response/phoneT/N']]
            lines.append('| ' + pre.rstrip('/') + ' | ' + ' | '.join(value(summary[k]) for k in keys) + ' |')
    lines += ['', '## 4. 旧事件支持衔接', '',
              '保持旧 32 clips / 13 speakers / 1411 queries、原 donor/k3/image3，不套新 guard20，不与官方 71 支持混称。'
              '实际 V 改变，因此正距离不再是不变量；没有将旧共享场的表示空间不变量套用到本生成器实验。', '']
    em = protocol['event_metrics']
    for geom in ['raw', 'unit']:
        table(geom + ' 事件端点',
              [(c + '/' + a, f'cell/{c}/{a}') for c in conditions for a in 'NT'] +
              [('phoneT−phoneN/' + a, f'phone/TminusN_template/{a}') for a in 'NT'] +
              [('phoneT−baseline/' + a, f'response/phoneT/{a}') for a in 'NT'] +
              [('source×phone template', 'phone/source_template_interaction')], em, events, geom + '/')
    lines += ['## 5. 拟合、时钟与输入控制', '',
              '- 仅旧 20 cal clips；750 paired speech occurrences 中 723 双臂都有实际 mel 中心帧，27 输入不支持逐项保留；82 exact 含 tone labels。',
              '- 15 eval-speaker LOSO；frame→occurrence→clip→speaker 等权。phone 按共同独立 cal speaker 数 k/(k+4) 收缩至 source global；缺失 speech label 使用 global。',
              '- Silence/gap/unknown/uncovered 中心恒等，四处理共享 speech mask，所有帧/clip 保留。没有 N/T 时间 warp。',
              '- 生成帧 i 的 PTS=i/25；实际 mel 块起点 s 对应中心 (s+7.5)/80，尾块用 M−16。窗口跨 phone，模板包含邻接上下文与相位组成。',
              '- Shrink 在 float64，单次 cast32；原 z 与 prototype 各显式 float32×.5 再 float32 相加，非 speech 直接 copy。',
              '- 同 image3、模型、ROI、25fps、FFV1、JPEG、batch32；原 FIXED A 不变且无新音频 forward/波形/MFA。', '',
              '## 6. 工程与独立审计', '',
              '- 1071 冻结输入/依赖先验核验；全部 74 旧官方曲线/端点及 1411 事件查询精确重放，最大差 0。',
              '- 首 cal N/T none/noop/cached 三模式现场：native z、像素、PTS、JPEG、V 与旧缓存 exact；身份门后才 fit。',
              '- 首 cal 四处理各原始/重复视频，现场独立媒体检查和 full-vs-stream；通过后才全部 eval。临时视频只在已封存 V/metadata 后释放。',
              '- 外部现场媒体复核范围是首 cal 22 个新 AVI 与 2 个旧 baseline 视频；592 eval 的创建时 FFmpeg/PTS/JPEG 门由 producer 执行，外部最终复核读取封存数组/metadata，未冒称对已删除视频作现场检查。',
              '- 独立 15LOSO/723occurrence 拟合重建，raw_mu、counts、training IDs 最大误差 0；最终距离/统计的精确误差见下列不可变 receipt。',
              f"- 三角/交互最大闭合误差：{analysis['triangle_interaction_max']:.17g}。", '']
    for path, digest in binding['receipts'].items():
        lines.append(f'- `{Path(path).name}` SHA256 `{digest}`；路径 `{path}`。')
    lines += ['', '## 7. 解释边界', '',
              '沿用历史 speaker、句子和模型，属于机制探索，不是新确认；LOSO 仅排除目标 eval speaker 的 cal clips。'
              '凸组合非负不保证仍位于训练流形；同权混合也会压低原残差和改变 phone 边界动态。'
              '同 λ 不等于 N/T 两臂相同 latent 改变量；原型残差包含说话人、句子偏移与 phone 内动态。'
              '处理后 T−N 差距扩大若来自 N 损伤更大，不能称 T 质量提高，不能由剂量差描述直接推出根因。'
              'phone 模板不是纯音素身份，不将 Sync-C 或事件 margin 当物理嘴型质量，不计算中介比例。'
              '模板来源是自然/TTS 音频经过 Wav2Lip audio_encoder 的 z，并非从 TTS 生成视频提取的可迁移信息。'
              '因此不能据此直接确认或反驳“TTS 视频含可迁移成分”的命题。'
              '固定自然音轨可以定位生成路径的操作响应，但不能将当前模型的机制推广至其他生成器。', '',
              '## 8. 可复现资产与资源', '',
              f"- protocol SHA256 `{sha(RUN / 'protocol.json')}`；fit NPZ SHA256 `{sha(RUN / 'fit.npz')}`。",
              '- `scores/*.json.gz` 保存 592 新 cell 全部曲线/指标，baseline_scores 保存旧桥接；summary/event_summary 保存全比较。',
              '- `event_query_metrics.npz`、`event_clip_metrics.npz` 保持全部事件值；feature seals 与 metadata 绑定 V/原 A/模型/生成像素证据。',
              '- 新阶段总持久上限 320MiB（main304+audit16），临时 /dev/shm96MiB，free−未写承诺≥4.25GiB；旧阶段资源门不回写。GPU/临时与最终字节释放情况见 resource_closure.json。', '']
    (RUN / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
