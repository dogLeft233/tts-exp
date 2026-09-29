"""Report the independent static lag calibration and repaired evaluation."""
import collections
import hashlib
import json
import shutil
from pathlib import Path

from scripts.experiments.tts_static_lag_calibration import BASE, ROOT, read, write


def row(label,r):
    return f"| {label} | {r['speaker_mean']:+.6f} | [{r['speaker_ci95'][0]:+.6f}, {r['speaker_ci95'][1]:+.6f}] | [{r['speaker_ci99'][0]:+.6f}, {r['speaker_ci99'][1]:+.6f}] |"


def main():
    selection=read(BASE/"selection.json")
    lines=["# 同静态生成协议的独立 lag 校准与共同负池修复", "",
           f"独立26条calibration音频在原38条eval相同静态生成协议下完成156/156个N/T评分，锁定共同k={selection['k']}（audio_index−visual_index）。所有arm/image组中位数相同，校准迁移门通过。以下为已见过eval之后修复协议的探索分析，不是全新确认性结果。", "",
           "## 冻结与执行", "",
           "沿用旧固定26cal IDs、13speaker；与38eval ID完全不重叠。3静态图、生成框、裁剪框、native Wav2Lip mel前端、权重、FFV1输出、SyncNet MJPEG提取与eval相同；仅新生成N/T，不生成PRE/POST。source PCM与audio embedding沿用经SHA核验的缓存。156个guard20 best-k整数中位数向0截断；六组中位数跨度须≤1帧。没有按eval选k。", "",
           "GPU执行前V100空闲，无foreign PID，使用共享gpu_lease；每cell检查占用。无训练。旧k=1第一阶段完整保留在 runs/tts_negative_pool_20260926；它的弱结果首先是动态cal→静态eval测量校准不匹配，不说明原生优势来自lag优化。", "",
           "| 图/臂 | cal中位数k | best-k分布 |", "|---|---:|---|"]
    for name,g in selection["groups"].items():
        lines.append(f"| {name} | {g['median']} | {dict(collections.Counter(g['k']))} |")
    lines.extend(["", "## 修复后的原冻结负池检验", "",
                  "所有MFA/phase/格点/最小支持/负池规则沿用第一阶段，按新k重建query支持。共同event/phase IDs、raw距离、单位范数和conditional rank定义不变。三图条内均值、speaker等权、bootstrap20000 seed20260926；支持和95/99%CI均公开。k±1是预先指定敏感性，不择优。", "",
                  "| 条件 | eligible条目/说话人 | queries | 排除IDs |", "|---|---:|---:|---|"])
    for folder in ("evaluation","evaluation_lag_-1","evaluation_lag_+1"):
        data=read(BASE/folder/"support.json")["rows"]
        good=[r for r in data if r["eligible"]]
        k=read(BASE/folder/"protocol.json")["fixed_audio_minus_visual_index"]
        lines.append(f"| {folder}, k={k} | {len(good)}/{len(set(r['speaker'] for r in good))} | {sum(len(r['queries']) for r in good)} | {','.join(r['id'] for r in data if not r['eligible'])} |")
    lines.extend(["", "| 端点 | T−N | 95%CI | 99%CI |", "|---|---:|---|---|"])
    for folder in ("evaluation","evaluation_lag_-1","evaluation_lag_+1"):
        k=read(BASE/folder/"protocol.json")["fixed_audio_minus_visual_index"]
        a=read(BASE/folder/"analysis.json")
        for name in ("official_queries_C","official_queries_fixed_median_margin","official_queries_fixed_mean_official_margin","raw_native_margin","raw_matched_margin","raw_did_margin","raw_matched_rank","unit_matched_margin","unit_matched_rank","raw_matched_AA","raw_matched_VV"):
            lines.append(row(f"k={k} {name}",a[name]))
    lines.extend(["", "主k=3共同池raw margin +0.436、unit margin +0.0434均99%CI正，负池差中差+0.019跨0；统一event/phase负池后没有观察到优势削弱，弱化纯负池组成解释。raw rank +0.00877跨0，unit rank +0.01293的99%CI正，不把所有指标都称阳性。k=2保留优势，k=4不确认，说明只有一帧偏移也影响局部可辨识性结论。", "", "## 各臂正负距离及采样组成", "",
                  "主k的positive相同，只有负池改变；共同池时间分布仍可能不同，故是event/phase与时间窗联合参照干预。以下保留每臂均值，避免把负距离变化直接叫同步改善。", "",
                  "| arm/pool | positive | negative | margin | rank |", "|---|---:|---:|---:|---:|"])
    a=read(BASE/"evaluation/analysis.json")
    for arm in "NT":
        for pool in ("native","matched"):
            vals=[a[f"level_{arm}_raw_{pool}_{m}"]["speaker_mean"] for m in ("positive","negative","margin","rank")]
            lines.append(f"| {arm}/{pool} | "+" | ".join(f"{v:.6f}" for v in vals)+" |")
    lines.extend(["", "| arm/pool | same occurrence | same phone other | different phone | gap |", "|---|---:|---:|---:|---:|"])
    for arm in "NT":
        for pool in ("native","matched"):
            vals=[a[f"level_{arm}_composition_{pool}_{c}"]["speaker_mean"] for c in ("same_occurrence","same_phone_other","different_phone","silence_gap")]
            lines.append(f"| {arm}/{pool} | "+" | ".join(f"{v:.6f}" for v in vals)+" |")
    lines.extend(["", "## 验证与限制", "",
                  "check_tts_static_lag_calibration.py 独立复算156个cal距离矩阵、best-k选择、校准组中位数，解码检查全部视频生成框外像素恒等，验证所有源SHA；再逐query/统计复算三个eval lag。结果见validation.json及三个eval目录validation.json。pytest静态协议两项通过，负池核心四项通过。源码及完整产物SHA保存在provenance.json/source。", "",
                  "MFA非跨臂物理事件真值、phone非viseme、200ms窗口跨多phone。cal/eval共享speaker，3图不可视作独立样本。共同事件负池的优势仍属于SyncNet表示中的条件可辨识性，不代表人类唇形同步真值。单位归一化改变几何；没有任何原生Sync-C贡献百分比或物理替换成功声明。共同k从独立ID确定，但协议修复发生在已见eval之后，结论仍为探索性。"])
    write(BASE/"report_inputs.json",{"selection_sha256":hashlib.sha256((BASE/"selection.json").read_bytes()).hexdigest()})
    (BASE/"report.md").write_text("\n".join(lines)+"\n")
    source=BASE/"source";source.mkdir(exist_ok=True)
    paths=[Path(__file__),ROOT/"scripts/experiments/tts_static_lag_calibration.py",ROOT/"scripts/experiments/check_tts_static_lag_calibration.py",ROOT/"scripts/experiments/tts_negative_pool.py",ROOT/"scripts/experiments/check_tts_negative_pool.py",ROOT/"scripts/experiments/tts_prepost_mel.py",ROOT/"scripts/experiments/tts_raw_video_transfer.py",ROOT/"scripts/experiments/tts_native_gain_attribution/syncnet.py",ROOT/"scripts/experiments/masked_tts_tfg_probe/direct_mel.py",ROOT/"scripts/experiments/static_image_bridge/score_worker.py",ROOT/"tests/experiments/tts_negative_pool/test_static_calibration.py"]
    provenance={}
    for path in paths:
        shutil.copy2(path,source/path.name)
        provenance[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in BASE.rglob("*.json"):
        if path.name!="provenance.json":provenance[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    provenance[str((BASE/"report.md").relative_to(ROOT))]=hashlib.sha256((BASE/"report.md").read_bytes()).hexdigest()
    write(BASE/"provenance.json",provenance)
    print(BASE/"report.md")


if __name__=="__main__":main()
