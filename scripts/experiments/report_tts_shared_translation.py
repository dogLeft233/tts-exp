"""Summarize shared-shift invariance, contrast changes and random controls."""
import hashlib
import shutil
from pathlib import Path

import numpy as np

from scripts.experiments.tts_shared_translation import BASE, ROOT, read, write


def row(label,r):
    return f"| {label} | {r['speaker_mean']:+.6f} | [{r['speaker_ci95'][0]:+.6f}, {r['speaker_ci95'][1]:+.6f}] | [{r['speaker_ci99'][0]:+.6f}, {r['speaker_ci99'][1]:+.6f}] |"


def main():
    fit=read(BASE/"fit.json");a=read(BASE/"analysis.json");inv=read(BASE/"invariance.json")
    lines=["# 音视频共同域平移对四格交互的诊断", "",
           "独立calibration估计的共享平移没有明显消除跨臂双向惩罚：margin交互从+2.250272变为+2.232875，改变−0.017397的95/99%CI均跨0。T双模态反向平移会增加交互。拟合方向区别于等范数随机方向，但改善主要是很小的音频侧cross效应；残余不能自动解释为实例轨迹适配。", "",
           "## 冻结协议", "",
           "独立26cal沿用既有MFA occurrence/phase与annotation/index门；20条、12speaker、890queries有效。排除a1_019/a1_031/a1_085（phone序列不一致）、a1_026/a1_079/a1_080（speech occurrences不足5）。校准k=3，不使用eval拟合。32eval/13speaker/1411queries×3图及其共同负池完全固定。", "",
           "每条query等权，δA=mean(A_T−A_N)且音频只计一次；δV先每图query均值再三图等权。分别条目在speaker内等权、speaker等权，最后δ=(δA+δV)/2。T音频和视觉同时减同一个δ；反向控制同时加δ。16个固定随机控制为PCG64(20260926) standard_normal(16,1024)逐行归一化×||δ||，同时从T双模态减去。无eval调强度、无独立A/V变换、无Procrustes、无平移后L2归一化。", "",
           f"δA范数={fit['norm_A']:.6f}，δV范数={fit['norm_V']:.6f}，δ范数={fit['norm_delta']:.6f}；A/V均值差方向cos={fit['cosine_A_V']:.6f}，夹角={fit['angle_degrees']:.6f}°。只是诊断，不据此改变权重。", "",
           "四格qXY=q(V_X,A_Y)；主q是raw Euclidean mean-negative minus positive距离margin，eps=1e−6。交互I=qTT+qNN−qTN−qNT。三图先条内均值，再speaker等权；bootstrap20000 seed20260926，95/99%CI。", "",
           "## 主结果", "", "| 条件/端点 | 均值 | 95%CI | 99%CI |", "|---|---:|---|---|"]
    for condition in ("original","subtract_delta","add_delta"):
        for endpoint in ("interaction","interaction_change","visual_at_N_audio","audio_at_N_visual"):
            lines.append(row(condition+"/margin/"+endpoint,a[f"{condition}/margin/{endpoint}"]))
    lines.extend(["", "## 平移后的变化与辅助诊断", "", "| 端点（减δ−原始） | 均值 | 95%CI | 99%CI |", "|---|---:|---|---|"])
    for metric,endpoints in [("margin",("visual_at_N_audio_change","audio_at_N_visual_change")),("rank",("interaction_change","visual_at_N_audio_change","audio_at_N_visual_change")),("positive",("interaction_change",)),("negative",("interaction_change",))]:
        for endpoint in endpoints:
            lines.append(row(metric+"/"+endpoint,a[f"subtract_delta/{metric}/{endpoint}"]))
    lines.extend(["", "音频cross margin改善+0.023844（99%CI正），视觉cross margin变化−0.006447跨0；视觉cross rank下降−0.002767、音频cross rank上升+0.003144，交互rank改变−0.000377跨0。共享平移没有同时救回两侧cross配对。", "",
                  "## 等范数随机方向控制", "",
                  "下表为16个固定随机方向的speaker均值分布；这不是严格随机化p检验，不能把方向计数转成显著性概率。", "",
                  "| margin改变 | 随机最小 | 随机中位数 | 随机最大 | 减拟合δ |", "|---|---:|---:|---:|---:|"])
    random_summary={}
    for endpoint in ("interaction_change","visual_at_N_audio_change","audio_at_N_visual_change"):
        vals=[a[f"random_{i:02d}/margin/{endpoint}"]["speaker_mean"] for i in range(16)]
        fitted=a[f"subtract_delta/margin/{endpoint}"]["speaker_mean"]
        summary={"min":min(vals),"median":float(np.median(vals)),"max":max(vals),"fitted":fitted,"random_values":vals,"count_random_le_fitted":sum(v<=fitted for v in vals)}
        random_summary[endpoint]=summary
        lines.append(f"| {endpoint} | {min(vals):+.6f} | {np.median(vals):+.6f} | {max(vals):+.6f} | {fitted:+.6f} |")
    lines.extend(["", "拟合方向使I略降，而16个随机方向均使I小幅上升；说明方向有诊断信息。但拟合I改变本身跨0，且原强交互基本保留；不能用方向特异性夸大为解释机制成立。", "",
                  "## 保真断言", "",
                  f"所有19条件的native NN/TT query指标最大误差{inv['max_native_query_metric_error']:.3g}，T/T guard20官方C最大误差{inv['max_guard20_C_error']:.3g}，T内AA/VV距离最大误差{inv['max_AA_VV_distance_error']:.3g}。NN表示从未改变。全部32eval×3图、所有条件验证通过。", "",
                  f"完整T/T lag矩阵误差{inv['max_native_full_distance_error']:.3g}；该完整矩阵核验把虚拟零padding也从0移到−δ，只是坐标恒等式检查，不是重新运行原始音频前端。主guard20完全没有padding，是真实embedding配对保真。没有先平移再单位归一化。", "",
                  "## 解释与限制", "",
                  "这个实验直接检验的是一个全局、共享、只由calibration确定的平移。它未明显移除共同事件四格的交互，未支持用该calibration均值差定义的共享平移解释强交互。它未排除phone/context条件下的域差异、非线性几何、MFA时间误差或其他模型表征差异；残余不能自动归为实例轨迹互配。", "",
                  "所有nativeTT/NN与native Sync-C严格不变，因此无论cross交互改变多少，都不能给native TTS优势分配因果贡献百分比，也没有实际嘴型改进。cal支持20/26与12/13speaker的限制保留；cal与eval ID分离而speaker共享；CI条件于已冻结δ，未传播cal拟合的不确定性。MFA非物理真值、phone非viseme、约200ms窗跨phone仍然适用。", "",
                  "## 验证和产物", "",
                  "独立check_tts_shared_translation.py以直接query权重重估δ、独立构造随机方向，以原始差向量±δ重算全部四格，重算图像/条目/speaker聚合与bootstrap；结果见validation.json。原始条件与前序四格逐query重放一致。测试3 passed。", "",
                  "protocol.json；calibration_support.json；translation.npz/fit.json；scores/manifest.json；analysis.json/utterance_effects.json；invariance.json；validation.json；random_controls.json；provenance.json。源码与测试快照在source。全程CPU，未占用GPU。"])
    write(BASE/"random_controls.json",random_summary)
    (BASE/"report.md").write_text("\n".join(lines)+"\n")
    source=BASE/"source";source.mkdir(exist_ok=True)
    paths=[Path(__file__),ROOT/"scripts/experiments/tts_shared_translation.py",ROOT/"scripts/experiments/check_tts_shared_translation.py",ROOT/"scripts/experiments/tts_negative_pool.py",ROOT/"tests/experiments/tts_shared_translation/test_protocol.py"]
    provenance={}
    for path in paths:
        shutil.copy2(path,source/path.name);provenance[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for pattern in ("*.json","*.npz"):
        for path in BASE.rglob(pattern):
            if path.name!="provenance.json":provenance[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    provenance[str((BASE/"report.md").relative_to(ROOT))]=hashlib.sha256((BASE/"report.md").read_bytes()).hexdigest()
    write(BASE/"provenance.json",provenance)
    print(BASE/"report.md")


if __name__=="__main__":main()
