"""Write the common-event four-cell report and provenance snapshot."""
import hashlib
import shutil
from pathlib import Path

from scripts.experiments.tts_event_cross import BASE, CAL, CONTRASTS, ROOT, read, write
from scripts.experiments.report_tts_static_lag_calibration import row


def main():
    p=read(BASE/"protocol.json")
    a=read(BASE/"evaluation/analysis.json")
    lines=["# 共同音素坐标的音频视觉四格表示诊断", "",
           "在独立静态校准锁定k=3的共同event/phase坐标中，单独把自然视频或自然音频替换为TTS对应表示都会降低margin和rank；同臂配对的正交互在raw、单位范数和rank上保留。结果支持对源条件匹配的依赖，不能解释为独立、可迁移的TTS视觉或音频单侧优势。", "",
           "## 冻结设计与支持", "",
           "主k=3由26条独立calibration ID、3静态图×N/T共156cells选择，组中位数全为3；k=2/4敏感性全部保留。沿用静态校准修复实验的annotation/index支持，不新增筛选：k3为32条/13speaker/1411query；k2为32/13/1410；k4为33/13/1431。每query负donor是完全相同occurrence/phase IDs，视频和音频分别在各臂自身原生格点采样。", "",
           "四格qXY=q(V_X,A_Y)，X为视频臂、Y为音频臂。q是mean negative−positive距离margin，另有conditional rank。主读数为qTN−qNN（固定A_N视觉替换）、qNT−qNN（固定V_N音频替换）、I=qTT+qNN−qTN−qNT；另报T条件简单效应。三图条内均值→speaker等权、bootstrap20000 seed20260926、95/99%CI；主三项按99%CI保守阅读。", "",
           "## 主k=3四格level", "", "| geometry/cell | positive D | negative Bmean | margin | rank |", "|---|---:|---:|---:|---:|"]
    for geom in ("raw","unit"):
        for cell in ("NN","NT","TN","TT"):
            vals=[a[f"{geom}_{m}_level_{cell}"]["speaker_mean"] for m in ("positive","negative","margin","rank")]
            lines.append(f"| {geom}/{cell} | "+" | ".join(f"{z:.6f}" for z in vals)+" |")
    lines.extend(["", "## 主k=3配对对比", "", "| 端点 | 差值 | 95%CI | 99%CI |", "|---|---:|---|---|"])
    for geom in ("raw","unit"):
        for metric in ("margin","rank"):
            for name in CONTRASTS:
                lines.append(row(f"{geom}/{metric}/{name}",a[f"{geom}_{metric}_{name}"]))
    lines.extend(["", "主raw margin视觉替换−0.905、音频替换−0.909、I=+2.250；对应raw rank−0.0515/−0.0544/+0.1147，99%CI均不跨0。unit margin及rank同方向。自然与TTS都更偏好同臂配对，这不是TTS专属同实例适配的证明。", "",
                  "对角T−N raw margin+0.436保留，而raw rank+0.00877未确认；unit对角margin+0.0434和rank+0.01293均99%CI正。对角优势和同臂交互是不同问题，不能将交互占比解释为总增益贡献。", "",
                  "## 正距离与负距离的交互诊断", "", "| k=3端点 | 差值 | 95%CI | 99%CI |", "|---|---:|---|---|"])
    for geom in ("raw","unit"):
        for metric in ("positive","negative"):
            lines.append(row(f"{geom}/{metric}/interaction",a[f"{geom}_{metric}_interaction"]))
    lines.extend(["", "raw正距离交互−2.287，负距离交互−0.0368跨0；unit也同样主要呈现在正对距离。此为该共同坐标统计量的算术定位，不能单凭这一恒等式作生成端因果解释。MFA相位对齐残差、跨phone上下文、同臂特有发音/谱形匹配都仍可能造成该交互。", "",
                  "## 全部预定lag敏感性", "", "| 条件/端点 | 差值 | 95%CI | 99%CI |", "|---|---:|---|---|"])
    for folder in ("evaluation_lag_-1","evaluation_lag_+1"):
        k=read(BASE/folder/"receipt.json")["k"]
        r=read(BASE/folder/"analysis.json")
        for geom in ("raw","unit"):
            for metric in ("margin","rank"):
                for name in ("visual_at_N_audio","audio_at_N_visual","interaction","diagonal"):
                    lines.append(row(f"k={k}/{geom}/{metric}/{name}",r[f"{geom}_{metric}_{name}"]))
    lines.extend(["", "raw margin交互k2=+1.902、k4=+0.629，99%CI均正；但k4对角优势不确认，音频在自然视觉下效应不确认。不能声称所有简单效应跨lag稳健。", "",
                  "## 验证、溯源与结论边界", "",
                  "每个同臂四格值与上一阶段共同池值逐query完全一致。独立check_tts_event_cross.py从源embedding重算四格距离/rank、重新聚合speaker及bootstrap区间；见三个目录validation.json。全源SHA链到静态校准选择、原始MFA支持和原生缓存，见protocol.json、receipt.json和provenance.json。源码快照在source，pytest四格2项、校准2项、负池4项合计8 passed。", "",
                  "该分析是共同注释坐标中的表示配对诊断；没有生成更优自然音频驱动视频，没有物理替换原始自然音频，没有人类同步真值。它削弱“TTS任一单侧表示本身就普遍更好、可迁移到自然另一侧”的解释，支持现有优势依赖音视频源条件匹配。不能区分真正的发音轨迹/上下文适配与MFA事件真值不准确造成的错配，也不能据此确证TTS同实例特异性（这里只有N/T两源，没有两个独立TTS实例）。", "",
                  "当前32条支持、13speaker与3固定历史图的范围有限。cal/eval仅ID分离，修复发生在已见eval之后，属于探索证据。单位范数改变几何，四格并非原生Sync-C的因果分解，所有百分比归因均不成立。"])
    (BASE/"report.md").write_text("\n".join(lines)+"\n")
    source=BASE/"source";source.mkdir(exist_ok=True)
    paths=[Path(__file__),ROOT/"scripts/experiments/tts_event_cross.py",ROOT/"scripts/experiments/check_tts_event_cross.py",ROOT/"tests/experiments/tts_negative_pool/test_event_cross.py"]
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
