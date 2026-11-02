---
title: Codex checkpoint - 2026-10-20T11-09-02Z - May purified TTS alignment and loader
  diagnosis delivered
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-20-t11-09-02-z-may-purified-tts-alignment-and-loader-diagnosis-delivered
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-20T11:09:02.357822+00:00'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-20T11-09-02Z - May purified TTS alignment and loader diagnosis delivered

## Summary
May云端TTS净化训练与匹配评测已交付；新审计发现净化时间映射会将倒退压成长时间定格，而全量AVE和实际训练加载匹配，尚不能量化定格解释多少Sync-C下降。

## Story
原问题是TTS数据能否提升唇形同步，后来按用户要求回到May首次成功baseline50，只换云端净化TTS训练数据、不做音频归一化、不用本地TTS。已完成同seed42完整initial/原AVE/原优化器与LR/4096rays/maxsteps16的60000次FP32更新，30个完整断点全部F持久核验。随后用户要求评测，得到同一自然heldout、同ROI/SyncNet/原PCM时钟的匹配结果：驱动−6/0/+6时新Sync-C5.296/5.703/5.911，对应自然训练baseline8.669/8.909/9.036。不能将SyncNet下降直接当感知下降或TTS总体因果结论。

最新用户要求先检查净化音画对齐、AVE特征、训练加载是否有问题。因此本轮不改数据、不重训，以实际保存映射、原音频、全量重算AVE和原provider真实collate复核。结果将问题集中到监督数据构造：局部SyncNet校正曲线先取整，然后对倒退源帧号做累计最大值，造成长平台；旧门仅要求12整段Sync-C均>5，没有限制冻结、跳帧或局部残差。该门不足必须明确承认，不把“文件读得对”冒称“标签时序合格”。

## Working State
权威诊断根F:/Datasets/tts-exp/may_tts_purified_diagnostic_20261020_v1。report.md、cpu_data_audit.json、mapping_mechanism_audit.json、loader_ave_receipt.json、holds.csv、local_residual_sync.csv、recomputed_ave.npy是证据入口。问题片段hold_case_preview.mp4及hold_case_diagnostic.png由最终训练JSON和原DOUBLE WAV构建，图已实际view_image；没有听音/盲评结论。新BM报告66-may-tts-alignment-loader-audit已发布并读回，parent实验与研究地图当前段及报告索引已更新，memory_publication_receipt.json passed。报告65是其他chat占用，不覆盖。

原数据根F:/Datasets/tts-exp/may_tts_purified_data_20261020_v1，416成员manifest SHA95cabf40c2bd50e62cf6763d9c3f7a9ba6b8a656d16db1fc39a50b5a772c0491。原训练根F:/Datasets/tts-exp/may_tts_purified_training_20261020_v1；最终step060000.pt SHA32e4be04483a290da463a20d98a1c28c42c104959ab0e2aeebcb0f1004ce3ba2。正式60000success/0skip，原参数不改，30fullcp与全trace已独立审计。原评测根F:/Datasets/tts-exp/may_tts_purified_eval_20261020_v1，150成员manifest SHA092dfba3ba27e3cfcc3d39629f9cde2131b0407327c4df5034a8d802b90b0326。报告64-may-purified-tts-training为完成评测；训练Task done，诊断发现不抹除实际完成记录。旧数据/训练/评测封存根不追加或覆盖。

易失remote /dev/shm/may-tts-purified-diagnostic-20261020-v1；ssh必须-b192.168.50.1 wjj@192.168.50.2。本次诊断job127728自然退出，0生成器构造/0optimizer，仅1冻结AVE eval。没有当前训练/评分/备份host或本chat自动化；ComfyUI是其他工作，不终止。Patrick/coarse7/fine14/Chris/Cathy/原24full仍暂停，不恢复或归档。

## Changed Files
新增未提交scripts/experiments/may_tts_purified_training_20261020/diagnose_loader_ave.py和tests/experiments/test_may_tts_data_diagnostic_v1.py；.codex/scratch/audit_may_tts_data_cpu_v1.py、make_may_tts_diagnostic_visuals_v1.py、publish_may_tts_diagnostic_v1.py及本checkpoint脚本为局部运行入口，精确执行源另存F。未改训练数学、prepared WAV/JSON、冻结模型源或历史科学资产。仓库大量其他任务dirty，不stage/clean/commit/push。CodeGraph先查但新鲜度拒绝后才直接读源码。

## Verification
12段实际DOUBLE WAV拼接逐样本精确，最终3914帧/156.56秒；3849监督aud_id40..3888连续无内部缺口，最终映射和全部原source几何逐项精确。原416成员fresh SHA通过。旧分段receipt在修复chunk001长度后有过时start字段，但最终组装已按实际段长重建连续时钟，loader不读这些旧字段；不是72帧全局错位。

3849监督行仅2905唯一源图，覆盖原4150的70.0%；相邻重复24.532%，源帧号最大跳34帧。全部12段从保存correction重现实际cummax映射，624/3914帧(15.943%)受单调钳制改变。最长45帧/1.8秒相同源图2944，实际音频105.64–107.44秒RMS0.068597；41个完整SyncNet窗口视觉嵌入唯一且方差0，音频嵌入41个各不相同，AVE在变化。局部0.48秒窗口1621个中14.682%的最佳lag绝对值>3帧，但同SyncNet短窗噪声大，不是独立物理对齐真值。

完整3915×512 AVE重算与训练cache逐位一致，原权重SHA1f75cdb5556938590c64221f6c47a8d20ab3e368d53c166023eedb12279db8c4。DOUBLE→float32逐样本精确，无额外norm/clip；独立80/25 mel中心和两次边缘padding精确。所有3849中心/8帧上下文/pose/eye匹配；108个实际provider collate样本的原source GT像素、ray索引、音频上下文、相机、landmark/rect/torso背景通过。TTS/自然matched AVE RMS0.864613/0.827853，zero占比0.549488/0.548381且有限，仅排除损坏/全零，非语义相等证明。

3项实际数据CPU测试直接调用通过，test_receipt.json保存；pytest未安装未运行pytest/整库。初次preview用system python3缺numpy，原失败保留；改用现有lexical env无需安装/改共享venv。matplotlib也只读复用既有plotdeps；图与源图实际查看通过。F重算AVE读回SHA及NumPy精确通过，无训练重放。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)
- 本地HEAD `2b71639c3b555293375d248350999377168d515d`；GitHub能否解析见checkpoint_git_evidence.json，不构造未验证commit链接。
- F:/Datasets/tts-exp/may_tts_purified_diagnostic_20261020_v1/report.md
- F:/Datasets/tts-exp/may_tts_purified_diagnostic_20261020_v1/memory_publication_receipt.json

## Observations
- [result] 发现真实净化时间映射退化：最长1.8秒相同嘴型标签对应变化音频；全量AVE/cache与实际loader未见身份、时钟或特征错配。
- [decision] 整段Sync-C>5不能独自作为训练标签验收；本轮仅诊断，未修复数据、恢复旧训练或启动新训练。
- [decision] 尚不能证明该问题解释全部3.206分下降，也不能推出TTS本身无效；需修正数据后的匹配对照才能估计贡献。
- [next_step] 在用户授权修订数据后，先设计限制冻结时长/映射速度和局部对齐的验收，再用修正数据做同baseline对照；不要直接自动重训。

只读GitHub证据：{"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 422}}。

## Relations
- continues [[Codex checkpoint - 2026-10-20T06-30-27Z - May purified TTS AVE qualified and training started]]
- relates_to [[May 云端 TTS 净化训练数据对照]]
- relates_to [[64-may-purified-tts-training]]
- relates_to [[66-may-tts-alignment-loader-audit]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1092条；大量其他任务改动，不批量stage/clean/commit。
