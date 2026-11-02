---
title: Codex checkpoint - 2026-10-24T08-03-15Z - Preference probe negative and bounded
  SyncNet mechanism route
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-24-t08-03-15-z-preference-probe-negative-and-bounded-sync-net-mechanism-route
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-24T08:03:15Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16144-7517-77a1-b372-027fba42da62
model: gpt-6-astra
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-24T08-03-15Z - Preference probe negative and bounded SyncNet mechanism route

## Summary
自然N、Qwen0.6B和1.7B的140条归一化音频偏好探针已完成并独立复算，没有取得HuBERT增量预测价值；下一步建议限定为当前三臂的局部视听匹配与错位参照组成诊断，避免重复历史时间置换和幅度分解。

## Story
研究目标是解释TTS驱动时SyncNet高分及1.7B相对0.6B的来源依赖反转，用户暂以评价器响应为对象，人类感知验证后置。前序完成四句短语定位与连续帧/波形核对，未发现重复一致的明显口部错误。随后定向文献检索支持先做轻量偏好探针；用户明确要求加入自然音频并指定gpt-6.1-sol medium子代理执行。/root/syncnet_preference_probe完成共同冻结HuBERT特征提取和小型Ridge分析头，父侧独立核验。未训练TTS/TFG，未新合成/视频/SyncNet前向、人评或SAE。

最新用户问应否直接看SyncNet内部音视频特征、探针失败是样本少还是时间汇总丢失，并要求避免旧实验重复。本轮只读核查BM、源码及官方SyncNet源；未启动新实验。判断不能用本轮单一负结果辨别样本量/表示/模型容量原因。有效独立组20，非140/300；均值和标准差消去编码后帧序列的显式顺序，HuBERT上下文化仍可能隐含部分时间信息；audio-only遗漏视频，原生分数又联合生成与评价，因此结构性缺项比盲扩样更值得先查。

重要去重发现：2026-09-26已做SyncNet音频模长/方向、频带修复、中点M/R及成对时间置换；09-27已做归一化残余、独立置换、anchor/search uplift；09-17局部时间rank正而跨音素event不确定；09-20共同播放时长后rank优势未确认。故不能把“首次看内部”“打乱时间后优势下降”“再拆B/D”包装为新机制。旧03-causal-mechanism中的共适应和英文不成立强说法已被后续地图边界修正，勿直接沿用。

下一路线仅为建议：在当前N/L06/L17三臂内，先用现成31lag×时间距离、时钟和对齐检查同样lag对应的内容是否难度不同（同一发音/跨发音/语音—停顿/真实边界），再在共同可测支持上比较同类错位关系；需覆盖完整约200ms感受野而非单中心标签，FA不作为真实嘴型标签。保留完整官方分数与单独的标准化诊断，不把加权后数字称官方C。若条件差异仍在，再补缺失的原始SyncNet逐窗A/V特征，检查各自时间变化及跨模态关系，针对一个重复候选作有限依赖性干预。不要先全层扫描、训练更大RM、跨N/T直接换轨或把由距离重建C当预测成功。这个内容条件化问题与旧时间rank/置换邻近，价值是定位三臂正负异质性，而非保证全新理论或已有因果答案。

## Working State
权威当前run：runs/syncnet_preference_probe_20261024_attempt01/。本机E:/Documents/tts-exp，远端/home/wjj/tts-audio/tts-exp，两者同相对路径。115文件9,598,753字节本机运输核验通过，未声称Baidu归档。远端SSH为wjj@192.168.50.2；旧本机E:/Documents/tts-audio/tts-exp退役，不混淆远端合法路径。

探针协议：N20、L06/L17各60，20来源；300pairs含180三臂主比较；共同资格剔两个来源的双方d3，286pairs/174主pairs。固定HuBERT最后层mean/std1536、训练折内PCA8、nested group Ridge；M1=arm+6声学+支持，M3=M1+HuBERT。N复用不增样，来源同折。主要MAE M1 .580837、M3 .597757；改善-.016920，描述性95%[-.047705,+.014794]，两批MAE均更差；资格MAE .585990/.616607。资格排序M1 61.111%、M3 63.889%，事后训练折family多数规则60.833%；不把M1全量65.556%称稳健机制。全部四项推进门失败，复杂RM/SAE停止；不否定所有局部视听机制。

更早资产：runs/qwen_phrase_visual_review_20261024_attempt02/，45文件和102链接通过。S0002去自动末字购区间后的局部差+1.654/+1.326/-.822，非物理剪辑/重评分。S0014两个下降短语没有跨draw一致的明显动作缺失。音频接口不支持直接听辨，只有波形/ASR/FA与模型看帧，不得说亲耳听出。

当前Git分支codex/cross-tfg-audio-head-20261003；HEAD 2b71639c3b555293375d248350999377168d515d。只读GitHub确认仓库，当前head PR查询为空；GitHub无法解析此SHA（422），仅作为本地SHA。工作树有大量继承修改/未跟踪文件，未清理或commit。当前新脚本仍未跟踪，run与SSH可用性是机器局部状态。

## Changed Files
- scripts/experiments/syncnet_preference_probe_20261024.py：新探针执行器。
- tests/test_syncnet_preference_probe_20261024.py：3项科学契约测试。
- scripts/experiments/syncnet_preference_parent_audit_20261024.py：不导入runner的独立闭式重放。
- run内report_source_attempt02.py、majority_source.py及source快照保存报告修复/多数规则。
- BM实验结论与研究主线与证据地图已在前一执行阶段更新；本轮仅新增此不可变checkpoint。

## Verification
前一执行阶段父侧实际重新检查140组输入/视频/评分/原float32距离SHA，160外层模型闭式重新拟合；系数最大误差2.524e-7，OOF1.888e-7，主指标和来源bootstrap误差0。本机115产物SHA无缺失错配。未独立重做HuBERT前向或内层alpha搜索；该部分仅源码/记录检查。3项测试是直接断言调用PASS，环境无pytest，不称pytest命令运行。
科学数组已保存后analyze因缺matplotlib报错；原失败现场保留，report-only attempt02用PIL完成，未重提取或拟合，科学SHA不变。
本轮重读parent_validation.md及canonical BM，CodeGraph核查SyncNet评价调用，官方源核对逐窗A/V距离、先均值再median/min，未重跑科学计算。gh CLI不可用，改GitHub连接器完成只读仓库/PR核查，commit未解析不构造链接。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Official SyncNet scoring source](https://github.com/joonson/syncnet_python/blob/master/SyncNetInstance.py)
- runs/syncnet_preference_probe_20261024_attempt01/report.md
- runs/syncnet_preference_probe_20261024_attempt01/parent_validation.md
- runs/syncnet_preference_probe_20261024_attempt01/parent_independent_audit.json
- runs/syncnet_preference_probe_20261024_attempt01/protocol.json
- runs/syncnet_preference_probe_20261024_attempt01/final_manifest.json

## Observations
- [result] 固定HuBERT全句汇总+线性头没有增量预测价值；原因未通过对照区分，不能直接归罪样本少或证实时间信息是缺因。
- [decision] 暂停复杂RM/SAE扩展，保持自然N及全部三比较、来源级统计、全量/共同资格两版本。
- [decision] 已做模长方向、时间rank、成对/独立置换、B/D分解不重复作为新实验；使用其结果与代码做基线校验。
- [blocker] 当前140条是否保留可直接消费A/V embedding缓存尚未资产核验；全20来源细粒度对齐/完整RF共同支持尚未核验；独立感知结论缺人工证据。
- [next_step] 若用户同意继续，将“同类错位参照下三臂局部匹配差异”冻结成有限诊断协议并先做缓存与对齐资格检查；当前仅完成路线判断，尚未启动。

## Relations
- continues [[Codex checkpoint - 2026-10-24T02-25-23Z - Four sentence phrase localization completed]]
- summarizes [[自然音频与双规模TTS的SyncNet偏好探针 2026-10-24]]
- relates_to [[偏好模型解释TTS同步分差的定向文献检索 2026-10-24]]
- relates_to [[三处稳定短语的逐帧与波形核对 2026-10-24]]
- relates_to [[TTS 时间辨识与同文本实例交叉实验 2026-09-17 结果]]
- relates_to [[电平匹配后的中点幅度与时间组织残余诊断 2026-09-27]]
- relates_to [[电平残余的分块独立置换与最优时滞搜索诊断 2026-09-27]]
