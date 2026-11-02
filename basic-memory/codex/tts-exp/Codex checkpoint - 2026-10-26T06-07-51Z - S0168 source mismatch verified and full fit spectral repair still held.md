---
title: Codex checkpoint - 2026-10-26T06-07-51Z - S0168 source mismatch verified and
  full fit spectral repair still held
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-26-t06-07-51-z-s0168-source-mismatch-verified-and-full-fit-spectral-repair-still-held
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-26T06:07:51Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a175f1-7164-7510-ae4b-f5af8e9d0fde
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-26T06-07-51Z - S0168 source mismatch verified and full fit spectral repair still held

## Summary
独立识别器与原镜像来源核验支持S0168参考覆盖不全；L6频谱修复完整300条复验仍0.749654未过0.750000，保持技术HOLD。

## Story
研究原目标是逐音素静态／动态／联合TTS差方向迁移，完整11臂、联合正向为唯一主检验。用户后续明确只做拟合集校准，不降门、不剔除样本，开发与确认先不动。前轮普通声码器替代prematched的候选及峰值0.95已将300条失败3→1，剩S0168W0257=0.7470257940624025。本轮用户要求先核完整内容，再区分编码与声码器损失，尝试修复并全拟合集复验。

最新权威入口 runs/phoneme_direction_20261026_repair03/report.md；full_postfilter_quality.json为完整100人300自然源的独立保存WAV读回。新有界L6频谱候选中位STOI0.900296250348263，S0168=0.749654298575322仍失败，其余299条通过，无新增失败。S0356=0.7595871803632507、S0039=0.7567075595636091。所有原门保持，不能把S0168按三位小数四舍五入成过门；长度、finite、RMS误差、削波、全局包络lag过门不抵消STOI失败。没有准入或冻结。

内容核验采用新下载并绑定文件SHA的Whisper-small，独立于原Paraformer。完整canonical、原始库存及6–9.024秒切段均识别后续网站报道内容，而冻结文字只有“十一月二十日消息”。3–6秒Whisper未输出文字，不能称无声音；网站名Enter／恩特等尚非人工真值。冻结文字、媒体与已有TTS未改。重新下载固定镜像版本31,341,139字节S0168包，SHA与原pin一致；W0257成员与Ubuntu库存和F盘库存逐字节相同。原转写文件SHA也与采集合同一致。问题存在于当前固定镜像来源配对，未发现本地拷贝接错证据，未证明官方各版本皆如此。原库存→canonical STOI0.9998642869278997，最佳标量增益约2.923355、相对残差0.005832，不支持格式转换为主要失真。旧文本范围外全非语音解释继续撤回；原1200条MFA零OOV结构通过不等于完整文字或边界真值。

分离诊断固定20条技术控制加3条旧失败，共23条；卷积前端、归一化后、L6、声码器线性首层分别用固定alpha100 Ridge预测15个原音频带log能量，五折同说话人留出。S0168弱帧RMSE依次3.087/2.922/3.796/3.748dB，实际声码器输出4.849dB。支持L6声学线性可恢复性下降，且声码器未充分还原仍可预测的细节；不是不可逆信息损失或两端责任比例。弱帧定义是400样本窗/320hop、频带总能量低最大帧20dB，仅诊断，非语音/静音分类。

据此只测试一个固定有界后校正：L6预测频带减解码频带，裁±6dB后乘0.5、5帧平滑，权重由解码声音在相对−20至−10dB淡出；1024窗320hop STFT保留解码相位。不读取原音频谱或混合原波形；原单一峰值和最终整体RMS合同保持。23条S0168为0.748903；再固定参数对完整100人300条五折交叉拟合，每折80人240条拟合、20人60条测试，同人三句共折。小型线性头确实拟合；WavLM/HiFiGAN/TTS/TFG未训练。拟合集交叉拟合仍属技术开发，不是独立开发验证。

## Working State
Ubuntu repo /home/wjj/tts-audio/tts-exp，隔离run同名repair03。完整300候选WAV、300份L6特征、23份分阶段特征、原镜像压缩包仍在远端；本地40份JSON、报告、五折头及源码快照逐SHA通过。所有本聊天作业已结束，无新自动化；其它聊天的数据生产控制器未改。

当前HOLD_RECONSTRUCTION_AND_INCOMPLETE_REFERENCE；未估计M/D/J模板、未运行新TFG或SyncNet、未跑新候选全量ASR/事件时序门、未冻结或自动恢复40人。旧repair01的37份、repair02的24份交付重新SHA核对均未改。共享wavlm_knn_vc_adapter.py保持正式prematched路线，普通声码器、峰值前处理及新学习后校正全部仍候选。

仓库大量既有dirty/untracked来自其它聊天；本轮新增代码同样未跟踪、未提交。当前分支无PR，经GitHub只读查询验证；当前HEAD仅本地验证，未生成可能不可解析的commit链接。

## Changed Files
- scripts/experiments/phoneme_direction_20261026/separate_encoder_decoder.py：分阶段声学探针和说话人留出。
- probe_l6_postfilter.py、calibrate_l6_postfilter.py、audit_l6_postfilter.py：有界候选、全300特征与五折拟合、原门独立读回。
- check_s0168_whisper.py、check_s0168_whisper_revision02.py：独立内容核验，保留初次PyAV接口失败。
- tests/test_phoneme_spectral_probe_20261026.py：实际波形零校正恒等/尾长/静音有限性/独立样本Ridge恢复。
- runs/phoneme_direction_20261026_repair03/report.md、delivery_manifest.json与权威JSON。
- basic-memory/Research/逐音素差方向算法与验证方案.md、Research/研究主线与证据地图.md、Experiments/逐音素静态动态差的拟合集技术校准与40人验证.md：最新HOLD与来源/分离证据，已读回确认。

## Verification
原保存WAV的red反馈实际复现0.7470257940624025并触发RED_S0168_FULL_WAVEFORM_RECONSTRUCTION。完整新候选全300独立SHA/WAV/头训练测试speaker不相交核验后仍单条失败；不称bug已解决。
远端固定torch环境20项测试通过，本机14项通过、两个依赖模块跳过，其6项已远端通过；compileall及diff空白检查通过。本地40交付SHA通过，旧61交付不变。模型SHA保持WavLM 6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f、普通HiFiGAN f98b760e0e5fd0019cffd3a9d22bab5c4c2fe38491532d7680a8ad07eaf3e8dd，上游commit c616845c4e309e24d5927f15adbdf277a3d65358。
失败保留：Whisper首次PyAV metadata_errors不兼容，revision02仅改标准PCM16 WAV读入；来源审计首次读取不存在W0258失败，revision02只验证W0257。没有把缺失相邻ID当确定原因。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 当前本地HEAD `2b71639c3b555293375d248350999377168d515d`，未验证GitHub可解析。
- 最新报告 runs/phoneme_direction_20261026_repair03/report.md；来源 source_archive_audit_revision02.json、source_transcript_neighbors.json、whisper_content_revision02_result.json；分离 separation_quality.json；全量 full_postfilter_quality.json。

## Observations
- [result] 完整300条修复复验仍1失败，S0168=0.749654298575322低于0.750000，未准入。
- [result] 当前固定镜像原音中存在两套ASR识别到的冻结文字外内容，重下载成员SHA等于库存；完整逐字参考仍非人工确认。
- [decision] 保留原门、全部样本和冻结参考/TTS；开发及确认不动，不做原音旁路、裁剪评分或阈值边缘增益搜索。
- [blocker] S0168重建门失败及完整参考/配对版本问题未闭合；分阶段线性探针不足以证明不可逆信息损失比例。
- [next_step] 建立S0168完整参考的可审计校订候选与同文本配对处理方案，再评估明确修订的重建路线。

## Relations
- continues [[Codex checkpoint - 2026-10-26T04-14-10Z - Full fit reconstruction reduced to one failure with transcript risk]]
- relates_to [[逐音素差方向算法与验证方案]]
- relates_to [[逐音素静态动态差的拟合集技术校准与40人验证]]
