---
title: Codex checkpoint - 2026-10-02T18-01-33Z - TTS mechanism concluded with complete
  portable delivery
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-02-t18-01-33-z-tts-mechanism-concluded-with-complete-portable-delivery
username: dogle
repo_root: E:/Documents/tts-exp
agent: codex
repository: dogLeft233/tts-exp
project: tts-exp
git_sha: 53b8cb3e8dc005b0ea81a693dea881ecde3afd62
branch: codex/tts-mechanism-20260930
started: '2026-10-02T18:01:33.7985617+00:00'
cwd: E:/Documents/tts-exp
model: gpt-6.1-sol
capture: deliberate
status: open
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
hostname: LAPTOP-E133GSLE
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-02T18-01-33Z - TTS mechanism concluded with complete portable delivery

## Summary

TTS增强机制主线已完成：正式报告25、独立声学与口型复算、完整大包恢复和最终便携恢复均通过，BM Task已DONE。

## Story

用户要求从本地Git、代码和网盘恢复未同步的TTS增强TFG研究，持续探明原因，特别寻找局部声学特征。ROOT维护理论、实验主线、BM和Git；实验与代码交由gpt-6-luna/max代理。我们沿固定评分音的生成端干预、生成/评分两流的native比较、音素平均谱形与内部轨迹移植、独立口型几何建立证据。当前成果是有清楚模型/语料边界的机制结论及可恢复交付。

## Working State

权威科学入口为 `basic-memory/docs/experiments/25-tts-tfg-mechanism.md`，BM permalink `tts-exp/docs/experiments/25-tts-tfg-mechanism`，raw SHA `7ca5965a8537d5ed0934aef943192560080b933ac95a61003a9c47ecf059724b`。范围AISHELL-1中文/cloud Qwen/Wav2Lip。gain改变log-mel及实际编码方向，在固定原评分音下带来正响应：男40speaker方向g20 C+.161[+.094,+.232]，内部mel+.171[+.101,+.245]。男native q00 RAW/RAW→q11 FIXED/FIXED gap-null配对缩减+.204[+.119,+.295]同时改变生成和评分两流；另一个固定A生成响应excess+.185[+.108,+.266]及女20speaker gap缩减+.245[+.118,+.388]分开报告。

局部谱形机制是接收音频的相对频带比值轨迹与发声/时序结构相容性。mu/rho干预保留自身F0/AP/UV/时钟/逐帧SP功率/全局RMS。Natural donor→TTS recipient rho−SELF_GRID固定A/raw/g15 C−.192[−.336,−.047]；joint−grid两方向负。FULL35独特局部负端点集中n/ŋ及少量早tɕ/晚a˥˩；约350ms查询覆盖包含共发音，guard15局部CI含0，不指定独立phone或单一formant中介。mu−rho/交互直接区间跨0，不按显著数量排名。54几何唯一联合99%正项为同方向rho−SELF_GRID开度+.001518908[+.000085767,+.002952050]，其余53跨0；开口增加与固定评分音C下降可并存，评分、运动、native估计量分离。

完整统计800视频/1600评分，11776计划端点=10304可估+1472短Tg20不完整，40规划分母保留，FULL/VALID重复不作复现。clean统计restore646文件及160phonemaps，ROOT665600比较/共同20000×40bootstrap PASS。54几何Primary/独立重建均实际exit0，44数组残差0；ROOT跨NumPy第二数学复算178项检查中派生量最大8.88e−16，在原容差内，54零分类一致。41页统计图和6张几何QA图已ROOT实际查看。

来源索引 `.codex/scratch/source_registry_report25_attempt08.json` raw `dd62486123cf1f485ec65c9c62cd2c6e1a53b3614b0e3d1e54de25b610e19eb6`；digest audit08实际150检查全部PASS，ROOT另外核146rawpins和适用字节数。完整fullvisual包已实际恢复1898成员/493892958B/29封印；ROOT qualification `.codex/scratch/shape_visual_family_archive_attempt10/full_visual_root_qualification_attempt03.json` raw `44428b1853cdc89185d5116ff9927826b98a1e3d85ba72aff9e19feffa78c418`。此前F15.7GB、C5GB、model1.34GB、stat60MB亦实际恢复PASS。源未删除/退休。

最后便携包 `.codex/scratch/world_phone_shape_final_delivery_attempt01/portable/` 已actual builder VERIFY→sidecar-only stage→portable restore PASS，88成员/24468297源字节/4分包/5外部大包引用。actual restored tree `portable/restore_probe/`；原完成receipt `control/portable_restore_attempt01/portable_restore_completion.json` raw `f4160c903bef4dff9979bb1c30c5f330a4b1d874c464c326ad6b9007e2c1870f`。ROOT独立重新哈希88source+88restore、tar/4parts、5refs及canonical元数据全部PASS：`.codex/scratch/final_portable_root_readback_attempt01.json` raw `5bc13c58f68f2f569eb3ebf07729f9e2bdb2fbf9a599e8476a3663ee5e84e9c3`。

最新包外completion `.codex/scratch/report25_research_completion_attempt01.json` raw `347a4363909bc10f45dd2f7130d362cabc2e60e694f2c3c65b75203a770d321c`，status RESEARCH_COMPLETE_WITH_ACTUAL_PORTABLE_RESTORE，绑定最新TaskDONE/报告/Experiment/ADR004/005/CONTEXT/HANDOFF/Git与实际恢复。冻结88-member包保留Task ACTIVE-before-pack快照，最新DONE在包外绑定，避免归档完成状态的自引用。报告和Experiment CONCLUDED；ADR004数值场实现完成、ADR005保持估计量边界，decision本身open用于后续维护。

### 持久与机器本地状态

科学源码11文件已scoped commit；本轮CONTEXT单文件commit `53b8cb3e8dc005b0ea81a693dea881ecde3afd62` 在 `codex/tts-mechanism-20260930`，local unpushed/noPR。CONTEXT92行、HANDOFF28行；两者冻结包内的历史checkpoint指针不改写，最新本checkpoint通过BM lineage可达。工作树中旧legacy/tests修改及untracked实验attempts仍在，未整体stage或清理。BM/runs/scratch是机器本地内容，靠已实际恢复归档持久保存。

SSH地址 root@connect.weste.seetacloud.com:12349、host ED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0 已在server-access持久记录；密码/API仅runtime。PrivatePTY50274与bridge2133易失，研究交付不依赖其继续存活。本地Python3.12/NumPy2.3与remote3.10/1.26跨版本核验保留原门。CodeGraph无可用索引已确认后rg fallback。Windows BM stdin Unicode曾失败且未产生部分note；改用typed --content argv后实际写/readback通过，保留失败attempt。

## Changed Files

正式报告25新增；Experiment、ADR004、Task更新；CONTEXT与HANDOFF更新；独立源索引、完整fullvisual归档与ROOTqualification、最后88-member便携包和外部completion已产生。全路径与raw pins见报告及包外completion。

## Verification

所有科学复算、图形QA、完整归档恢复和本轮ROOT最终便携核读为实际PASS；BM报告/Experiment/ADR/Task正文与类型/状态均实际直接readback。未运行pytest或unit tests。同chat55/legacy0搜索、直接前checkpoint身份、Git状态/GitHub canonical/noPR/pushedbaseline在本次写前核验。所有失败历史与原支持/容差保留。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local unpushed HEAD `53b8cb3e8dc005b0ea81a693dea881ecde3afd62`.

## Observations

- [result] 正向增益编码路径和局部接收音频相容性已定位，独立口型端点将运动与SyncNet分数分开。
- [result] 正式研究Task DONE；完整源恢复与最终便携交付均实际核验完成。
- [decision] 保持原支持集、容差、失败来源与score/motion/native边界；最终冻结包状态由包外completion单向绑定。
- [next_step] 恢复时先读报告25与包外completion，按固定语料/模型/估计量边界使用已完成结论；扩展研究需明确新范围。

## Relations

- continues [[Codex checkpoint - 2026-10-02T17-00-12Z - Final visual archive running and evidence synthesis ready]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[音素平均谱形与内部轨迹的因果分解]]
- relates_to [[25-tts-tfg-mechanism]]
- relates_to [[004-canonical-float32-score-field]]
- relates_to [[005-separate-score-motion-and-native-estimands]]