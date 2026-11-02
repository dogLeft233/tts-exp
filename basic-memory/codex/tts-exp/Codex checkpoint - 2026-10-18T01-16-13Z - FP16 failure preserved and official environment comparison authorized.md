---
title: Codex checkpoint - 2026-10-18T01-16-13Z - FP16 failure preserved and official
  environment comparison authorized
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t01-16-13-z-fp16-failure-preserved-and-official-environment-comparison-authorized
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 01:16:13.843453+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6-astra
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-18T01-16-13Z - FP16 failure preserved and official environment comparison authorized

## Summary
四身份准入仍有效，原FP16训练的部分epoch失败已经封存；用户最新明确授权先核查新增适配、在官方原始环境对照定位原因，并要求先不改FP32、保留失败记录。

## Story
原目标是四新身份、两seed、三训练增益的24个完整SyncTalk模型及训练/驱动/评分三轴27格。四人已独立准备并fresh统一准入，实际训练却在Brown/seed42/coarse ep5/+6发生非有限loss并停止。旧17:07 checkpoint只描述失败前状态，不能用它重启105372、133984或旧job。

用户追问为何采用官方方法仍失败。只读核查确认官方网络/loss/AdamW/EMA/FP16保留，但不是原样官方环境与入口复现：README测试Ubuntu18.04/Python3.8.8/Torch1.12.1/CUDA11.3；实际Ubuntu24.04/Python3.10.20/Torch2.5.1+cu121/CUDA12.1，且有配对RNG、whole-recording音频上下文、独立split、逐epoch重启/全state恢复和额外验证。已向用户纠正“完全按官方”的过度表述；前置资源资格不证明长程FP16稳定，立即建议FP32也过早。

最新人类指令原文：“那先核查我们加的适配代码，并在官方原始环境下做对照，查清失败原因。先不要改成 FP32，保留已有失败记录。”因此等待一般诊断许可的旧状态已被本授权覆盖。下一工作是独立命名的FP16适配审计/环境对照，不是批准FP32稳定性修订或将诊断更新并入旧训练。原失败完整保留，科学结论仍未产生。

## Working State
权威Task为Tasks/准备并核验四个新身份的 SyncTalk 完整训练数据.md（48328B接近50KB）；当前头部仍blocked/current_step4，须改当前头部或拆关联任务卡记录新授权，不按epoch追加。Spec为Experiments/SyncTalk 四新身份官方完整训练电平实验 Spec.md，IMPLEMENTED/EXECUTION_FAILED。研究地图已记录数值执行失败。ADR038–043保持，原数据准入SHA fc437965bbf097a771f2d952508b688d981c52b65a99e266dadaa68290040a84、selectionSHA cc27679ed33817158f138d33864d9f6ccea16279fb85482f70ae5a78235792c7不撤销。

ROOT=/dev/shm/synctalk-multispeaker-data-20261017；FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017；Ubuntu wjj@192.168.50.2。完整失败权威目录controller_qualification/nonfinite_partial_epoch_failure_attempt01：failure.tar.gz 20成员逐SHA通过，archive SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72，原路径保留。独立prefix诊断与原草案在该目录；旧12案例FP16/FP32草案未执行且不符合最新先不改FP32约束，不能直接启动。

原supervisor105372 failed退出；observer133984已training_stopped_storage_observer_stopped，完成14/errors[]；remotejob610686/child610696 rc1。新只读ps未见train_epoch或supervisor，V100约322MiB占用为其他进程。实际记录4636attempts/4484成功/152skips，最后global28636/累计28480，minimumscale2.938735877055719e-39，firstscale<1 global27842。controlled_training.py:66在原scaler.step/update之后、trace写入之前断言，失败步NaN/Inf和optimizerdelta未知。global28637/frame2931只是参考推断输入。无partialepoch完整state，最近+6全state为ep4；诊断从ep4重放必须另列，不能冒称重建丢失失败state或无损续训。0/−6 ep5各6000attempts/5998成功，累计29994/29995；ep5三档块/粗phase/24模型/27格未完成。原ep4三档计数不等是中途观察，当前停因非finite而非phase预算裁决。

完整数学源快照仍attempt26；27为首AMP、28为中途计数不等，29 metadata_replacement_nonfinite_partial_failure_attempt29为23文件失败/Task/spec/map/只读诊断/草案补充，不冒充新的225全数学源。所有旧false与sourcepins保留。原分析合同SHA9ede60017cb5bfe5cbdfc66c47b98115d4e8081e00d64905440c2e64cf75adc2与FP16/defaultGradScaler/no fallback、65536rays/maxsteps16、60000/102000attempts、阶段末成功更新严格配对保持。

本轮CodeGraph已读controlled_training.py、train_epoch.py、execution_state.py、runtime_common.py。关键核查点：每步域RNG/occupancy RNG保存恢复、额外val中EMA与非注册counter、epoch恢复与原连续训练等价；UpdateCounter依赖register_step_post_hook可能不适配Torch1.12，Python3.8不支持Path.is_relative_to；官方环境对照需明确只做接口兼容、不得误称零适配。当前未修改这些原冻结源，也未运行新数值诊断。

空间实测Ubuntu root559MB、SHM约1019MiB、/run/user/1000约3.2GiB空闲，RAM available13GiB，C约12.74GB/F2.29GB。无docker/proot命令、sudo -n无权限。已有autoavsr Python3.8.20/Torch2.1.2+cu121不符合原版，不能冒充。正式原环境部署尚未完成，需要隔离环境/编译与真实版本验证，不能改共享venv或清未验原数。失败前保存的空间优化observer不盲重启；旧cold worker125192已2/2完成，原失败stage1重复tar已退休，访问须依locator显式恢复，不重复上传/cleanup。原BASE Cathy像素退休尚未执行。

synctalk-27 ACTIVE，最新保存prompt仍旧“等待修订许可”，需要更新为本用户已授权适配核查+官方环境FP16对照，同时保留原实验未完、无FP32、失败不覆盖与安静通知政策。所有24模型/27格/独立QA/报告/持久交付后才停用并保留chat。

## Changed Files
本轮仅scratch只读证据与本checkpoint写入脚本；原数值源未改。既往失败封存、Task/spec/map和attempt29已经存在。Git工作树1037项dirty含大量其他任务；无新commit/PR，不全stage/clean。SHM、临时环境、scratch均机器局部易失，F archive/sourcecaps才是恢复证据。

## Verification
先前失败20成员独立SHA及4636真实prefix sampling/pixel/ray/LR对0dB精确由已封存review支持；本次再次计算failure.tar.gz全SHA一致。只读检查官方README/requirements、实际runtime/OS/工具链、CodeGraph当前适配和SSH进程/容量。原官方训练loop未见非finite loss guard，不能把它可能继续运行理解为数值健康。根因仍未知，低scale仅症状。尚无新FP16对照/FP32/新正式训练。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)：本次GitHub API确认canonical URL；当前branch openPR查询[]。
- 本地Git `2b71639c3b555293375d248350999377168d515d`：GitHub解析422，未确认推送，不构造commit链接。
- [官方固定README](https://raw.githubusercontent.com/ZiqiaoPeng/SyncTalk/43079fb3f349675f6cb33031bb07262f7566cdd5/README.md)
- [官方训练脚本](https://github.com/ZiqiaoPeng/SyncTalk/blob/43079fb3f349675f6cb33031bb07262f7566cdd5/scripts/train_may.sh)
- [Torch2.5.1 GradScaler](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/amp/grad_scaler.py)

## Observations
- [result] 原Brown/+6 ep5数值失败已完整封存，已记录prefix与丢失失败步分开；四身份数据准入保持有效。
- [decision] 最新用户授权适配代码核查和官方原始环境FP16对照，先不改FP32，保留失败；旧待答草案不能替代此最新范围。
- [blocker] 非有限值首发模块与原因未定位，原环境尚未部署，根盘容量限制必须隔离处理。
- [next_step] 冻结独立FP16审计合同，核查适配/恢复等价并部署验证官方Python3.8.8/Torch1.12.1/CUDA11.3环境，进行同输入同state的有界对照定位首个非有限值。

## Relations
- continues [[Codex checkpoint - 2026-10-17T17-07-16Z - Four identities admitted paired training running and byte exact storage qualified]]
- tracks [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[043-synctalk-byte-exact-epoch-cache]]