---
title: Codex checkpoint - 2026-10-10T00-48-03Z - Actual Wav2Lip and MuseTalk spectral
  response evidence
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-10-t00-48-03-z-actual-wav2-lip-and-muse-talk-spectral-response-evidence
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-10T00:48:03Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-10T00-48-03Z - Actual Wav2Lip and MuseTalk spectral response evidence

## Summary

固定8段Fit音频的Wav2Lip与MuseTalk各32项真实前端测量已完成并由ROOT逐数组复核，显示近静态谱形可造成时间变化的条件特征响应；Ditto仍在CPU设备绑定处停止，联合SPSA恢复尚待Muse共享helper模块来源修复。

## Story

用户要求持续解释TTS对TFG的增强原因，并做出适配多个TFG、未见说话人超过固定RMS的自然音频增强头。增强PCM仅驱动生成，交付/SyncNet评分保留完整自然PCM及原时钟。ROOT负责理论、协议、BM与独立资格，代码/实验仍由GPT-6 Luna/max承担。用户指出已用重置卡；原三个Luna/max确已恢复且完成实际工作，不再把旧额度错误当阻塞，也没有替换模型。

初始720四快照全部NOT_ADMITTED，step060只是最小违例初值。冻结actual initial720→20 joint SPSA→tell5/10/15/20四个full120中心→final720→portable/封闭Dev+Confirm不变。当前仍0新native科学score/0tell，工程输入验证与机制前端测量不充当新Sync-C效果。机制诊断使用固定首batch8段/4说话人，原72 natural/full/DC units另加独立24 RMS appendix，绝不反馈优化/selection/admission。

## Working State

**声音对照已经独立资格。** step060完整mask的DC平方能量占99.173%–99.617%，时间残差RMS .03916–.05720 dB，full/DC波形RMS差 .000478–.000708。DC是postprojection mask逐frequency的时间均值，同源complexSTFT/窗/N逆变换并匹配实际full RMS，无第二次tanh/peaklimiter。先前ROOT独立8段source/full/DC重构QC03 self a91ea07a221643d96bbd15a54ed0255eb7ebdc1c70d55829173a65ceec9688ed仍有效；裸θF32SHA 7a5948001c71b4bc2ee007ab5b563353d7b6ac6c44d40b041a337c6b14aef5bc不同于canonical/controller dc4b3843324d7dfa5c429690806bd981f72574ca7be8f26426fdd38ab4d76888。

ROOT实际SSH读回32个WAV及3个source文件，并对8个RMS独立F32 gain重构：.codex/scratch/root_provisional_rms_actual_readback_attempt01/root_actual_qc.json self175834351cd6d2505aa5a655d6f518d30faf8463de368c160e5ae7abde3cc51b PASS raw/PCM/N/16k monoF32/finite/exactgain，最大RMS误差1.7425665e−9。RMS apply_global_rms(target=.0376838172675746,peak_ceiling=None)，methodconfigSHA35cccbb73298f3b5b6f446be93b799268eda6c39b7b59ed8771c262adc902637。实际音频stage/appendix在 /dev/shm/fit40_step060_provisional_native_72_frontend_attempt03/，manifest自哈希分别246cb37bbb367c9f0ef0fed22aa9f69dd6e51b09fa02871cdfb92e791dcf853f/42c009fa7d217e0ddde927275a789f4041f61c757e9ab925de019cb6c986f2c8。builderchild688774/t7589610 rc0，outer Bash CRLF exit rc2保留，不重复生成。

**两类真实前端数组已完成。** Wav2Lip worker04 child693799/t7659164 rc0，remote /dev/shm/fit40_step060_provisional_native_72_frontend_attempt04/execute_attempt01/wav2lip_gan_bac9a81e/output；local .codex/scratch/fit40_step060_provisional_native_72_frontend_attempt04/actual_readback_wav2lip_attempt01/output/。receipt raw272c67a3cf8e92f3de3bca0cb10b237dd73246517a33d2b69e8b2bc599eda54f/self2b14276b0eafc2b40afbc12dbfb071322e527074fb09056d93bfd7a6d1a9442a；32 NPZ3786823B。Py3.8/Torch2.1/NP1.24，实际80×16 mel→512 encoder，CUDAhidden/CPU F32/threads1。

MuseTalk worker05 child697743/t7737080 rc0，remote /dev/shm/fit40_step060_provisional_native_72_frontend_attempt05/execute_attempt03/musetalk_fixed_portrait_attempt16/output；local .codex/scratch/fit40_step060_provisional_native_72_frontend_attempt05/actual_readback_musetalk_attempt01/musetalk_fixed_portrait_attempt16/output/。receipt142387B/raw731ed5a5cc1e80dd80eedd260a3b0537d44923f230b08bc27c260fd5e4d1a512/self708408cb4e009c8ae575edfde893f07be4c7d26c0f1b210e9df59850b887bb62；32 NPZ244798299B。Py3.10/Torch2.5.1/NP1.26/Transformers4.39.2 activeAudioProcessor/Whisper30秒全局attention、10tokens×5states=50×384。CPU F32明确不同于原生generator FP16；只作描述性前端诊断。Signal完整本地33pins与37file总树QC02亦PASS self983a203fe59f5a0834aaaa11047de3df15705288467a948309ebe4d3f1ea356c。

ROOT独立工具 .codex/scratch/root_qualify_provisional_native_features_attempt01.py 对每模型实际SSH full33 rawpins、receipt self、32输入身份/NPZkeys/F32形状/finite/query_index、48comparison和528标量重算均PASS。W QC .codex/scratch/root_provisional_wav2lip_feature_qc_attempt01/root_actual_qc.json self275529c5ee54ccbbd8b7c2bead798fd64a89ad8b8f639e76：equal-speaker mean ||Ffull−Fdc||/||Ffull−Frms||=.10074103166091411，||Fdc−Frms||/||Ffull−Frms||=1.0232036295014226；RMS→full响应平方能量的time-centered份额均值.7801476506652789，range .74236196–.80289963。Muse QC .codex/scratch/root_provisional_musetalk_feature_qc_attempt01/root_actual_qc.json self84b440fac88f0461a3257eff33ac24d15ad41200388364e93858de8a63d31454：相同两ratio=.24180333872118856/.956470419552307，RMS→full time-centered relativechange=.05531004035733513，full→DC=.014647047141935727。每模型先8clip归约再4speaker均值，不能跨模型比较绝对norm。证据支持此近静态EQ通过输入随时间变化的谱内容产生时变条件轨迹，而不是只加常数embedding偏置；norm ratio不是加性因果份额/SyncNet收益或视觉同步证据。仍需Ditto和原clock40ms局部谱成分分析，不能把此seed机制直接等同整个TTS效应。

**Ditto仍在feature前失败。** worker04缺probe、worker05 Muse whisper-tiny→whisper真实pin路径与Ditto parsecfg data_root→weights_root/ditto_pytorch修复已ROOT完整diff/source/readback；没有改模型资产/下载/建symlink。worker05 source52510/eeb1ee8ea5bed94e199666074d11ace3d3a42eb946d2823e7411034297af2cdf，worker06只加cfg["w2f_cfg"]["device"]="cpu"，52551/3488828ae615d160bc36aa715b8e836bf255eec04621cff5844ec78168cac690。05实际Ditto child697288/t7729122与06 child698616/t7747268均rc1 `FrontendPacketError: Ditto feature path is not the pinned CPU ONNX route`。ORT请求CUDA后CPUfallback不满足hubert.device guard，nested device被构造器覆盖/未生效；0 Ditto feature、无最终receipt。Signal正读exact Wav2Feat→innerw2f→Hubert构造器与actual parsed cfg/modeltype/device/providers，先定位覆盖再单行fresh修复，禁止再猜字段或放宽guard。hiddenCUDA/threads1/8GiB有效RAM/5GiB存储gates保持，native20优先。

**SPSA控制恢复。** session02首对UNKNOWN/childrc2，actual30file663372B独立proof .codex/scratch/root_session02_pair01_actual_failure_review_attempt01/ self67fcac16dc5c6689666ab9cabf9896090d5dea627e9ba1b55fc685e5b2c0ca2c仍权威：12dispatch文件是控制/status/ledger/log，render_audio_score_files=[]，3event至POINT_DISPATCH_INTENT、0score/0tell、captured PID gone；禁止blind replay。修复runner07正确四字段pin与point_id、dispatcher09一sample只apply一次再三模型projection、child05/center07绑定；同θ/RNG/方向/科学协议保持。

package09实际封存于 job_control/fit40_spsa_execution_package_attempt02/execution_package_manifest_attempt09.json，27453B/raw0b28cb8e97d169aa61c8e80b029880a4cfab06f7e38f0d03be978724a93b8b0c/self f16a665dc194e39a860c8ffa0b48d9ea77028818c2b3683e19fa2a279396e133。runner07 rawd0e8f06c121ea52ca6ed695d8e5caf3fdd28a9a00edc1b1c35c648f9a9df2a2e；child05 d7f5e77c616c928a2bddba99497f3a0b5ec0540aac67f8045bcdfb301e4147f1；dispatcher09 e63d51118a6e469a9731c1f3e23797808eb647f6a6609e4dabca1a4fbbe26ba5；center07 39999903f12058091ff5e1488310e5219a0a00380c1dd0a704907ec929f17840；Museadapter11 80476B/7583acc23d311c7b9b69dd45607644fc58ea9f257e679599fb24f00dd13a630f，用source-plan-derivedhelperroot，nativeproject_root仍资产。ROOT实际远端完整6file byte/self/AST与07→09仅既定pins diff PASS：.codex/scratch/root_package09_actual_source_readback_attempt01/root_actual_qc.json self06f4f25f92d7dff537548ba915c0be25472475a4043b6508ab2634f50ed10595。package09已资格source但不是所有adapter runtime通过。

ASR04已生成±8共享WAV/±3×8projections，Wplus纯验证先通过。新validator05全6先因sidecar07/request09绑定失败；fresh transport-onlyrebind只改packagepin+self，其余canonical内容相同，receipt15636B/rawd6d4a6b45f203495cb314efcc3c759180ef6abaf6957c7c0b4d1fdb0b25b12d8/selfbbbd592d673a68fd31b7404a726fdf1ce35d79024b80bcda85c5483e67eba728，未重算波形。actual validator06 W±/Ditto±四PASS，Muse±预建strictdir违反freshleaf precondition失败。fresh07只跑Muse±并保留四PASS，此次进入adapter后helper07→05 `_verify_loaded_module`拒绝audio_io unbound source origin，仍0model/render。ASR正在比较actual expected/loaded路径再窄修adapter，不能松来源检查、改helper科学函数或复制另一闭包。需修复后新packagepin及sidecarsrebind；不要再重复波形/全面初始720。

bootstrap10 27866B/32d430419fafcebfe70faca498f80b3509e229a14da160f7523cf08cb9f83f0b已ROOT完整08→10diff接受。session03 / coordinator10 / lineage03尚未写完或启动。Dev准备最小lineage03绑定session02真实30files、eventidentity、0score/tell/media、PIDabsence，freshsession03 event0/iteration0/no pendingask且首ask与旧pair01完全相同；package09可能需后续Muse修复版本。不要genericcandidate_count=0误拒绝计划metadata/control文件，不做泛化恢复框架。等六模型side纯验证通过与最终sourceclosure稳定再ROOT资格session03/launch20。

**持久/机器状态。** BM Taskactive/Experimentimplemented与HANDOFF已同步RMS资格/W32实际结果，但metadata/observations仍有Wstarted/Mpreflightpending旧值；本checkpoint补充Muse32/ROOT QC、Ditto06失败、validator07sourceorigin失败，下次mutable同步要一并修正。原目录 .codex/scratch/root_sync_rms_qualified_native_frontend_attempt01/ 和 root_record_wav2lip_frontend_result_attempt01/保存before与fullactual readback。Experiment schema精确检索结果[]，不声称已加载；Router/实验指令/CodingSession/Task schemas本次已完整read。

B:/tts-exp-archives/provisional_step060_wave_diagnostic_attempt01已53file18151482B两次fullreadback并原件保留，receipt .codex/scratch/root_provisional_step060_backup_attempt01.json self78dfdf0f81b2b2a117f09354813a8f2f0642968b0c8dbbc0cc0566d4c1118a45；未含新RMS/W/M数组、无云同步保证。新完整96frontend库存完成后追加create-only备份，不删移原件。wjj@192.168.50.2:22 publickey/V100 UUID GPU-d12b3309-c299-da83-064d-a3cd619bd560；旧AutoDL westd18625 DirectD0未知不重放。secret永不入文件/argv/log。默认Python无NumPy，ROOT用 C:/Users/dogle/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe NumPy2.3.5。

Git codex/cross-tfg-audio-head-20261003/5688784dd5c6c1272074f0c4b6c5ff79f117c907 (local/unpushed)，本次canonical repo/PR[]实查；五旧tracked dirty及大量新untracked/ignoredscratch/BM保留，未新commit/push。SHM/runtime/本地raw读回未git备份均fragile；CodeGraph目录有但无index实际报错，不再重试/索引。三个Luna/max均实际工作，ROOT无新增tests。

## Changed Files

- source repairs runner07/child05/dispatcher09/center07/Muse11及封存package09、bootstrap10；旧失败/source/package全保留。
- frontend worker04/05/06；真实W/M64NPZ和32WAV与原始receipts完整本地读回；Ditto失败日志保留。
- ROOT32WAV/RMS full QA、package09六文件full QA、W/M各32NPZ与528标量独立QA脚本/receipts。
- BM mutableTask/Experiment/HANDOFF W32/RMS已更新，Muse最新结果及metadata尚待同步。

## Verification

ROOT实际执行32WAV输入身份/8exactRMS重算PASS，package09六完整原始文件readback PASS，W和Muse各33file SSH byte比对/32NPZ/48comparison/528标量PASS。没有运行新tests、Dittofeatures、video/SyncNet、tell或中心score。ASR四purevalidators PASS与Muse07origin失败是agent实际回执，尚待ROOTfullartifact资格。SignalMuse完整byteQC已PASS。不存在新的跨TFG效果已超过RMS的结论。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；local SHA `5688784dd5c6c1272074f0c4b6c5ff79f117c907`不构造未证实GitHubcommit链接，branchPR[]。
- .codex/scratch/root_provisional_rms_actual_readback_attempt01/root_actual_qc.json
- .codex/scratch/root_provisional_wav2lip_feature_qc_attempt01/root_actual_qc.json
- .codex/scratch/root_provisional_musetalk_feature_qc_attempt01/root_actual_qc.json
- .codex/scratch/root_package09_actual_source_readback_attempt01/root_actual_qc.json
- .codex/scratch/fit40_spsa_candidate_rebind_attempt01/

## Observations

- [result] W/M各32真实前端测量与ROOT完整数组资格PASS；近静态DC与full差异相对RMS→full均值约.101/.242，支持时变输入响应机制的8clip描述。
- [result] 四个W/Ditto±共享纯验证PASS；两个Muse07仍因audio_io实际模块来源绑定在模型前失败。
- [decision] 冻结20jointSPSA/四full120中心/final720与封闭Dev/Confirm，provisional96诊断不反馈优化准入。
- [decision] 保留所有失败与共享输入；仅最小asset/device/source-origin/control修复，不放宽科学、来源或CPUprovider gates。
- [blocker] DittoCPU构造器device绑定未通过，Museadapter helper module-origin待修，freshsession03/native20尚未启动；0新科学score/tell。
- [next_step] 定位并修复实际Ditto device及Muse helper source-origin，使96机制读回完成、六side纯验证通过，再用绑定失败谱系的freshsession03恢复冻结native20。

## Relations

- continues [[Codex checkpoint - 2026-10-10T00-16-22Z - Qualified DC mechanism audio and actual shared CPU inputs]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[030-lineage-bound-session-after-pre-native-control-failure]]
- relates_to [[三类TFG音频前端的谱缩放与时间上下文]]