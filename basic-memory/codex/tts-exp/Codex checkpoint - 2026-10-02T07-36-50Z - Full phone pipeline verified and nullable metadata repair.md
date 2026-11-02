---
title: Codex checkpoint - 2026-10-02T07-36-50Z - Full phone pipeline verified and
  nullable metadata repair
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-02-t07-36-50-z-full-phone-pipeline-verified-and-nullable-metadata-repair
username: dogle
project: tts-exp
capture: deliberate
status: open
trigger: compact
repo_root: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
repository: dogLeft233/tts-exp
cwd: E:/Documents/tts-exp
git_sha: ab40ffafdabaf113fc9b2961111d5eec61017ff3
agent: codex
hostname: LAPTOP-E133GSLE
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
model: gpt-6.1-sol
started: '2026-10-02T07:36:50.4000430+00:00'
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-02T07-36-50Z - Full phone pipeline verified and nullable metadata repair

## Summary

全1600评分距离场核验已封闭；完整160方向音素映射与局部声学窗口审计通过，fresh18修复静音metadata的null转换，完整11776联合分析仍待实际完成。

## Story

用户要求从本地BM/Git/百度盘恢复TTS增强TFG研究并持续推进到基本探明原因，尤其局部声学原因；最新“继续”维持该目标。ROOT维护理论、主线、BM和Git，gpt-6-luna/max代理负责代码/实验。已合格证据建立增益→log-mel→编码器方向→嘴部运动→固定音频评分链条，以及鼻音邻域约350ms感受野谱形响应。剩余mu/rho与SELF_GRID分解细分平均谱形和内部轨迹，不能停在排除项或启动任务，须完成联合族、独立重建、全部图与报告25。

## Working State

### 科学冻结与合格可恢复来源

五V×两A，160方向/80片段/40说话人/800视频/1600评分；q00/q10/q01/q11与SELF_GRID保持自身F0/AP/UV/clock/length/逐帧SP功率/全局RMS。SELF_GRID为两次端点保持插值。9contrast/3456整体+8192音素+128局部整体=11776；共享100全A置换、20000×40共同99%max-|t|。短Tg20保持40规划/INCOMPLETE，不改39分母。MODEL14实际complete/exit0/467925calls零mismatch，不重跑。

模型包05在 `C:/Users/dogle/.codex/tts-exp-artifacts/20261002/world_shape_gpu14_bundle05/`，193片/1344285321B，tar raw994e95170f3013d138e15627fd6d3a79b281f026b3fc64e97188bafa32e603b6；restored_verified_attempt02完整4340文件/1389777066B物化，ROOT第二次全集合/size/SHA PASS。证据 `.codex/scratch/world_phone_shape_mean_dynamic_gpu_attempt14/readback/root_restored_disk_readback_attempt01.json`。历史archive03 rawb00a15b1e6324e3be132a15deb822712c5031d2b9f3334b4460c276cada78c94，完整7685成员/6784文件5027202296B逐文件实际transient恢复/fsync/双SHA及33bindings/12gates/570本地证据、before/after库存ROOT PASS；不是全5GB同时物化。报告 `.codex/scratch/world_phone_shape_mean_dynamic_archive_attempt03/independent_readback_report_attempt02.json` rawc829fb72e807a7d0179b8621da238867712469c21b2b5dfb3e59a6037b04ca20。旧F归档、所有失败partial、远程源保留，source_retired=false。

理论入口 `.codex/scratch/mechanism_synthesis_outline_20261001.md`。旧RMS/MEL/WORLDv6/Local08/视觉09已合格；当前figure08全部29来源/产物与实际审图PASS，不含未来mu/rho效果。新 registry `.codex/scratch/source_registry_report25_attempt01.json` rawb0d620068f99b344a57a6ccccc2acd8e08e7b6945a3e3b31eb355f38a8211462由代理实际创建，ROOT尚待全文读回。女性paired gap-excess fullA/raw/guard20实际源 estimate0.2453916459083557/99%CI[0.11801328821301457,0.3879915927994252]已ROOT核对精确pointer。FIG05 converter来源不支持精确绑定，禁止补造；可绑定已合格图片和值来源。

### 完整距离场审计与consumer17真实失败

ROOT `.codex/scratch/world_phone_shape_mean_dynamic_consumer_attempt14/root_canonical_field_release_readback_attempt01.json` status ROOT_FULL_1600_NUMERICAL_AND_INPUT_CONTRACT_READBACK_PASS：全部1600唯一行/3200 raw+unit cross/32400480元素在原1e-4内0miss,max1.9073486e-6/2ULP；zero298580元素max9.5367432e-7/2ULP。saved cross/zero→lag逐位一致，12780非空metric组残差0，20空组按冻结保留。1600score+160context NPZ/930028745B绑定hash、六组float32/finite/nonnegative/shape PASS。NumPy重算curve370/396180越过原2e-6,max4.7683716e-6。ADR004保留独立cross/zero原1e-4、savedlag精确、metric原2e-6，observed/null/local统一已核验冻结CUDAfloat32距离场，不增容差、不改科学。

Consumer17 raw0aea5d17558c3985453119525890529f5a24addac41539b07d6b49a52d355470/family09 rawec678ea13fc3cc5dba364d272eb590cb7e93597120b8adccf0dfc815a54d03ff，fresh CODE/JOB/OUT14。ROOT全部14requests/7sources及完整5预检文件wire+disk/160ctx1600rows320桥接4327库存11776规划PASS。精确10分析release rawc3db641527927867ea8c8be21610ea215f68cdd81d8bceb83049d8ecfe2d40f7。Controller962390/ticks749587557、worker962402/ticks749587573实际06:54:33Z启动，06:56:14Z exit2：_phone_intervals第510行int(None)。完整5实际失败文件ROOT readback PASS，`runs/recovered_20260930/independent_validation/world_phone_shape_mean_dynamic_consumer_attempt14_analysis_failure_readback/`。执行失败不写科学阴性，14partials不覆盖。完整距离验证仍有效。

### 空值诊断与完整音素流水线

全160方向5803全tier/5344matched/459unmatched，432null均空label、unmatched、未选中；5371整数，0missing/invalid，另11selected-unmatched不改既有matched筛选。`scripts/experiments/independent_world_word_index_null_audit_20261002.py` rawd56411deee35171e910fe8c3811f639f5a1869fb5cd3c21c7bf98d8d682caa93，receipt `runs/recovered_20260930/independent_validation/world_word_index_audit_attempt01/word_index_null_audit.json` rawfc94208b8f7354a08d71efa6afcb0d0a63289085e0137c722acb6fbfc7f889d8。ROOT第二次.NET全160sidecar/SHA/token逐行readback PASS。word_index只是metadata，不参与ownership/weights/boundary。Fresh18仅保留None而非int(None)，不删完整tier；raw85f5c83ab1c0a5ab2eaeae5b9af4b385b447a423a3d9a42f385439136d89847b，ROOT minimaldiff与代理审阅PASS。

完整phone pipeline实际PASS：`scripts/experiments/audit_world_shape_phone_pipeline_20261002_attempt01.py` rawdb497f24b2eb0f076d8cfb775ab6abb9d3a8f3410eacc32b6019dd7b1a4bc478；`runs/recovered_20260930/remote/world_shape_phone_interval_audit_attempt01/phone_interval_pipeline_audit.json` rawdb0c89f67b523a3952107c8082e160eb4d2ca750a7a92b7cf7f08adf6745a570。160旧F tar map-member原始SHA与physical manifest/sidecar嵌入payload、160context JSON+NPZ SHA、五cell官方80Hz/25fps/16melstarts、full query+5frame CSR/四support闭合；14929queries/429886unioncolumns。独立centerweights/owners/boundary逐位相等，STFT50最大误差1.3322676295501878e-15。432None保留；与metadata临时sentinel诊断的所有下游weights/counts精确相等，不读score/effects。此audit不重新hash整个15.7GB tar，先前归档已合格。ROOT源中央段补读与顶级receipt已读，全部160receipt逐行最终门限摘要待完成。

Fresh family10 raw07c57208a3b7355bb7157aa89f5f96be519c8e92c6fd572753e512e28bc58767、worker15/controller18/CODE-JOB-OUT15正在冻结。ROOT预备creator15与09只读fullpreflightrequest，未执行。ROOT审阅fresh全部pins/请求后00–07 ENV、inputrelease、08预检、09完整读回/精确分析release，再实际完成11776。

原recon07 prepared/未执行，另发现worker把compact family CLI summary当full receipt读取manifest SHA会必失败。Fresh08 scientific recon rawdf83fe7a5afc1d8c7a92bbc8606899c9589dfa190cb0a3b9668244f26e6395c6保留nullable；worker08 raw69c787b897f0ac1adfdc7338109767086b97f1f3c15d5647236023d8f4bb2674；launcher08 rawb5b287c6bd1b7e3f010a1ce6fc5c4b8e4a0e3bd2919b226c20992bc2e60088a2由代理freeze，完整request index仍待。Full family receipt用于manifest/cohort/family，compact仅状态/覆盖/receipt路径/日志SHA，counts相互一致且subprocess returncode0；pin18/10/namespace15，ROOT尚待源/整包审核和实际运行。独立复核8320local×80窗口定义/权重/观测-置换G，再11776共同bootstrap；不声称第二次全V×A。

### 最终交付与易失入口

renderer03 rawa7093a1585c88cf86fbdda634d5e800315077797ae4bd04c82803be27de5b2bd prepared：完整11776CSV/global9contrastPDF/local全部PDF；g15/Cfixed预选144overview端点，不删g20不完整。最终small segment builder/portable restore/明确member selection均prepared，需结果/family/recon/报告/图完成后实际完整segment恢复；已合格bulk03/05用绑定引用避免复制。

私有PTY50274/hiddenbridge05PID217452/port2133易失，当前健康；服务器root@connect.weste.seetacloud.com:12349与hostpinED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0已在既有BM `tts-exp/docs/reference/server-access` 保存，本轮实际搜索命中。密码/key仅runtime，不入文件/BM/argv/日志。MODEL CODE12与PSF05封存树不变，禁止误换namespace15。Git HEADab40本地未推/无PR；不stage-all无关dirty/tests。CodeGraph目录存在但未索引，实际拒绝explore后rg fallback。

## Changed Files

Task实际最新5375chars/current_step4仍记录17/14 nullableFAIL；ADR004实际3385chars/full_distance_audit_pass_nullable_metadata_blocker。须同步160pipeline PASS和fresh15实际阶段。CONTEXT91行dirty，ignored HANDOFF67行已正确记载AP/fullnumericPASS/17失败与下一步，pipeline新PASS待补。18/10、null/pipelineaudit、recon08/source registry等未跟踪本地；scratch/runs/BM与bulk可恢复源状态区分。

## Verification

实际模型和历史全成员恢复复核PASS；ROOT全1600score数值/输入闭合PASS；17/14完整预检PASS后真实分析FAIL、5失败文件读回PASS。432null全tier ROOT独立.NET核验PASS；160map原字节/完整phoneRF pipeline实际PASS、源和receipt原SHA已ROOT读回。未运行pytest/添加unit tests，只有科学/source/contract检查。Router/实验指令/Coding Session及配置实际加载、同chat和legacy全页搜索、AP身份及Git canonical/noPR/pushedbaseline在创建本checkpoint重新只读确认。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local unpushed HEAD `ab40ffafdabaf113fc9b2961111d5eec61017ff3`.

## Observations

- [result] 全1600评分独立距离/serialized指标闭合；完整160方向5803recipient tier及14929局部查询的RF定义核验通过。
- [decision] 保留432静音metadata null与完整tier，ownership和matched权重规则不变；失败来源/partial不可覆盖，fresh18/10/namespace15只做执行修复。
- [blocker] 完整11776因果族仍未实际完成，fresh15与recon08未ROOTrelease/执行；最终报告25、图、可恢复smallsegment未交付。
- [next_step] ROOT完成160pipeline逐行门限读回并审阅fresh18/10/worker15/controller18全部pins，执行完整11776联合分析。

## Relations

- continues [[Codex checkpoint - 2026-10-02T06-47-59Z - Canonical score field verified across full factorial]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[音素平均谱形与内部轨迹的因果分解]]
- relates_to [[004-canonical-float32-score-field]]