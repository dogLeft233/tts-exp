---
title: Codex checkpoint - 2026-10-02T06-47-59Z - Canonical score field verified across
  full factorial
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-02-t06-47-59-z-canonical-score-field-verified-across-full-factorial
git_sha: ab40ffafdabaf113fc9b2961111d5eec61017ff3
branch: codex/tts-mechanism-20260930
repo_root: E:/Documents/tts-exp
model: gpt-6.1-sol
username: dogle
status: open
repository: dogLeft233/tts-exp
started: '2026-10-02T06:47:59.3486326+00:00'
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
capture: deliberate
agent: codex
trigger: compact
cwd: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
project: tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-02T06-47-59Z - Canonical score field verified across full factorial

## Summary

完整800视频/1600评分的可恢复来源已合格；全部1600实际评分的数值审计定位CUDA与NumPy float32路径差异，fresh17保留独立距离校验并统一观测/置换/局部指标使用的冻结距离矩阵。

## Story

用户要求恢复未同步的TTS增强TFG研究并持续推进到基本探明原因，特别寻找局部声学原因；最新“继续”保持这一目标。ROOT维护理论、主线、BM和Git，gpt-6-luna/max代理负责代码和实验。已有合格证据建立增益→log-mel→编码器方向→嘴部运动→固定评价音频评分的链条，以及鼻音邻域约350ms感受野内的谱形响应。剩余冻结mu/rho与SELF_GRID分解负责细分平均谱形和内部轨迹；不能在预检或排除项停下，必须完成11776联合族、独立重建、图和报告25。

## Working State

### 冻结科学与可恢复来源

理论入口 `.codex/scratch/mechanism_synthesis_outline_20261001.md`。五V×两A，160上下文/80片段/40说话人/800V/1600评分；q00/q10/q01/q11与SELF_GRID保留自身F0/AP/UV/clock/length/逐帧SP功率/全局RMS，SELF_GRID两次端点保持插值。3456整体+8192局部音素+128局部整体=11776；9contrast、共享100全A置换、20000×40共同99%max-|t|均冻结，短Tg20保持40规划/INCOMPLETE而不救为39。模型14实际complete/exit0，467925calls/0mismatch，不重跑模型。

完整模型包05 `C:/Users/dogle/.codex/tts-exp-artifacts/20261002/world_shape_gpu14_bundle05/` 193片/1344285321B tar.gz raw994e95170f3013d138e15627fd6d3a79b281f026b3fc64e97188bafa32e603b6；`restored_verified_attempt02/` 全4340文件/1389777066B已实际落盘，ROOT第二次全成员集合/size/SHA核验PASS。证据 `.codex/scratch/world_phone_shape_mean_dynamic_gpu_attempt14/readback/root_restored_disk_readback_attempt01.json`。评分jsonl rawb85e9064e70c90624d4dcddf8e8bd02c9111ffc205e75d65b3f3d7afa0d4527e；manifest55e34a1ffd13f61d27e7a9b9b630a2514cce98c40f4812340a658b581947bdd0。

历史archive03 `C:/Users/dogle/.codex/tts-exp-artifacts/20261001/world_phone_shape_mean_dynamic_20261002_attempt03/` tar5040926720B rawb00a15b1e6324e3be132a15deb822712c5031d2b9f3334b4460c276cada78c94；ROOT独立全7685成员/6784文件5027202296B实际逐文件transient恢复/fsync/stream与disk双SHA、33绑定/12gates、570local证据438731946B与before/after库存PASS，随后全ledger再读PASS。报告 `.codex/scratch/world_phone_shape_mean_dynamic_archive_attempt03/independent_readback_report_attempt02.json` rawc829fb72e807a7d0179b8621da238867712469c21b2b5dfb3e59a6037b04ca20。不是全5GB树同时物化。source_retired=false，旧F归档与失败partial均保留。

### 完整数值审计与实际分析失败

Consumer15存储schema候选仅比较producer实际14keys/empty3keys，D0/search_gain从curve派生。ROOT独立全部1600 score NPZ/hash/12800支持×geometry审计PASS：12780非空+20空，curve/scalar残差0。`.codex/scratch/world_phone_shape_mean_dynamic_consumer_attempt11/metric_schema_parity_root_readback_attempt01.json` raw686158be0c52e0e58c71ebfba7d894780ef6ad7cb4eb287d8f359b6a4595010d。此核对仅saved lag闭合，不能混称CPU cross闭合。

ROOT实际160 maps/token auditPASS：5803全tier/5344matched/459unmatched，11个冻结selected-label unmatched token暴露于7ctx，160phase/support countcells不同。Consumer16仅将诊断token计数筛matched，完整tier仍用于ownership/ambiguity，RF权重及统计不变。审计 `.codex/scratch/world_phone_shape_mean_dynamic_consumer_attempt11/matched_phone_token_counts_actual_audit_attempt01.json` rawaf0465b65c9c5c6351f576cfee16d4d8a90d4a47a71bc5175298893e0d2ef63f。

实际consumer16/family08/worker13/controller16位于fresh CODE/JOB/OUT13；来源分别96f7bdeac8a937d40f6d677abf62b3d1f875fb6b0de210f97bd2c133c85dce0f、9094972bbf264c622dc50d412c267047f1ac1723c7a8059a1ee60d74697bb615、1c1eb1ffa33f4aae7260c5145184cd9582c59c9f10e61d742011c6faf5fdd03d、9c203a6b95113a0347e290958ae4ea916e06cb4a78c88f90266b39089899c738。ROOT全部14packets/7sources、完整预检5文件wire+disk/160ctx/1600rows/320桥接/4327库存/11776规划PASS后，exact10分析release124424a58890d1098e4ea3640c9391a349f9d49543f1177c727e1543edcc0b62。Controller940813/starttick749337754与worker940826/starttick749337770真实运行，06:14:35Z exit2：S0062_BAC009S0062W0121__tts_to_natural__q01__A_RAW/raw/guard15 curve reconstruction不同。完整5失败文件ROOT wire+disk/PID+ticks状态核验PASS，保存在 `runs/recovered_20260930/independent_validation/world_phone_shape_mean_dynamic_consumer_attempt13_analysis_failure_readback/`。未完成族，不能当科学阴性，所有执行来源/partial不覆盖。

全量数值audit `.codex/scratch/world_phone_shape_mean_dynamic_consumer_attempt13/lag_numerical_parity_audit_attempt01.json` raw2a8c2da877db7601c04464e357d73227e2d7ca37a78b59d569d5baab2fa9bec1，代理实际完成且ROOT直接读全文。1600唯一行/3200 raw+unit cross、32400480元素均在原1e-4容差内，0miss，max1.9073486e-6/2ULP；zero298580值max9.5367432e-7/2ULP。保存lag与producer loop和consumer vector gather均全部3200逐位一致；12780非空组saved lag→curve/scalar残差0。NumPy cross重算curve370/396180值越过原2e-6，max4.7683716e-6，首失败2.8610229e-6。补充 `.codex/scratch/world_phone_shape_mean_dynamic_consumer_attempt13/lag_numerical_input_contract_audit_attempt02.json` raw1172b6168b5bf8565a3ddc6c4eda4fa60e65b110d97d0c174d285dfd235ed944 PASS：1600score+160ctx NPZ/930028745B的绑定SHA，唯一ID，全六组float32/finite/nonnegative/shape、savedlag闭合，Python3.12.14/NumPy2.3.5。ROOT源码和补充receipt最终独立门限摘要待完成。

ADR004 `tts-exp/docs/adr/004-canonical-float32-score-field` 已创建/实际读回2241chars；决定在看effects前记录：独立重建V/A→cross/zero仍用原1e-4，随后验证saved cross/zero→saved lag逐位相等，原2e-6严格metric检查从该lag重建，所有observed/null/local用同一份独立验证的冻结CUDAfloat32 cross/zero。不增容差、不改定义、不重跑模型。Fresh consumer17 raw0aea5d17558c3985453119525890529f5a24addac41539b07d6b49a52d355470/family09 rawec678ea13fc3cc5dba364d272eb590cb7e93597120b8adccf0dfc815a54d03ff已准备，机制代理独立minimaldiff审计PASS；fresh worker14/controller17/CODE-JOB-OUT14正在准备，尚未ROOTrelease或执行。

### 交付与易失状态

原recon05有63hex consumerpin缺末f，未执行且保留；freshrecon06修正为64hex/raw044a88905c0bfa594eb14a5d23efe9ef84b059b098ea844d75d25c03830c0117，仅prepared/未执行。新recon07需pin17/09/namespace14，独立8320local×80CSR/weights/counts和11776完整bootstrap。renderer03 a7093a1585c88cf86fbdda634d5e800315077797ae4bd04c82803be27de5b2bd prepared静态全合同PASS：全部9contrast/11776CSV，global/local PDF全部端点，g15/fixed-anchor overview144预选端点，不能删g20INCOMPLETE。旧fig08全29source/output与实际审图PASS。

最终small segment builder/portable restore/explicit selection helper已静态review，仅prepared；必须实际完整analysis/family/recon/control/report/fig/BM完成后ROOT明确成员列表，绑定已合格bulk03/05而不重复其字节，再实际portable全树恢复。报告25尚未创建。

私有PTY50274/bridge05 hidden217452/port2133易失、当前健康自动重连；服务器root@connect.weste.seetacloud.com:12349/ED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。密码/API key仅runtime环境，不入文件/BM/argv/日志。MODEL CODE12和封存PSF05必须保留，不能误换namespace14。Git当前ab40ffafdabaf113fc9b2961111d5eec61017ff3本地未推，无PR；无关dirty/tests保留不stage-all。CodeGraph未索引已拒绝explore，rg fallback。

## Changed Files

CONTEXT91行dirty，ignored HANDOFF67行已同步consumer16/13真实失败、数值路径审计、ADR004与fresh17/14待执行。Task4725chars/current_step4最近实际读回PASS，但仍有旧pending状态需更新。新consumer17/family09及recon06/renderer03等untracked，scratch/runs/BM和bulk在本机，最终segment待交付。

## Verification

ROOT实际全4340模型文件与7685历史ledger PASS；全部1600savedscore/schema与160maps/counts PASS；namespace13完整预检及320桥接PASS、实际analysisFAIL完整5文件读回PASS。全1600 numerical audit及finite/hash补充实际完成/ROOT全文直接读回，完整distance误差解释无新effects。未运行pytest或新增unit tests。Router/实验指令/Coding Session及配置已加载；同chat/legacy全量搜索、AO直接身份核验和Git canonical/noPR/pushedbaseline在创建本checkpoint时重新只读核实。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local unpushed HEAD `ab40ffafdabaf113fc9b2961111d5eec61017ff3`.

## Observations

- [result] 全3200几何行独立cross校验在原容差内，saved cross/zero→saved lag逐位一致，12780非空组savedlag指标残差0；失败是CUDA/NumPy float32距离场混用。
- [decision] ADR004冻结统一的已独立核验CUDAfloat32距离场用于observed/null/local，保持原1e-4距离与2e-6指标容差及全部科学冻结。
- [blocker] fresh17/09/namespace14尚未实际完成联合分析；11776效果、第二份独立重建、最终图/报告25仍未交付。
- [next_step] ROOT直接封闭全1600数值和输入receipt门限审核，审阅fresh17/09与namespace14全部pins，执行完整11776联合分析。

## Relations

- continues [[Codex checkpoint - 2026-10-02T05-48-06Z - Full metric parity and qualified recoverable sources]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[音素平均谱形与内部轨迹的因果分解]]
- relates_to [[004-canonical-float32-score-field]]