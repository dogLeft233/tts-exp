---
title: Codex checkpoint - 2026-10-02T05-48-06Z - Full metric parity and qualified
  recoverable sources
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-02-t05-48-06-z-full-metric-parity-and-qualified-recoverable-sources
repo_root: E:/Documents/tts-exp
model: gpt-6.1-sol
git_sha: ab40ffafdabaf113fc9b2961111d5eec61017ff3
username: dogle
started: '2026-10-02T05:48:06.0156091+00:00'
status: open
hostname: LAPTOP-E133GSLE
project: tts-exp
agent: codex
capture: deliberate
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
repository: dogLeft233/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-02T05-48-06Z - Full metric parity and qualified recoverable sources

## Summary

800视频/1600评分和历史谱形证据均完成可恢复归档及ROOT全成员核验；完整联合分析因派生指标字段映射失败，fresh修复已通过1600实际评分矩阵的零残差检查。

## Story

用户要求从本地BM/Git/代码/网盘恢复TTS增强TFG研究，持续推进到基本探明原因，重点寻找局部声学原因。ROOT负责理论、主线、BM与Git，gpt-6-luna/max代理负责代码/实验。已合格证据建立增益→log-mel→编码器方向→生成嘴部运动→固定评价音频评分的因果链，并将WORLD谱形响应定位到鼻音附近约350ms感受野。剩余音素平均谱形mu与中心化内部轨迹rho的冻结干预负责细分谱形机制；最新“继续”保持完整分析/报告/可恢复交付目标。

## Working State

### 科学冻结与合格本地来源

理论入口 `.codex/scratch/mechanism_synthesis_outline_20261001.md`。mu/rho q00/q10/q01/q11+SELF_GRID双插值保持自身F0/AP/UV/clock/length/每帧SP功率/全局RMS，160上下文/80片段/40说话人/800V/1600评分；3456整体+8192音素+128局部整体=11776，共享100全A置换和20000×40联合99%max-|t|，9contrast不删减。短guard20保留40规划/INCOMPLETE；不能改为39。模型14实际complete/exit0，全五运行flags、467925calls/0mismatch，无需重跑。

完整模型包05已在 `C:/Users/dogle/.codex/tts-exp-artifacts/20261002/world_shape_gpu14_bundle05/` 保留193片/完整1344285321B tar.gz SHA994e95170f3013d138e15627fd6d3a79b281f026b3fc64e97188bafa32e603b6。`restored_verified_attempt02/` 全4340常规文件/1389777066B实际落盘、source/member SHA及ROOT另全4340集合/size/hash重读PASS，证据 `.codex/scratch/world_phone_shape_mean_dynamic_gpu_attempt14/readback/root_restored_disk_readback_attempt01.json`。完整evaluation_results.jsonl rawb85e9064e70c90624d4dcddf8e8bd02c9111ffc205e75d65b3f3d7afa0d4527e；封存manifest55e34a1ffd13f61d27e7a9b9b630a2514cce98c40f4812340a658b581947bdd0，planb7c6da80a380a387ef6c1ef3b7d62d1181d419ddf58d83f214891ac9827b3c85，runtime3e5629a3612a630dbb9833e5f60b527b3dcc1d8d84ec32a0fa14774a8f790bef。

历史archive03也已完成ROOT独立验证：`C:/Users/dogle/.codex/tts-exp-artifacts/20261001/world_phone_shape_mean_dynamic_20261002_attempt03/` tar5040926720B SHAb00a15b1e6324e3be132a15deb822712c5031d2b9f3334b4460c276cada78c94，18roots/7685成员/6784files5027202296B、33绑定/12gates、570local evidence438731946B。旧verifier01的32MiB JSON guard FAIL保留；fresh02只扩大已封存JSON大小边界及新probe路径，实际exit0全成员单文件transient恢复/fsync/stream与disk双hash、before/after库存、local570/旧partial全部PASS。ROOT另完整7685ledger再核验PASS。report `.codex/scratch/world_phone_shape_mean_dynamic_archive_attempt03/independent_readback_report_attempt02.json` rawc829fb72e807a7d0179b8621da238867712469c21b2b5dfb3e59a6037b04ca20，ledgerb9a83dd3fc45e164b338ae046ac49659c841efbe58a72d731ca2020924608d6e。这是逐文件临时恢复，不是5GB全树同时物化。source_retired=false；旧F完整归档和失败partial保留。

### 完整预检通过与分析真实失败

Consumer14 raw72d8bb31a99e54f928acd91ec3e95e740b41dc76f842d8da0c501108f8410c3f/family06 rawa7ebd0897b0fd68db1cb091ceb8c118c2ac1ed61f98663725103d46a04fd9ab0，worker11 raw87f90b711d6e51e9d42349313454c373a11771a3788620505f64430e9018b356/controller14 raw9a2700e268d5c38451f7ac968114f79bb4c5573ee2537a0714e49ce5aa7b0a5c：ROOT全diff/合同审核及namespace11八项staging实际PASS。13字段ROOT输入release6b171a8e2ff95a18fb6c17204db0cc963bfbebb2bb63e479cd553048913b6b0c。实际全预检98.85s CONSUMER_PREFLIGHT_PASS，160ctx/1600rows/320endpoint bridges/4327inventory/11776规划全部核对；receipt9febde485ae7c91d20c5689a88d2a54ded4b37011b4ed67b8c0c66687ecc29a0。ROOT完整五文件wire+disk SHA与counts核验后生成exact10字段分析release5216da3f474e4d6b443b518c0936f1ded41fbc02e703dc5a11c1f44cf3e584d7，授权实际CPU analysis。

Controller919121/starttick749053606与worker919133/starttick749053623真实运行后exit2，首条评分报 recorded D0 None !=11.709453582763672。producer实际序列化D_fixed_anchor/C_fixed_anchor/D_min/B_median/C_best与curve/lag-axis/optimal-column/indices等；D0/search_gain是派生量，未存储。consumer14误要求它们为存储字段，这是工程合同错误，非科学阴性。全五实际analysis文件已ROOT wire与disk重读PASS：`runs/recovered_20260930/independent_validation/world_phone_shape_mean_dynamic_consumer_attempt11_analysis_failure_readback/`；analysis_exit raw91438dde964e157aad63c48e8dcec4a6de6af9a8bc4afa465c4bfb3e1184bd44。partial输出/failed receipts均保留，family/recon未执行。

Fresh consumer15 candidate0523c59f5bd6e2c19d4680fb3b25f7807d15e1b32b265044c144ca3ed29669de仅比较真实serialized keysets/完整curve/lag与anchor metadata，再从已核验curve[15]和D_min独立派生D0/search_gain，与重建指标比较。独立代理静态PASS，remote代理已对本地合格恢复1600唯一行/全部1600score NPZ检查hash与12800 support×geometry groups：12780非空+20空，最大curve/scalar残差0。RAW/unit每种geometry guard20各1590ok/10too_short，其余支持完整。没有读取效应比较。

ROOT额外发现完整recipient tier传入token诊断但consumer计数未筛matched，recon04则筛matched；`matched_phone_token_count=len(intervals)`也将全tier称作匹配数。机制代理确认静态两处真实缺失，remote正在读全部160实际phone maps查数量。若有未匹配selected-label tokens，需最小fresh诊断修复：保留全tier ownership/ambiguity，仅exact/phase token counts和matched总数筛matched，RF weights/端点/族统计保持。consumer15 finalpin/namespace12阶段暂待该审计，不把candidate当最终release。新的family07/worker12/controller15与recon05准备中，旧14/11执行来源不能覆盖。

### 交付与易失状态

原root reconstruction04 source20189b8daea2e6aa58ae29edc7ca79f03d45085f0406487d9332880a4131d157及packet仅prepared，保留未执行；新recon05需pin最终consumer/family，独立CSR/全tierownership/phaseweights/8320local×80clip及11776共同bootstrap。renderer03 rawa7093a1585c88cf86fbdda634d5e800315077797ae4bd04c82803be27de5b2bd仅prepared，预选g15/fixed-anchor全部144overview端点，全部9contrast/11776CSV与PDF保留。报告25尚未创建，旧fig08已全29source/output与实际审图PASS。

最终segment工具 `.codex/scratch/world_phone_shape_mean_dynamic_archive_attempt04/archive04_segment_bundle.py` rawbdf71cf0c3dd7392019b942081db7c26383e2e5b15533c747de862ec604c5868与portable restore1dc7bc6baa683d07ba80367a6a76f7ec1c2a8d15eb3e02d48538524d94cd3471 prepared。新explicit selection helperattempt05 raw3c757554c127dece9bcdf7a08919f4d668ab2ab0d871970f03b9f61447e08dab只允许ROOT明确全文件列表，校验SHA/NFC/casefold/Windows名与冻结范围；不自动发现/重复bulk/删除源。最终resultpack/control/recon/report/fig/BM确切成员待实际完成，模板仍PENDING。

PTY50274/bridge05 hidden217452/port2133易失但可自动SSH重连。服务器root@connect.weste.seetacloud.com:12349，ED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；所有密码/API key只在runtime环境，永不入文件/BM/argv/日志。TCP响应code与8MiB上限/7M chunks保持。

## Changed Files

HEADab40ffafdabaf113fc9b2961111d5eec61017ff3本地未推；runner12/wrapper12封存blob已PASS。新consumer14/15、family06/recon04/renderer03与操作脚本untracked。CONTEXT91行dirty，ignored HANDOFF61行仍consumer14预检pending；BM活动Task4269chars/current_step4实际写入并读回PASS（CLI前置ANSI导致写返回解析失败，ROOT读实际note确认，未重复写），现在需同步真实预检PASS/analysisFAIL/fresh候选。保留无关dirty/tests，不stage-all。CodeGraph未索引、explore曾拒绝，rg fallback。

## Verification

ROOT实际模型4340全文件二次读盘与历史7685完整恢复ledger均PASS；namespace11全预检与ROOT320桥接/13输入/10字段release PASS；actual analysis失败五完整日志ROOT读回PASS；consumer15派生量schema修复的全部1600矩阵/12800group零残差检查由remote代理实际完成，机制代理独立源码审核PASS。ROOT尚未直接复核该新parityreceipt；没有新mu/rho效果、完整bootstrap、最终图/报告。未运行pytest或新增unit tests。配置/Router/实验指令/Coding Session已读，40同chat/0legacy搜索及AN直接身份核验，Git canonical/noPR/pushedbaseline只读核实。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local unpushed HEAD `ab40ffafdabaf113fc9b2961111d5eec61017ff3`.

## Observations

- [result] 模型800V/1600评分完整本地恢复及ROOT4340成员二次读盘PASS，历史归档7685成员也完成ROOT独立实际恢复/完整ledger复核。
- [decision] 按实际producer存储schema比较原始曲线和metadata，再独立派生D0/search_gain；冻结模型/11776族/40规划分母不变，不重跑模型。
- [decision] 完整recipient tier必须保留用于ownership，匹配音素诊断计数单独筛matched；待实际160 maps量化后冻结fresh代码。
- [blocker] Consumer14分析因未存储派生字段误读失败；fresh修复全矩阵parity已PASS但尚待ROOTreceipt和token计数审计、新namespace12真实预检及完整联合分析。
- [next_step] 完成实际phone-map matched计数审计、审阅fresh consumer15与12800group零残差receipt，随后在fresh12授权并跑完11776独立联合分析。

## Relations

- continues [[Codex checkpoint - 2026-10-02T04-56-14Z - Qualified model restore and full analysis contract repair]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[音素平均谱形与内部轨迹的因果分解]]
- relates_to [[003-immutable-execution-repair-provenance]]