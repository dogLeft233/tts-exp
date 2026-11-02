---
title: Codex checkpoint - 2026-10-02T03-55-14Z - Completed factorial sources and execution
  plan shadowing repair
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-02-t03-55-14-z-completed-factorial-sources-and-execution-plan-shadowing-repair
capture: deliberate
started: '2026-10-02T03:55:14.7129186+00:00'
repo_root: E:/Documents/tts-exp
git_sha: ab40ffafdabaf113fc9b2961111d5eec61017ff3
model: gpt-6.1-sol
agent: codex
repository: dogLeft233/tts-exp
status: open
project: tts-exp
trigger: compact
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
cwd: E:/Documents/tts-exp
username: dogle
branch: codex/tts-mechanism-20260930
hostname: LAPTOP-E133GSLE
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-02T03-55-14Z - Completed factorial sources and execution plan shadowing repair

## Summary

冻结的800视频/1600评分已完成封存；第三次独立预检定位到执行计划被物理声学计划覆盖，正在创建最小修复的新入口完成11776端点联合分析。

## Story

用户要求从本地BM、代码、Git与网盘恢复TTS增强TFG研究，持续推进到基本探明原因，尤其寻找局部声学机制；授权服务器实验、历史数据、Qwen和必要重跑。ROOT维护理论、实验主线、BM与Git，gpt-6-luna/max子代理负责实验代码。最新“继续”保持此目标。已合格证据连接RMS/log-mel、真实编码器方向变化、WORLD谱形对应、鼻音附近感受野与独立嘴部运动。剩余mu/rho分解要直接区分音素平均log谱形、中心化内部轨迹及交互，完成全部冻结端点、恢复归档、图和最终报告25后再收尾。

## Working State

### 权威科学与模型入口

理论入口 `.codex/scratch/mechanism_synthesis_outline_20261001.md`，现有CONCLUDED BM实验与前checkpoint保留各cohort/端点/区间。WORLD局部全16640规划/14560可估计/2080短Tg20不完整，237负/0正，共同99%max-|t|4.947547631337023，主要n/ŋ附近约350ms感受野；FULL/VALID重复不能算独立复制。视觉09全1600轨迹，整体80片段/40说话人，T→N SP-RT开口+0.001501779，pointwise99%区间正；固定−85ms音素处理−RT区间含零，RT−FIXED降低音素分离。评分、几何、感知端点分开。

冻结mu/rho：q00自身、q10供体mu、q01供体rho、q11两者、SELF_GRID双次端点保持插值；自身F0/AP/UV/clock/长度/每帧SP功率/全局RMS固定。160上下文/80片段/40说话人/800V/1600评分；3456整体+8192局部音素+128局部整体=11776，共享100全A置换、20000×40联合99%max-|t|。整体RAW/RT、局部RAW，短Tg20保持规划40/INCOMPLETE。q10保持log-ratio轨迹，不保持所有归一log向量邻帧范数；SELF_GRID双核与供体单核不同。

模型14实际在2026-10-02T02:14:37Z complete/exit0：RUN `/dev/shm/tts-research-20261001/world_phone_shape_mean_dynamic_gpu_attempt14/run`，JOB `/root/autodl-tmp/tts-research-storage-20261001/job_control/world_phone_shape_mean_dynamic_gpu_attempt14`。ROOT最终5运行flags/14moduletypes/467925calls/0mismatch/shutdown与全1600行顺序已核验。核心本地 `.codex/scratch/world_phone_shape_mean_dynamic_gpu_attempt14/root_core_readback_attempt01/files/`。Rawplan b7c6da80a380a387ef6c1ef3b7d62d1181d419ddf58d83f214891ac9827b3c85；physicalmanifest55e34a1ffd13f61d27e7a9b9b630a2514cce98c40f4812340a658b581947bdd0；finalruntime3e5629a3612a630dbb9833e5f60b527b3dcc1d8d84ec32a0fa14774a8f790bef；all_files8076463991b66b1d7fbeddc322b3bb5e38fd072a00b121178663b63eb8ebc0ce，4327成员1376583771B。Protocol没有execution_plan_sha256，须通过code_seal/release.bindings桥接rawplan；metadata01实际失败、02准备未执行、03实际PASS。

### 实际第三预检失败与精确修复

consumer03原8d1d…f36失败缺args.parent_manifest_sha256；consumer09改用已检查parent binding后actual07失败缺_fingerprint字段参数。consumer10 raw86bee2f5d8ac3a5095234f9e9568745e19a08cc969554e861c8832f2f5f2749a仅补CAL manifest_sha256；family03 raw4996e916472d2d251c7e2c9ff31fcb5e875e8a460a0178b544d651a086536312。所有统计公式与模型封存不变。

原prepared08/analysis10有路径/OUT/CALraw错误，均未执行、原样保留。ROOT生成 `.codex/scratch/world_phone_shape_mean_dynamic_consumer_attempt08_v2/request_index_root_attempt01.json`（2959d51a8562f6efe7ba90cd5be0753677d3d19308ab22c622d4d3a979e2631d），仅修mkdir envelope到正确OUT08_v2；全9请求/6源实际SHA闭合。实际8个staging/环境操作PASS，worker08_v2 raw799d9e9cfe86e79f6ed1bad3ad9d05a503c3f1deba5995914455e09001db6694。ROOT13输入release actual125f6880f3fad59eb40c8ee62da276da670d9c58843e6fe2cad1e506393cde9f，上传并执行预检后失败：0/1600/1600；stderr rawc3cf306b33e2310e4f2f028200cacb1e70aa114c16ab84c9bf0c3de03095416e。OUT08_v2不存在、无任何分析授权。

五份实际输入release、预检receipt、环境receipt、stdout/stderr通过单exec完整base64读回，ROOT逐项SHA/size通过，保存在 `runs/recovered_20260930/independent_validation/world_phone_shape_mean_dynamic_consumer_attempt08_v2_preflight_readback/`。实际请求 `.codex/scratch/world_phone_shape_mean_dynamic_consumer08_v2_analysis12/03_readonly_preflight_receipts_and_logs_request.json.response.json`；bridge响应code字段而非exit_code。

ROOT及独立代理已定位：consumer10 preflight约864的plan是MODEL14执行计划；约938用checked_inputs[evaluation_plan]读入物理波形计划覆盖同名plan；1079取contexts及1315返回plan都错误。remote_recovery正在准备consumer11，将938–1068物理QA/repair区全部改为独立physical_plan变量，保持execution plan用于上下文/返回/下游。mechanism_audit审全部变量流与fresh diff。必须fresh verifier04/worker09/CODE-JOB-OUT09/controller12；旧实际08_v2文件不覆盖。不重跑800V、不改变冻结统计。原正确controller11 a690fe8ea599f4a2272d0c1803cd44d9dce9992cc07e91bb24e7e8f3d76f4193只准备未上传，逻辑可作为新namespace模板。worker检查CODE12协议副本与参数新CODE副本均绑定相同冻结字节，路径不同不构成重跑理由。

### 恢复归档与图的实际边界

pack05全4327原始文件/800V/1600行/320桥接远端独立SHA通过，实际COMPLETE/exit0。OUT `/dev/shm/tts-research-20261001/world_shape_gpu14_verified_bundle_attempt05`，archive1344285321B、193×≤7000000B、SHA994e95170f3013d138e15627fd6d3a79b281f026b3fc64e97188bafa32e603b6。manifest实际本地c796d6bca7e3d525843efebed89fd84a7522263e9f2ae391d6fec8af1432b9a3，receipt4f829985e70c276bbdedf88fa0bb1922b8d2c87a95efcffcbee156a402781963。ROOTchecker01错误uppercase complete已保留，fresh02实际完整193顺序/total/all4327同库存/runtime核验PASS，见 `postrun_package_attempt05/readback/root_pack05_metadata_readback_attempt01.json`。

模型包本地下载单writer hiddenPID252260/start2026-10-02T03:12:36.9054963Z，已实际118/193片，目标 `C:/Users/dogle/.codex/tts-exp-artifacts/20261002/world_shape_gpu14_bundle05/`，download_progress.jsonl；尚无archive/download_receipt。下载器b3eba108f6673defbf4141c09cf0b26eb638f3b8e2d14c90441a99d5a2a7516c；不要重复writer。restore checker01静态发现eager fstring KeyError未执行，fresh02只改.get及新输出names，raw8ba8c0b24b9ccd426072794515dd258c6e9175965968042315d995e38c697ea6；request60f4abe7ac8dfe17b37f6a37008222ca203b34f83716a5dad2ad56a4e9a8482e。ROOT与另一代理静态通过，待实际193片/总SHA/receipt后完整4340成员解包并ROOT落盘再哈希。

archive03 singlewriter hiddenPID160280/child201876仍活，18roots/7685members/6784files5027202296B，inventory6f70b8b63cb19b4d67041523c6943c1d60a9e8965737fbc0753e9bb3d3982aa5。最新代理实际4093640704B、STREAMING超过80%，目标 `C:/Users/dogle/.codex/tts-exp-artifacts/20261001/world_phone_shape_mean_dynamic_20261002_attempt03/`。尚无封存/实际恢复，保留source_retired=false。旧F15.7GB完整恢复PASS；旧02失败3994052608B partial保留。final segment04最新tool12b85c37edc554284dacd6daa7611ec3df787baa1853ddc98d64e8cfd3f54d09仅准备，scope03 raw6f79c1c494569c1a42330286e4be33eaaabd2fce4892bd683e18f5ecb1f6db6e：03与pack05仅完整恢复后reference，不重复bulk，新consumer/控制/报告/图/BM仍待冻结实际inventory。

合格旧图08全29input/outputSHA及实际审图PASS，manifest3e331c0eeac2583568e0cffc94f2f0da2469d0ead74386326efe07105849ef22。Fresh分解renderer02预选overview count误×9静态抓到、未执行；03只去多余len(CONTRASTS)，rawa7093a1585c88cf86fbdda634d5e800315077797ae4bd04c82803be27de5b2bd，bindingtemplatea9072b577d0538ecd23c317f016c8f46a877f086a70535069bf5b308d5eb8ae2。全部9contrast×方向2×A2×geometry2×estimand2=144端点；guard15/C_fixed_anchor预选可读图仍用11776族共同CI，全PDF/CSV保留短guard20。模板不可执行、未读新效果/未渲染。

私有PTY50274/bridge05 hidden217452/port2133是易失运行，ENV-only凭据；服务器root@connect.weste.seetacloud.com:12349，ED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。密码/API key不入文件/argv/日志/BM。单exec完整小日志读回避免SFTP延迟；传输8MiB上限用7Mchunks。

## Changed Files

HEADab40ffafdabaf113fc9b2961111d5eec61017ff3本地未推：仅runner12/wrapper12两个源已Gitblob/rawSHA核验。CONTEXT91行dirty已标模型complete/联合分析pending；ignored HANDOFF57行尚停consumer10待预检，需下一实际阶段更新；outline保存理论与primary鼻音研究来源。BM活动Task最新5197chars active/current_step4，在 `.codex/scratch/bm_main_task_consumer07_repair_readback.json`，第三失败未更新。报告25未创建；consumer10/family03/renderer03未跟踪，保留所有历史dirty/tests，不stage-all。CodeGraph存在但未索引，已实际explore拒绝，rg fallback。

## Verification

实际模型全量/最终runtime/core封存、包05远端全成员和193完整manifest核验PASS；consumer08_v2实际第三预检FAIL，五份完整真实日志/receipt读回与精确变量覆盖诊断PASS。根源是输入对象覆盖，不是科学阴性。fresh11尚未完成，11776实际联合分析未执行。restore02/renderer03静态修复独立复核PASS，实际恢复与新图未运行。未运行pytest/unit tests。Router/实验指令/Coding Session及配置已读，38同chat/0legacy完整检索、直接前AL身份确认；Git/仓库canonical/无PR/pushedbaseline已只读核实。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local unpushed HEAD `ab40ffafdabaf113fc9b2961111d5eec61017ff3`.

## Observations

- [result] 800视频/1600评分已全量完成，4327远端产物封存通过；第三CPU预检因物理计划覆盖执行计划而失败，无新分析结果。
- [decision] 保留全部执行失败，fresh11只修计划变量别名及新源绑定；模型、冻结族、置换、bootstrap和样本分母保持。
- [decision] 独立分析与完整恢复均取得实际结果后才写CONCLUDED、渲染新图和报告25；归档源保持source_retired=false，不重复大包下载。
- [blocker] freshCPU入口/11776全量独立分析、本地全成员恢复与报告25尚未完成。
- [next_step] 完成并审阅consumer11执行/物理计划分离修复，在fresh09通过真实全160/1600/11776预检后授权全量独立分析。

## Relations

- continues [[Codex checkpoint - 2026-10-02T03-09-13Z - Sealed factorial completion and independent consumer repair]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[音素平均谱形与内部轨迹的因果分解]]
- relates_to [[003-immutable-execution-repair-provenance]]