---
title: WORLD 声源谱形与局部能量的双向移植
type: experiment
permalink: tts-exp/experiments/world-声源谱形与局部能量的双向移植
status: concluded
project: tts-exp
started: '2026-10-01'
tags:
- research
- world
- spectral-shape
- causal
---

# WORLD 声源谱形与局部能量的双向移植

## 问题与因果设计

在固定recipient音素时钟及评分RAW PCM下，测试TTS的WORLD声源坐标或谱形是否可以双向改变Wav2Lip响应。基于CAL-only的voiced比例/AP/谱平滑描述选择干预，root尚未读取fresh模型评估效果。协议与操作定义已经独立审阅；CAL waveform实现与内部QA完成，独立全数组QA及EVAL操作继续。

权威文件：openspec/changes/confirm-world-source-filter-20261001/spec.md（本地Git54599e0，未push）。复用20cal/80eval和固定primary face；WORLD own-RT为对照，量化RAW↔RT桥接。EX（F0+voicing+AP）、SP（归一谱形×own谱行总和）、BOTH、shared-voiced F0/AP子臂，加EX/SP局部能量匹配敏感性和E能量-only。主联合为EX−RT在N为正、T为负，各C/C_anchor的99%四项CI；40speaker×2固定意向分母。另报告within-WORLD四格，不能与RAW音频四格混同。

映射使用固定MFA/publisher text及same word position+phone LCS，phone normalizer v1只NFC/outer strip，保留音标属性；silence/unmatched为identity并记录实际dose。SP频率grid必须相同，简单bin sum。局部包络Hann801点/5ms网格、8次固定gain迭代（每次±3dB、累积local±12dB），globalRMS scalar另记；identity容差及残差/饱和由cal QA检查。保留FLOAT PCM和exact原长度，不做移位优化SNR。

## 当前状态
CAL独立复核完成：9060参数/波形检查和3810能量迭代检查均0错误，120最终F32波形与gain NPZ bitexact。报告SHA c3c2865a2d2fdade5c8aae280bbed4a0fcbcc785fada32caf3e42cd578aad8d1 / f05b23eda4e8d6ff213cffa1bf441979ec21c2ba8fe1009a232e094cfba4e590。

有效root freeze runs/recovered_20260930/independent_validation/world_cal_qc_freeze_20261001_v3.json，SHA0a5b694975efeb07b9d1e9c0beb10f0142bb4203ba4a026d7ff965477991ab9d。旧identity多一字符修为真实SHA，100记录和7027旧SHA字段不变；实际runner world_phone_transfer_20261001_overlay_attempt04.py SHA8143f5d51831693d03a4b9ce1f8f3a42134b5d903ed9111f749e61cc878b3e95，数学/合成/映射/能量AST与CAL相同。原160grids及physical-only2probe/2LAB均SHA核验。S0064“侵蚀”修复只用于物理操作，原视觉MFA79/80不改。

CPU attempt01因identity、attempt02因缺精确绑定normalizer退出；均在EVAL波形生成前且保留。补齐normalizer a5ae2fa9c5f57e25a090dc0432738fd1d6b29655fa1139ff45a0098d3b4a79b3与mapping05后，private world_phone_transfer_20261001_overlay/out/world_physical_eval_attempt03 已完整完成：2026-09-30T23:52:24Z至2026-10-01T00:12:03Z，exit0、stderr0、80/80 pairs、160 recipient contexts、9arms各160ok。1440主波形+320QC+160固定源=1920WAV/1920NPZ/160JSON，共7027943631B。物理资格80/80，原视觉79/80。

权威manifest world_phone_transfer_evaluation_physical_overlay.json raw SHA27cb2cdf2246920c0667dd5bcc5d744253b3c6b7d407270c1254ffaf986aa3b3；plan raw f92dbcfc71ecb775d98c3cd33ec19b4dd99d206c27d155eba8ae9bc51204861e。controls在scientific output外。以上是输入操作结果，尚没有WORLD模型效应。

执行别名已冻结到 openspec/changes/confirm-world-source-filter-20261001/evaluation-aliases.md，SHA d14260e47869003756a0efd6e84719011c0a2550b3af07ead5f62b902fb22f3f，本地commit2b78a03未push。每recipient10unique视频/10A缓存/36unique V-A矩阵/76explicit labels；全1600videos/5760matrices/12160labels。九臂包含RT identities，RAW和within-WORLD两完整四格族均保留。bridge q00=(V_FIXED,A_RAW), q10=(V_RT,A_RAW), q01=(V_FIXED,A_RT), q11=(V_RT,A_RT)，比较WORLD合成与fresh unsynthesized FIXED，不混称RAW→FIXED音量路径。

方向名是donor→recipient：tts_to_natural为N recipient，EX−RT主期望正；natural_to_tts为T recipient，期望负。主4个C/anchor联合门槛和40speaker意向分母保持。GPU producer正在实现，全部产物seal之后才独立汇总，不能停在CPU波形。

新输出根 /root/tts-research-system-20261001/0700；共享缓存与有界JPEG置于seal外，启动前预算空间。用户授权百度盘解决容量，保留活跃输入及恢复索引，只有已核验归档且消费者完成的副本可释放；纯下载cache可另按verified路径清理。
## Observations
- [status] implemented
- [result] CAL9060+3810独立检查零错误；EVAL物理输入80pairs/160recipients/9arms各160ok，1440主+480QC波形完整生成，exit0/stderr0。
- [decision] EX保持联合F0/voicing/AP；shared-voiced子臂及残差不称纯voicing。源时钟、全球RMS和RAW评分端保持。
- [decision] 两四格族RAW/within-WORLD及fresh FIXED合成bridge完整保留；1600视频/5760unique矩阵/12160aliases用共享缓存执行。
- [verification] raw manifest27cb2cdf…6aa3b3、planf92dbcfc…861e、v3freeze0a5b6949…ab9d、aliasannexd14260e4…2f3f。原视觉79/80不被物理80/80覆盖。
- [next_step] 冻结并实际运行九臂双向共同runtime GPU网格，独立重建全部评分后按四联合门槛与支持域报告。
## Relations

- relates_to [[TTS 来源方向与声学机制确认]]
- advances [[解释 TTS 的 TFG 与 SyncNet 优势]]
- uses [[TTS 机制研究服务器 weste 12349]]

## 独立物理波形复核完成

QA04实际PID330968/exit0，80pairs/160recipient contexts/1,920physical WAV/1,920gain NPZ/160RAW sources（全部2,080 WAV），failures0。报告`runs/recovered_20260930/remote/world_physical_eval_qa_attempt04/independent_physical_waveform_qa.json`669,127B/SHAe78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d，root下载、SHA与完整状态读取PASS。QA04 sourcefbd78734c80174074ca8477b969702acc7612e0dd9b22ade35f08b51a6834817；producer8143f5d51831693d03a4b9ce1f8f3a42134b5d903ed9111f749e61cc878b3e95不改。QA03 E失败是consumer错误使用recipient-envelope作target；04严格复现producer生成RT-envelope×固定source-phone ratio，没有改容差、筛样或算法。02/03旧失败保留。

冻结E迭代干预的local energy residual并非零：median-context p95abs1.52dB、pooled-context p95abs6.75dB、maxframe20.71dB。EX_EM .198dB（context-p95.343/max.522）与SP_EM .101dB（p95.172/max.302）更接近目标。E是有限迭代干预配方，后续解释不称精确局部包络匹配，不据高残差排样；在波形结果产生前已记录。SPpower重建maxerr1.07e−14，RT/EX/SP/F0/AP/BOTH与recipient clock均严格通过。原visual MFA79/80与physical overlay80/80分别保留。

WORLD05 neural preflight已经完整160contexts/1,600videos/5,760matrices/12,160aliases，GPU尚未实际生成。按aux实测持久产物预计1.67–1.82GB加暂存；cache28项清理只读preflight已实际PASS、apply未执行。真实RMS先跑，再单GPU WORLD，retain MKV/latent/context/scores，逐video清除未seal暂存JPEG/crops。没有删除科学NPZ，当前不需要先归档。

- [verification] 独立QA04全部160context物理波形通过；这不代替尚待执行的WORLD神经效应统计。
- [decision] E残差留作冻结配方解释，EX_EM/SP_EM用于更严格的包络控制，未因残差改变分析cohort。
## 守控RAM重跑与源码封存（尚无EVAL效果读取）

旧v5 ENOSPC保留158/160失败run及所有部分产物，缺final五flag/count证据，未读取效果。RMS09已迁移data并精确恢复旧v5前向日志；可恢复E+B归档02正在stream，fullmember/readback/E完整restore/源beforeafter相同后才受控释放精确旧树。archive client fc8e839e3bd3faa893d0a3309a614885cc658db1c5d34e5ba56d49a801f6b7cb。

Fresh WORLD v6保持science05 source275a3169c7c7531a01d5942bd06278843faae70724f0e9ae5b7cf8c90ac7a6ac、原attempt05 plan rawbba8bf428a1e5bc0a7e4eafa203976ea2c988d4332d2f82a02d2015e90f81d6d/payload1410aface9e20f23421b42ec871f00344c424d56ea4a0d87c921b670bd4b11bc、physicalQA04 e78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d完全不变。worker03 2d5d81d84210bb913bf68c74b849982e1e5b0a8e54f74c9303c220c4c1b05a1c实际mkdir/upload/stage/launch PASS，原runner绝对路径满足seals，RAM run /dev/shm/tts-research-20261001/world_phone_transfer_gpu_attempt07_v6/run，persistent data JOB world_phone_transfer_gpu_attempt07_v6_tmpfs_worker03。实际09:55Z启动PID9429/runner9466，最新读取106/160零失败、688954前向调用、fiveflags一致/mismatch0；这是运行状态，不是完整科学结论。

wrapper5 SHA708a33d67a1e734f3247ffb3cccd6605a0a596621fac6adff818bf61cd623afa在每模型forward之前核matmul/cudnnTF32 false、benchmark false、cudnn deterministic true、detalg true，compact count/hashchain原子sidecar，不重复巨型日志。resource gate tmpfs8GiB/cgroup余量16GiB/hostavailable12GiB，30s stop门2GiB/10GiB/12GiB以及probe失败USR1受控退出。控制文件persistent，最终全部科学产物仍须逐成员持久copy/hash/readback；RAM不能作为完成归档。

新local-response05源码766d7141037475c80bf70e28ef17ae88ecf86adff4e6bdfd5489b316ee1f6add及addendum04 f9fb38658c05cdd6dcae714f64ad61560f1bd125f81954b28c7773637199d8f6已封存，16640 planned family/16CAL labels/20k40sp draws/100fullRAW-A共享π；实际v5只读metadata inspector02 159NPZ/160slots/0readerror、fullsupport全部等于V且A更长，source05 strictclock gate符合当前producer，无scorearrays/effect读取。Local05上传CODE/spec/worker/inspector全SHA实际PASS，但尚未preflight/分析，等待完整v6+consumer04。五flags、full160/12160/5760/1600、allfilesseals及consumer04 source9feaf39eb6cad01c9dc76d86535fae4ce95379a8ed775a2e70e0bc93286eb056仍为执行门。

本地未推commit01876da9c33329bbdab3e3bf5a4ee2cf3ac20229封存四科学源，root逐字节worktree/stagedblob SHA相同、scoped whitespace PASS；没有pytest。新的平均谱形/内部轨迹五cell独立研究计划另见 [[音素平均谱形与内部轨迹的因果分解]]，不改变九臂现有统计或挑选EVAL结果。

- [decision] 旧失败run完整可恢复归档与fresh160run分别保持身份，固定输入/控制和完整消费者门。
- [verification] Local05实际stage七请求SHA PASS；v5 metadata全159 support=V=共同prefix、no effects。
- [next_step] v6完整runtime/seals后执行consumer04、local05和独立visual，形成数值与机理结论。
## 完整v6封存身份与独立复算

WORLD v6保持原science05/external execution_plan/physicalQA04，实际160contexts/12160scored rows/5760matrix artifacts/1600video artifacts/0failed，runner退出0。完整gate10再次核验10887成员路径覆盖与全部SHA，missing/unlisted/hash mismatch皆0；wrapper1,043,511 Module forwards五flag逐次通过、flag mismatch0/finalshutdown，Torch2.5.1+cu124/CUDA12.4。ID-only probe确认matrix_id为36类label/每context36/qualified5760，visual_id为10类label/每context10/qualified1600，计划row identity零不一致；先前operational ID scope判断错误保持失败记录，未更改任何科学数组或样本。

独立scientific consumer05实际exit0、audit PASS/error0、160contexts/12160rows/5760matrices/10887完整成员核验与runtime审计PASS。Worker finished12:12:12Z，overall1仅后续report packer失败，不能将打包错误写成科学验证失败。完整统计表正取回，root尚未读效应数值。source consumer05与local06只修final status.execution_plan_sha256键，科学定义保留；两者以及新CAL source/spec/plan已在本地未推commit5f2ed0dbe11f8aedfcfc7303b898c0405650a017。

以下身份由真实metadata响应程序解析，不手抄。前checkpointX对几项raw SHA的文字转录含重复/漏字符，不能拿checkpoint prose当源码或hash真相；实际文件JSON、gate10和此处读回是权威。
- all_files: `dfbbc35bea359d12ae7383671534fb6e1e08e7f62532de9faad49c37226ffe8c` (1440013 bytes)
- code_seal: `1e3dc1d573383fd91226e475f2c880f2aa412589664739ebd15e8466227d2498` (1850 bytes)
- execution_plan: `bba8bf428a1e5bc0a7e4eafa203976ea2c988d4332d2f82a02d2015e90f81d6d` (7555487 bytes)
- protocol: `8c29c8f757c29a88642f69301197560e2356ed641a959bb0bd59fa0f109473ee` (23290 bytes)
- results: `e494f4dacaeb30520631f061a2b7f0e80ca05da716f7b609bcd7bdba64feaf9d` (145971767 bytes)
- rows: `d0f81f8d35fa9761d746c7420259d8605fc9191b1665ab171b0813fce844881f` (5016080 bytes)
- run_status: `361b4514f0bab18edcd5e114517cd5b80e3ec76fcb2bd56f873e237e7cda4b7f` (3313 bytes)
- runtime_flags: `7455c2594579dca28ddd099aea25b7b9006694e5fab40baa4a064bae301581b1` (3551 bytes)

权威本地文件：.codex/scratch/world_v6_root_seal_metadata_attempt01_request.json.response.json；.codex/scratch/world_gpu_attempt07_v6_consumer_attempt08/launch_gate_attempt10.json (raw bab02d7ec9f24d9f82a12846d4fedc3df6b1ebc3a1c3c6618b9700e029ebe146)。run为/dev/shm/tts-research-20261001/world_phone_transfer_gpu_attempt07_v6/run，externalplan在/root/tts-research-system-20261001/world_phone_transfer_gpu_20261001_attempt05/preflight/execution_plan.json；producer从未复制plan进RUN，勿改sealedrun。
## 完整独立评分结论与局部后续

WORLD v6完整80clips/40speakers/160directions，独立consumer05退出0/audit PASS、所有10887成员验封、12160score rows/5760matrices复核。完整五成员结果包已实际取回，root逐成员大小/SHA、输出inventory/seal/source identity、audit→statistics binding全部验证；本地权威`runs/recovered_20260930/independent_validation/world_v6_consumer05_attempt02_root_readback/`。统计表raw SHA b11f5530820ac82d654d77240bbb05b93ef2b6ab554a8e2c05e1b34fb9313b88，audit fa06babb3a106bf1ed3acc337a6dc3bddc2c7df35dbd176f6c545c80cf599e30，root_readback_receipt保存完整五成员身份。共17024 bootstrap记录、2432 time-null记录；empty guard20在natural_to_tts一条短T上保留INCOMPLETE，不改40speaker分母。

在固定原RAW评分音、guard15、raw距离、C_fixed_anchor下，TTS→自然的生成G：SP −0.224938 [−0.318379,−0.129965]，SP_EM −0.225681 [−0.319151,−0.135094]，BOTH −0.258025 [−0.344713,−0.171715]；自然→TTS分别SP −0.283170 [−0.408206,−0.161779]，SP_EM −0.285832 [−0.399729,−0.173145]，BOTH −0.272037 [−0.385907,−0.171041]。区间为说话人两句先平均、40speaker等权、共享20k draws的pointwise99% percentile CI，并非这些次级端点的family同时区间。

谱形双向损失在unit距离、固定within-WORLD RT评分音、FULL/VALID/guard15及所有合格guard20、C_best与固定anchor中一致；SP、SP_EM、BOTH各64个计划G组合中56个合格CI均负，8个因短T guard20不完整。100 full-A共同时间置换扣除后的excess也各56个合格CI均负。局部能量匹配未移除SP损失，因此该结果不是单纯全局RMS或移植引起的局部包络差异。

固定RAW、guard15、raw、anchor的声源生成G：T→N EX +0.016973 [−0.028877,+0.063715]、F0 +0.016179 [−0.029797,+0.060569]、AP +0.003416 [−0.022781,+0.029853]；N→T EX +0.006078 [−0.049173,+0.058211]、F0 −0.018052 [−0.055386,+0.019958]、AP +0.008117 [−0.021796,+0.037089]。预设raw EX guard20主门两个T→N CI跨0，N→T guard20因短T不eligible。保留F0次级例外：T→N unit RAW guard20 C_best +0.007084 [+0.000334,+0.013808]、anchor +0.007386 [+0.000619,+0.014067]，且time excess也正；不能写成F0完全无效。

谱形四格同时改变生成和评分时有大正交互：guard15 raw/RAW T→N SP G−0.224938、E−0.204513、I+0.426003，total−0.003448 [−0.111251,+0.102672]；N→T G−0.283170、E−0.336981、I+0.482328，total−0.137824 [−0.281939,+0.021560]。within-WORLD SP的G两向−0.311137/−0.336537，I+0.512202/+0.535694。解释为谱形改变会影响生成流与评分流的匹配关系的证据；该interaction不单独等于decoder非线性，也不是自然/TTS两者原生总优势的中介比例。

FIXED→RT合成桥本身损伤须单列：guard15 raw/anchor G T→N−0.226340 [−0.285760,−0.166249]、N→T−0.175423 [−0.235469,−0.116716]。WORLD臂比较以各受体RT为生成基线，不能把这项合成损伤算入TTS声学原因。SyncNet结论尚不表示人类嘴型更准确；独立1600视频视觉分析已完成并合格，详见关联视觉复核笔记。

局部08完整327-file包与共享20k×40/max-|t|独立复算已PASS：16640计划/14560可估/2080短Tg20不完整，237联合负向/0正向，152 FULL/VALID phone-window检出（76比较）集中n/ŋ；phone guard15全部跨零，不能将局部窗视作孤立音素中介。独立视觉analysis09与1606成员元数据全复核也已完成，见[[WORLD 谱形干预的独立嘴部运动复核]]：T→N SP/SP_EM整体开口+0.001502/+0.001480的点态99%区间正，但全部干预的−85msphone-shape区间跨零，RT桥损伤单列。

事前冻结平均log谱形×内部残差+SELF_GRID五格继续执行。CAL200与EVAL800波形完整独立重合成、所有端点wavebridge/root全record读回已PASS；物理原freeze仍GPU=false，freshGPU08 actualstage完成，no-forward preflight在执行。80/40/160/800新V、1600fixed RAW/RT scoreNPZ、11776共同max-|t|和40说话人计划分母不变。未因双向SP损失挑选音素或改阈值；当前WORLD双向移植结论CONCLUDED，后续mean/trajectory科学状态仍IMPLEMENTED。