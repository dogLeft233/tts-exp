---
title: Codex checkpoint - 2026-10-02T01-12-48Z - Spectral factorial runtime restoration
  and qualified figures
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-02-t01-12-48-z-spectral-factorial-runtime-restoration-and-qualified-figures
started: '2026-10-02T01:12:48.9070017+00:00'
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
hostname: LAPTOP-E133GSLE
cwd: E:/Documents/tts-exp
agent: codex
branch: codex/tts-mechanism-20260930
repository: dogLeft233/tts-exp
username: dogle
repo_root: E:/Documents/tts-exp
git_sha: 6b65dc1b025403aea49375bbbbec2a9b12f828a1
model: gpt-6.1-sol
project: tts-exp
capture: deliberate
status: open
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-02T01-12-48Z - Spectral factorial runtime restoration and qualified figures

## Summary

完整局部声学与独立视觉证据已通过复核，报告图08已封存并提交；剩余平均谱形/内部轨迹分解正在恢复原运行依赖，新归档将保留上次连接中断的原始部分文件。

## Story

用户最初要求从本地BM、代码、Git与百度盘恢复未同步的TTS增强TFG研究，随后授权远程重跑、Qwen及数据下载，并要求持续推进到基本探明效应，尤其寻找局部声学原因。Root负责理论、实验主线、BM和Git，代码/实验由三名gpt-6-luna/max代理承担。不要在排除项、预检或启动任务时结束；必须完成冻结的800视频/1600评分/11776联合族独立分析、桥接、可恢复归档与正式报告25。

现有证据把效应分成真实编码器方向变化、谱形与视频/评分对应关系，以及独立嘴部运动三个层次。增益在log-mel中移动真实编码器轨迹；WORLD谱形移植改变生成端和评分端的对应；局部联合检验主要集中于鼻音附近的感受野。平均谱形与内部轨迹的剩余分解负责进一步定位谱形的有效部分，科学冻结保持不变。

## Working State

### 已验证证据与持久入口

WORLD v6、Local08、CAL04/QA08、EVALproducer04/QA04和visual09均保持已验证状态。Local08共16640规划端点/14560可估计/2080短Tg20不完整，共享20000×40联合max-|t|临界值4.947547631337023，237负向/0正向；152 FULL/VALID局部检出等于76种比较，集中n/ŋ、少量早tɕ和晚a˥˩，所有局部g15区间含零。它们是约350ms感受野响应，不能写成孤立音素的中介比例。局部族的评价音频是A_RAW，整体端点才同时含RAW/RT。

EVAL完整160方向/800波形/1120输出/1440输入/5344token/0歧义，QA04全部独立重合成bit exact、320 q00/q11波形桥接通过。Manifest raw55e34a1ffd13f61d27e7a9b9b630a2514cce98c40f4812340a658b581947bdd0，QA rawd39fbc2a801c6ae4e615b1bf1644eb214bbc576fa08a77aadafb32567bb33238。完整本地证据在 `runs/recovered_20260930/independent_validation/world_phone_shape_mean_dynamic_eval_source04_attempt01/` 与 `world_phone_shape_mean_dynamic_eval_independent_qa04_root_attempt01/`，QA15整包ROOT复核通过。

视觉权威 `runs/recovered_20260930/remote/world_visual_analysis_attempt09/` raw5d0b8d18ec4565572292d5dbbdf43d25b92e72fbb3a0120eac0731536f31f98a。ROOT实际重跑418汇总/5016字段/15840定义及1606tar成员、1600sidecar复核全部通过；1600轨迹无失败。T→N SP-RT整体开口+0.001501779[0.000089710,0.002943500]，SP_EM+0.001479994[0.000182840,0.002813177]，80样本/40说话人。所有8处理−RT固定−85ms音素分离区间含零；RT−FIXED重合成降低两方向音素分离。音素端点52/53合格片段，但16/17完整双片段说话人，整体几何80/40。区间是pointwise99%，没有感知盲评；metadata复核未独立重提取几何。

图08 `runs/recovered_20260930/independent_validation/report25_figures_20261002_attempt08/` manifest3e331c0eeac2583568e0cffc94f2f0da2469d0ead74386326efe07105849ef22，五组PNG/SVG及全部29来源/输出哈希ROOT通过，实际图片审阅完成；未含未来分解数值。脚本 `scripts/experiments/plot_report25_world_visual_local_20261002.py` raw8e8a4b9f660fb76c2b0429406782b4dfa908a475a8699623485d8e41060e5608在当前单文件提交6b65dc1。图将WORLD/视觉pointwise99%与局部simultaneous99%分清，热图纳入所有规划音素/阶段/支持；完整SVG可放大。旧RMS三图不变。UI打开热图返回queued，未确认在侧栏显示。

BM世界/局部/独立视觉结论已CONCLUDED；形状分解仍IMPLEMENTED、有完整800波形而无新模型效应。活动Task最新实际读回 `.codex/scratch/bm_main_task_runtime13_readback.json`；理论在 `.codex/scratch/mechanism_synthesis_outline_20261001.md`。CONTEXT91行；HANDOFF忽略文件的归档02状态待更新。

### 剩余冻结分解与运行状态

q00自身均值/轨迹、q10供体均值、q01供体中心化log轨迹、q11供体两者、SELF_GRID双次端点保持插值；自身F0/AP/UV/clock/length/每帧SP功率/全局RMS固定。五V×两A/800V/1600评分，共3456整体+8192局部音素+128局部整体=11776，共享100全A置换与20000×40联合99%max-|t|；短Tg20保持40规划分母/INCOMPLETE。不得换定义、救成39或改科学冻结。

实际GPU11桥接失败经ROOT全160上下文/320阵列只读审计确定z512、V1024，原检查误要求V512。Fresh runner12只将V512检查改成V1024，SHA7f09f33dafbac7095c39d216f7e76947cad82e3101ed300681b105ef8bac3c15，wrapper12 SHA13a36b03bf4101db2c987798fe5bcbd23413880a0adb02efed3b896ef5985d96；科学不变。GPU12通过该桥接后缺python_speech_features。所有03–12失败/实际响应保留，10本地候选后改过且远程下载返回FileNotFoundError，不能用当前10代替实际执行源码366a78...。

GPU13复用不变CODE12，通过原v6封存路径恢复PSF，不安装升级包。PSF三文件树a419b997acfd21da8f4c7324e5be22e32357d085e37ac3376165e676c1e67c17；CPU/GPU两阶段独立来源/API审计且必须完全相等。Packet `.codex/scratch/world_phone_shape_mean_dynamic_gpu_attempt13/packet_attempt01/` index0189859be4c14bfa2a670ed3598e2eef1c4444bb7c86f6781ab467c3a90757fc，ROOT全11请求/6源码检查通过。实际00–04已成功，05源码16哈希与builder PASS、bindings47e5c5d71104d047702ad6761f908a01a254f73c5d85e7fb9670c6b5fbd33f93，但末尾Python状态断言失败。06实际状态与独立runtime02只读探针正在取回；需要读取完整CPU产物区分wrapper/assertion与真实预检结果。尚未授权任何新模型forward。

JOB13 `/root/autodl-tmp/tts-research-storage-20261001/job_control/world_phone_shape_mean_dynamic_gpu_attempt13`，RUN13 `/dev/shm/tts-research-20261001/world_phone_shape_mean_dynamic_gpu_attempt13/run`，CODE12 `/root/tts-research-system-20261001/code/world_phone_shape_mean_dynamic_attempt12`。worker13 raw87ef77ed51e6ac599ce4f88f11f155c5c81f9ee816f9926de0d509eab80c31a9，builder919735...cf2。完整160contexts/1600rows/全部assets/inputseals/no-forward验证后，ROOT另写exact `gpu_forward_release.json`，9bindings含实际execution_plan_sha256、runner/wrapper/物理QA与freeze。原physicalfreeze始终GPUfalse。模型状态须实际PID/starttick/exit sidecar及运行五flags，不能以启动返回当完成。

Independentconsumer08 raw8d1d52e7285ff1710f0191d864ab170a4cb6f25d89e607900ccbc1164e393f36未修改，编译/静态11776端点/有限掩码/共享置换/20000×40存储检查通过，尚无新模型数据实际运行。需完整封存后独立重算全部矩阵与family，再ROOT第二份完整bootstrap/CI复核。

### 可恢复归档与易失运行

旧F15.7GB归档 `F:/tts-research-archives/20261001/world_v6_attempt07_v6_attempt01/` 已逐成员实际恢复探针、完整23435成员/20991文件ROOT复核通过，sourcebeforeafter相等/source_retired=false。当前RAM源保留。

新archive02实际隐藏启动PID236232在2026-10-01T20:11:50Z成功，后因TCP10054/end-of-data失败；不能再称streaming。完整失败partial3994052608B rawbe8e2bcafc4d881e2cbdd876ee9fdcc4c6375f8ab1d6b12c1d6a781f1188781d，路径 `C:/Users/dogle/.codex/tts-exp-artifacts/20261001/world_phone_shape_mean_dynamic_20261002_attempt02/world_v6_attempt07_v6.tar.partial` 保留；无最终封存/恢复/源退休。第一次PowerShell+@空PID错误也保留。

Fresh03仅准备，未启动。`.codex/scratch/world_phone_shape_mean_dynamic_archive_attempt03/` client198265bd2b1b68cae1ae102bea9e01e48c5d1c115d31d7129fc60fb53565acc3、engine643504b57b01c42a7871486795048161a80883fa120b28a46b5a7e13b75c8426、readonlyrequest53a8370f895cff82abcee006e7f8e67a540c76b3e78c5fad30a551352cc98095、conditionalstreamrequestfaf390397179ca524c3a3dad98c06498c84a053baab453339991bdd72d02e9b2。18源根/33bindings/12gates与02相同，预期7685成员/6784文件/5027202296B、tree4ea15a...052、manifest23eb...857。30skeepalive/300sreadtimeout、仅一次全流重试，各partial独立保留，重新hostpin/inventory/diskgate；不追加旧partial。先ROOT完整读回新只读inventory，再hiddenENV启动/全流/实际恢复/前后库存相等。CODE12/GPU13/消费者/图08/最终报告需随后另一最终归档，不能修改已归档CODE08。

私有PTY50274和SSHbridge05仍活着，runtime-only凭据/端口2133，不在argv/file/BM/log输出。入口root@connect.weste.seetacloud.com:12349，hostpinED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。`dispatch_requests_root_attempt02.ps1`写实际.response.json，8MiB传输上限用gzip/chunks。隐藏归档需单独ArgumentList变量、Start-Process -WindowStyle Hidden -ErrorAction Stop、有效PID后再写STARTED，credentialENV随后清掉。PTY中的SecureString尚可用；进程不可当持久凭据。

## Changed Files

当前本地未推送HEAD6b65dc1单文件提交图脚本；前a4ddb提交CONTEXT与视觉四源，ccc3提交物理producer/QA/tracker与控制。Runner12/wrapper12/consumer08与操作包仍有未跟踪文件；原dirtylegacyconsumers/validators/tests保留，不stage-all。BM/runs/scratch/HANDOFF均本机易失，最终归档未完成。CodeGraph有目录但无可用索引，已实际拒绝explore，rg fallback，不自行索引。

## Verification

ROOT实际视觉418/5016/bootstrap及1606成员/1600sidecar复核PASS；图08全部29SHA与实际图片审阅PASS；物理EVAL/QA/CAL及QA15完整归档读回PASS；新GPU13实际源/metadata成功，末尾状态断言失败正在读取产物；archive02实际失败、03未启动。未运行pytest/unit tests，仅科学/合同/源码/编译检查。BM Router/实验指令/Coding Session加载完成，34同chat/0legacy搜索并从全部本地frontmatter有效started选择最新、直接BM读取身份确认。仓库URL/无PR/分支SHA/pushedbaseline已只读核实。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local unpushed HEAD `6b65dc1b025403aea49375bbbbec2a9b12f828a1`.
- `.codex/scratch/mechanism_synthesis_outline_20261001.md`：机制推理与已验证架构一手来源。

## Observations

- [result] 局部声学响应集中于鼻音附近感受野，独立视觉量化小幅整体开口变化与重合成的音素分离损伤；分数、运动与感知端点分开。
- [decision] 科学冻结与历史代码不可变；所有失败原样保留，操作修复只创建新入口，完整no-forward验证后另行精确授权。
- [decision] 保留02中断partial，新03重新整流而不盲目追加；保留远程源直到完整恢复与前后库存独立验证通过。
- [blocker] 新800V/1600评分与11776联合分析尚未实际完成，GPU13末尾状态断言待实际产物诊断；归档02失败，03只读预检及最终报告25待完成。
- [next_step] 读取GPU13完整预检与独立runtime探针，修复实际阻塞后精确放行并完成冻结800V/1600评分/11776独立分析，同时完成fresh03归档与恢复复核。

## Relations

- continues [[Codex checkpoint - 2026-10-01T20-11-41Z - Spectral bridge preflight and qualified local evidence]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[音素平均谱形与内部轨迹的因果分解]]
- relates_to [[WORLD 谱形干预的独立嘴部运动复核]]
- relates_to [[003-immutable-execution-repair-provenance]]