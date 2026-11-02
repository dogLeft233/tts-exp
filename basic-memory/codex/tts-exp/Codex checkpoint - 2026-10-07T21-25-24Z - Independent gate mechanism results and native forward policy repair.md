---
title: Codex checkpoint - 2026-10-07T21-25-24Z - Independent gate mechanism results
  and native forward policy repair
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t21-25-24-z-independent-gate-mechanism-results-and-native-forward-policy-repair
git_sha: 7606d6565cf6b608d255e94423045f8123b2f813
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
agent: codex
model: gpt-6.1-sol
trigger: compact
hostname: LAPTOP-E133GSLE
repo_root: E:/Documents/tts-exp
capture: deliberate
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
cwd: E:/Documents/tts-exp
project: tts-exp
started: '2026-10-07T21:25:24.3779085+00:00'
username: dogle
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T21-25-24Z - Independent gate mechanism results and native forward policy repair

## Summary
冻结增强头的四臂 Fit40 门控实验已完成并下载全部原始数组，根 agent 独立复算确定时序门控主要改善 D0；Wav2Lip 的实际前向确定性修复正在准入。

## Story
用户最初要求解释中文 TTS 提高 TFG Sync-C 的原因，随后要求交付适配多个 TFG 的自然音频增强头并持续迭代。增强 PCM 仅驱动视频，最终换回完整原始自然 PCM 和源时钟。根负责理论、协议、BM、私有服务器连接与独立复核；代码和实验由 GPT-6 Luna/max 子 agent 执行。当前冻结 32ms/8ms STFT、四频带静态项和正负谱通量门控、逐帧去公共偏移、3dB 投影及全局 RMS 的 step120 候选，先完成正式跨模型验证再决定新方法。

本轮 Fit-only 四臂诊断实际完成160/160，78.041s。40条/20说话人，原始门控、保留同一静态权重但门控全零、正负门控共同循环移动±64帧(512ms)。根完整审阅33请求，下载16元数据和12分块，重组四份原始NPZ共34,385,900B，核对整文件SHA；独立复算全部曲线、门控、几何、原PCM/时钟与原rescore13完全一致。实际门控相对RMS：Cbest+0.057816875、Canchor+0.148420191、far+0.141070795、D0−0.046865845。static Cbest+0.048502278，两种错位+0.057581210/+0.057511687。因此该冻结头的大部分Cbest增益在静态或门控错位下保留；不能把static消融当作重训后的最优静态模型，也不能把非线性投影的均值比例当作因果贡献率。

根另外用20说话人配对100,000次bootstrap(seed20261003)对三对比×Cbest/Canchor/far/D0形成12项描述性99%同时区间。正确门控相对+64和−64的D0分别−0.009903300[−0.017618920,−0.002187680]和−0.009763873[−0.017014146,−0.002513599]；Cbest仅+0.000235665/+0.000305188，其同时区间跨零。逐query平方距离的精确径向/角度对称分解误差<1.5e−13，正确门控相对两错位的角度项分别−0.142834663/−0.133641575、单位余弦+0.000604627/+0.000565106，次级边际99%区间方向一致。局部时序确有小幅方向对齐改善，Cbest主要是频谱及评分校准效应。这里只是Fit冻结神经模型/原生裁剪可微代理，不是官方编码视频或跨模型泛化结论。门控均值约0.502/0.480；全零消融也改变平均频谱形状和投影，错位是更干净的时序对照。候选权重与Dev协议未据此改变。

Wav2Lip strict10两条生成PASS但精确重复失败：AVI15/141帧不同，最终MP4131/141帧不同，所有帧时钟一致。实际读取11个face_detection源42975B确认SFD构造器及detect/batch_detect均设置cudnn.benchmark=True；构造器无warmup。因此新wrapper03在实际s3fd前向pre-hook恢复统一False策略并逐次记录，W2L前向断言策略，VideoWriter写入前逐帧hash，保留原DIVX/FFmpeg及原推理源码。根完整审阅wrapper02与03差异，修复未缩进的writer赋值、固定种子/帧数和models运行时origin遗漏；当前wrapper draft 37153B SHA14c214634c377aea12652f3fd79fcdb64dc72ad60a3afc28c3aaa04e9faaee9e，runner12 draft66065B SHAdd857d6978a9a6241bdc3abbe5563b5e454aff6f09c9c7f0fbda4858bd0f9037。最后差异及新runner尚待根审阅，未上传、预检或运行。CPU子进程预检必须测得CUDA未初始化、NoUserSite、所有实际源与包origin正确；正式两条须保留构造器/实际前向/写入前像素和原音评估完整证据。

## Working State
当前服务器root@connect.westd.seetacloud.com:18625，GPU GPU-d09622d0-3535-e662-bb45-3df2b049fcea。密码/API只在私有运行时，不写文件。持久namespace /root/tts-research-system-20261001/cross_tfg_audio_head_20261003；易失bulk /dev/shm/cross_tfg_audio_head_20261003。私有PTY57588/bridge07/port12249是机器局部连接，prompt可用且GPU空闲。单次bridge上限8MiB，超限文件分块后核整SHA。

MuseTalk/Ditto strict09的16份原始startup/shim/adapter/141×31 NPZ已下载，根独立QC03全部PASS，保留各自模型实际manifest lineage。Fit120/rescore13的39文件37,137,342B持久快照及全部目标整SHA重读PASS。Dev180 metadata03实际恢复完整20源，但无任何Dev模型运行：20clip/10说话人×3种子×3模型，正式新360 identity/RMS对照+180候选。验收各模型正增益、等模型额外+0.050 over RMS，以及D0/Canchor/content/visual/GUARD20和完整失败分母。Confirm80/40未见说话人与LatentSync继续封存。

实际SFD本地权重89843225B SHA619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543已核，禁止load_url回退。本次checkpoint前还实际执行models/__init__.py单文件只读身份请求，response SHA9bb7c0a1ff296b016405ca1c91702fd158a9b11b77a7b3cafd803adb27be98467，需接着读源并交给dev。Fit诊断41文件39,432,642B小型持久快照packet已准备未运行：`.codex/scratch/fit_only_gate_timing_diagnostic_durability_attempt01/`。新copier从已审copier01仅改三个schema/目标slot字面量，须审diff及15请求后实际派发，不能使用旧copier02错误index_copy路径版本。

Task92 active/Exp91 implemented实际读回PASS，但仍称原始数组待复核，已过时；下一次fresh prior后更新93/92并保存结论实验。binder02/template05/lineage01已审源但未stage/run；必须将相同wrapper和额外权重/API/包依赖接入实际Dev全链，动态绑定三种子与逐clip真实前端帧数，不能用floor(raw秒×25)假定帧数。先完成W2L资格，再逐模型lineage/binder/容量→360→180→join/rawmedia/readback。

## Changed Files
HEAD仍为本地未发布7606d6565cf6b608d255e94423045f8123b2f813，branch codex/cross-tfg-audio-head-20261003。CONTEXT93行/HANDOFF20行尚未更新本轮数组结论。五个既有portable_source_axis/overlay/tests修改未动。新实验源未跟踪、runs/scratch/HANDOFF忽略，均为机器局部；本轮未提交或push。根新建两份独立诊断receipt；子agent更新wrapper03/runner12与诊断备份packet，无通用pytest。

## Verification
权威指针：`.codex/scratch/root_gate_diag02_whole_readback_attempt01/`含四NPZ及root whole/numeric/angular receipts。numeric SHA5d169ba0bc35321bcc77e7f187a1f4b5c87af395ff93df4b5a9786cc410fcdd4；angular SHA5d7ab0843a6aceeeb15f85e22801faed8ac8562c6fa272c5663879b8b9e02f85。33请求索引在`.codex/scratch/fit_only_gate_timing_diagnostic_readback_attempt01/readback_index_attempt01.json`。前一不可变FJ把q14 response SHA多写一个d；以实际整SHA564c457796b9e80963c4a50a9d1e96ad2ba721dadc46fe073dd1061163cc58fe为准，FJ不修改。M/D根QC03在`.codex/scratch/root_strict09_native_raw_qc_attempt01/independent_readback_attempt03.json`。本轮实际源/权重只读、数组数值和门控验证均PASS；W2L新repair预检/重复、诊断持久snapshot和Dev540尚未执行。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
本次fresh HTTP200 HTML确认仓库、已推commit及当前分支无匹配PR。HEAD为本地未发布。

## Observations
- [result] Fit四臂160/160及全部原始数组独立复算PASS；正确时序门控对D0有小幅稳定改善，Cbest在错位中基本保留。
- [decision] 保持step120和Dev/Confirm封存协议；机制诊断不用于候选选择，必须实际跨模型原音换轨验收。
- [decision] 在SFD实际前向处修复其benchmark覆盖；逐次记录实际策略和编码前像素，保留原生成算法/编码器及旧失败。
- [blocker] Wav2Lip repaired repeat尚未实际准入，Dev全链须同一wrapper/源依赖和逐模型原始lineage，容量必须实际测量。
- [next_step] 读取已执行models initializer源并完成wrapper03/runner12最小差异审阅，然后执行CPU预检和两条Wav2Lip实际严格重复。

## Relations
- continues [[Codex checkpoint - 2026-10-07T20-58-33Z - Completed gate interventions and verified native policy gap]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[016-source-clock-roi-and-full-delivery-pcm]]
- relates_to [[015-same-gpu-deterministic-controls]]