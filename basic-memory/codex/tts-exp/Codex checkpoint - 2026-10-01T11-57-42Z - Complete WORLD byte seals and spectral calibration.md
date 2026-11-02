---
title: Codex checkpoint - 2026-10-01T11-57-42Z - Complete WORLD byte seals and spectral
  calibration
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t11-57-42-z-complete-world-byte-seals-and-spectral-calibration
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 98895927bdc229d79953cb8bce54cee2b8824e83
started: 2026-10-01 11:57:42.366885+00:00
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

# Codex checkpoint - 2026-10-01T11-57-42Z - Complete WORLD byte seals and spectral calibration

## Summary

完整WORLD生成及10,887成员字节验封已完成；音素均值谱形与内部残差五格CAL生成也已退出0，独立效应复算和完整物理QA正在推进。

## Story

用户要求恢复并自主推进TTS增强TFG的原因研究，最新强调音量/能量已有实验，应探究局部声学原因。root维护理论与BM，代码实验由gpt-6-luna/max三代理完成。已有RMS真实encoder、mel分区、男女复现的完整独立结果支持表征方向轨迹作用；新主线固定受体时钟、全局RMS及其他WORLD成分，做双向音素局部声源/谱形移植，然后进一步分离平均log谱形与音素内轨迹，并用SELF_GRID检验映射平滑。

CAL诊断发现映射动态降低具有插值贡献，因此均值×残差四格加SELF_GRID是在读后续EVAL效应前冻结的针对性实验。不能把映射平滑直接当TFG因果结果，也不能用SyncNet分数替代视觉准确性。最终报告25等待完整WORLD、局部响应、独立视觉和这项五格后续及持久化结果，不在启动或排除项处停下。

## Working State

持久设计：`openspec/changes/factor-world-phone-shape-mean-dynamics-20261001/eval-design-root-freeze.md` SHAe17d257e43a78af6d7a291641ade6825b61c430ba51b8052383cab54dbea31fb，local commit e4e6eee。80clips/40speakers/160contexts，800新视频，固定原始RAW/RT两A，总1600完整scoreNPZ。3456全局+8320局部共11776 planned max-|t|，同20k×40speaker bootstrap，100 full-A共享置换；guard20短T保留计划40/80并INCOMPLETE，不救援39。

本轮进一步冻结六个统计端点的C定义：`metric-definitions-root-addendum.md` SHA9a6d9c14ad6d067911bd3e754d11e974787afe77a7b4bc05e278d92046ad33e6，当前local未推HEAD98895927bdc229d79953cb8bce54cee2b8824e83。family C=Bmedian−curve18固定anchor；Bmedian/Dmin/D0(curve15)/search(D0−Dmin)/offset其余五项。C_best=B−Dmin和D_anchor只做完整描述，不给11776同时区间。BM新谱形实验已实际append/readback PASS。

WORLD v6完整运行：`/dev/shm/tts-research-20261001/world_phone_transfer_gpu_attempt07_v6/run`，160contexts/12160scored rows/5760matrix artifacts/1600videos/0fail，runner退出0，19780 producer events与1043511 Module forwards，5flags每次断言、mismatch0、hash-chain shutdown。science05 SHA275a3169c7c7531a01d5942bd06278843faae70724f0e9ae5b7cf8c90ac7a6ac，wrapper5 SHA708a33d67a1e734f3247ffb3cccd6605a0a596621fac6adff818bf61cd623afa；external plan rawbba8bf428a1e5bc0a7e4eafa203976ea2c988d4332d2f82a02d2015e90f81d6d/payload1410aface9e20f23421b42ec871f00344c424d56ea4a0d87c921b670bd4b11bc。source05从未把execution_plan复制到RUN，勿改sealedrun。

核心raw seals：protocol8c29c8f757c29a88642f69301197560e2356ed641a959bbb0bd59fa0f109473ee；rowsd0f81f8d35fa9761d746c7420259d8605fc9191b1665ab171b0813fce844881f；resultse494f4dacaeb30520631f061a2b7f0e80ca05da716f7b609bcd7bdba64feaf9d；all_filesdfbbc35bea359d12ae7383671534fb6e1e08e7f62532de9faad49c37226ffe8c；code_seal1e3dc1d573383fd91226e475f2c880f2aa4125896647739ebd15e8466227d2498；run_status361b4514f00bab18edcd5e114517cd5b80e3ec76fcb2bd56f873e237e7cda4b7f；runtime sidecar7455c2594579dca28ddd099aea25b7b9006694e5fab40baa4a064bae3001581b1。全10887 files hash/coverage PASS已实际08gate。overall08 failed仅unique_artifact_coverage与protocol_flags metadata；NPZ/评分效应尚未读取。

操作gate需符合实际ID语义：root刚读08 metadata unique_visual_ids=10，而(sample,direction,visual_id) refs=1600；因此09拟要求global1600错误，尚未派发。原08已要求global10却artifact失败，matrix_id也可能复用label；remote正在做全部ID-only probe确认每context36、qualified5760。protocol确为3flags加enforced_at_each_instrumented_call=true，status只有3，sidecar五flags和versions在after_policy/after_runner。fresh gate10必须按冻结plan identity字段核完整集合，不能改分数/删样本。09source实际hash497704ab259a28f9258aedbad712c6fa949d85880cb4ccd87c536abeadf2c02f，agent曾抄错，需以文件hash为准。

scientific consumer05未运行：`world_phone_transfer_independent_consumer_20261001_attempt05.py` SHA8e40c3017ca7c53924302522678c18854c9b0702ce2223821b03d7fd05166a92，仅修consumer04 final run_status键execution_plan_sha256，科学定义不变。输出仍`/dev/shm/tts-research-20261001/independent_validation/world_consumer04_attempt02/output`。固定launch请求fresh attempt02已root读完/embedded Python修闭合quote，等待完整gate PASS。old local05 line225同错；mechanism代理受命fresh06唯一键修复，16640family不变；全160context metadata/父consumer PASS后才preflight/分析。

新CAL完整actual：source SHA4f4f1fe5d59565c87706cedf5531c3f29c649819d7a1c61413f0dee8508ed88e；spec01456c701f63863342820d5a3a87f98e7e3c93fa534aee79d977a3ca26f096d9；plan raw594592fe09a763634b3f757220096b0f08c038923aec418816442fc99b0879e7/payload85b94f1f7ccdb2a7c7dbceadf72d4e0f298019c64122514df9249a2d9442c65a。CAL04 JOB`/root/tts-research-system-20261001/job_control/world_phone_shape_mean_dynamic_cal_20261001_attempt04`，RUN`/dev/shm/tts-research-20261001/world_phone_shape_mean_dynamic_cal_20261001_attempt04/calibration_run01`，PID560917退出0/stderr空。preflight已root下载SHA3490be9fcb548d80cefd4c607fc78dfb2b7a5af1531df76706eef43f98defb28全20/10/40/200且EVAL读取false。新full报告16,389,362B/rawdc23a765794b0dd6e735be26955dbcb39e4ce06dff4fbe1be9f1c6b926ab7efa，gzip1,178,478B/raw88862372ff202f32e044abca2dc440a485794831acdcd4d84fe43a79003da10b；output inventory PASS283files/801875807B，manifest payloadf3b8045245edcf21ebc30c944f61318f8ac3538669152d64f68c882543d75a1f，producer inventoryabdf8e14d7957f595862f5628ba43c35924d569ca361e22b97eae98dce287e86。14/15download请求remote仍误指persistent/root路径，root未执行，local正在fresh correction。需完整报告40/200/0failure及单独全部公式重构QA，才EVAL物理800。

新GPU runner、5flags wrapper与11776 consumer由mechanism开发，未freeze/运行；必须等actual EVAL physical schema和完整独立QA。remote MediaPipe隔离CPU环境+1600旧WORLD全视频tracker正在准备，frozen aperture/velocity与−85ms exactphone定义保留，不能用新五格800替代旧10臂1600。

存储持久决策已BM实际readback：F:/tts-research-archives/20261001空约30.6GB用于新RAM全run、物理/评分/视觉/报告完整tar和restore，before/after source逐文件hash、tar成员path/type/mode/size/hash及wholearchive读回全部要求保留。旧v5 E+B慢archive02仍在运行，最后2.952GB/约4.107GB，不重启、不删除原run；未完成新F归档。旧v5 158/160失败不可补拼；旧retirement02必须完整双副本/restore/root复验后exact目标退役。已完成RMS1680files/1.807GB迁移与缓存2.693GB清理勿重复。

脆弱运行时：root PTY50274 ENV BRIDGE_TOKEN_PRIVATE/BRIDGE_PORT_PRIVATE=2133，bridge05 PID217452，getpass凭证只daemon内存、每次操作前inactive才重连、模糊mutation不重试。server root@connect.weste.seetacloud.com:12349，host ED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。密码/key永不文件/BM/argv/输出。`dispatch_requests_root_attempt02.ps1`调度当前ENV桥；新状态13/full与08/12ID probe刚实际PASS。

## Changed Files

01876da9提交四个合格WORLD/local源；9889592提交metric addendum。新CAL source/spec/plan、consumer05和新runner/视觉源仍untracked待qualification；原dirty测试与旧consumer/validator/rebind保留，未stage all。CONTEXT91行、ignored HANDOFF和主Task/WORLD笔记尚有stale阶段状态，下一完整结果时更新。BM storage decision、shape metric addendum写读已实际PASS。

## Verification

本轮实际读取v6完结metadata/core rawseals、08全成员10887逐hash PASS、forward counts/flag records；CAL04 full PID560917exit0、283file尺寸inventory PASS但full科学报告未下载。静态审核fresh09gate/launch发现ID域误判与CAL下载路径错，尚未执行依赖步骤，要求fresh版本保存原失败attempt。Git root/branch/status/HEAD/host和BM23samechat/0legacy/no-more/前W identity完整核实，Router/实验指令/Coding Session及bm-writing已加载。没有pytest。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- local unpushed HEAD `98895927bdc229d79953cb8bce54cee2b8824e83`。
- `.codex/scratch/world_gpu_attempt07_v6_consumer_attempt08/`：已实际失败metadata gate与全byte PASS；fresh候选保留。
- `.codex/scratch/world_phone_shape_mean_dynamic_cal_20261001_attempt04/`：actual full status/manifest hashes。
- `runs/recovered_20260930/independent_validation/rms_consumer_attempt07_pass_readback/independent_rms_factorization_report.json`及`mel_partition_consumer_20261001_attempt07/independent_mel_report.json`：已结论主机制，勿重跑。

## Observations

- [result] WORLD全160contexts生成完毕、10887完整成员字节验封PASS、1043511forward无flags违例。
- [result] 音素谱形均值×内部残差与SELF_GRID的CAL200格已生成退出0，待full报告/独立QA。
- [decision] 新11776family中的C固定为B−anchor18；best-min confidence做完整描述，定义先于新EVAL读取冻结。
- [decision] F盘用于RAM完整持久归档并逐成员restore，旧E+B归档持续运行。
- [blocker] gate需按context-qualified matrix/video IDs修正；新CAL full科学状态与重建QA、旧local06/视觉、新EVAL和完整归档尚未完成。
- [next_step] 读取全部WORLD ID-only覆盖探测，审核新完整gate并actual PASS后启动独立consumer05。

## Relations

- continues [[Codex checkpoint - 2026-10-01T11-16-47Z - Local spectral factor design and full WORLD verification]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[音素平均谱形与内部轨迹的因果分解]]
- relates_to [[WORLD 音素局部响应与声学剂量]]
- relates_to [[研究服务器缓存清理与可恢复归档]]