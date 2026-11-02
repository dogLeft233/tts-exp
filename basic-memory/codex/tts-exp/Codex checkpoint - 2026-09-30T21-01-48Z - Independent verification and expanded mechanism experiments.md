---
title: Codex checkpoint - 2026-09-30T21-01-48Z - Independent verification and expanded
  mechanism experiments
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-09-30-t21-01-48-z-independent-verification-and-expanded-mechanism-experiments
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: fcc4438e9eafa17c78dc49e205c418461bd0aa54
started: '2026-09-30T21:01:48Z'
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

# Codex checkpoint - 2026-09-30T21-01-48Z - Independent verification and expanded mechanism experiments

## Summary
男性双脸主实验和重采样对照已完成，女性40条Qwen输入已封存；独立消费者修复数值断言后重核，方向/范数分离与物理声源移植继续推进。

## Story
用户要求从本地代码、Git、BM与网盘恢复TTS增强TFG的研究路线，在提供的GPU服务器上持续探索正向机制并得到明确结论。允许Qwen API、重跑与下载。Root维护理论与证据解释，三个gpt-6-luna/max子代理分别负责本地输入/声学、GPU实现、独立验证与视觉测量。凭据只在运行时内存；服务器身份已持久记忆，不写密码或API key。

主线把生成器变化、评分器变化、视觉音素关联和事件时序分开。新男性CAL为10×2、EVAL为40×2，Qwen100条同文本自克隆完成。双脸主实验2,560行已生成；独立CPU先验证全归档、参数约束和缓存距离，再统计预注册四门、随机特异性、unit/null/anchor与支持集。Root仍未读取新EVAL科学效应。

CAL几何显示来源正向干预主要转动512维编码，范数约降0.1%。因此在读EVAL前冻结新的方向×范数四格实验，直接区分两种生成响应。物理实验在同音素、recipient原时钟上移植WORLD excitation、谱形与局部能量。CAL360波形内部QA合格，仍需独立数组复核与单条MFA分词修复overlay。Female40输入完成，沿用男性d/u/k，不重新拟合。

## Working State
Durable anchors: HEAD fcc4438e9eafa17c78dc49e205c418461bd0aa54为本地未推送提交，GitHub查询当前分支无PR。主实验真实绝对路径为 /root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/run/evaluation_20260930_retry3/；上一checkpoint缺少路径中的run/，本条明确更正，不修改旧checkpoint。主protocol SHA8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39；plan SHA19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab；producer SHAe50a31eaa2493b83e87cb4426b742e8f28e73836cbb9374b2319e7b60c940880。

Core archive: runs/recovered_20260930/remote/evaluation_20260930_retry3_analysis_20261001/analysis_transfer_manifest.json，SHA98350b737fcb40434da7fb653b9560869adaf4c9d85cc070f6c55939df47dd73，301片/6311成员；results JSONL SHAfd1f9ce1b23861de5eea7d87470f74e52451f1cc8bb050cbe85141a62975d3fa。服务器完整成员预检通过，本地下载仍62/301，继续用1–2传输通道。继承链3033/3033通过；正确sidecar为 transfer/evaluation_20260930_retry3_analysis_20261001_attempt02/inherited_artifact_verification.json。

独立CPU namespace /root/autodl-tmp/tts-research-20260930/independent_validation/source_axis_consumer_20261001_5f2941a_attempt06/，子代理报告PID672569、单CPU线程、CUDA空、完整preflight通过。此前attempt04因nullCI索引崩溃；attempt05正常产出INCOMPLETE_SUPPORT，但2560行均被random arm bitwise cast断言拒绝。独立float64求根满足同一KKT、不同终点造成float32最大差9.31e-10/1.49e-8/4.66e-10；attempt06改为既有strict-KKT一致的cast误差界并额外核u·delta实际舍入界和norm metadata。旧失败保留，不能当科学阴性。必须先读row failures和完整N-axis支持，再读效应。

重采样对照 run/resampling_sham_20261001_attempt01/：80 clips/320 rows/0 fail/exit0，20:20:24Z完成，四格包括q00均新forward false。producer SHA736b779c45b6ba0431f083bff1d73d0efc0b2b4d48c7182aa92ffc078e465aea。独立320行消费者由local_recovery编写，不复用硬编码2560消费者，不导入producer评分函数。CAL true/false工程比较8匹配行mel/z/PCM/A/V/距离/decoder/pixel均bitexact，但只覆盖2CAL；不追认旧EVAL状态。不可变cuDNN correction SHA616db2f886d7cca122a8e4aaa9fe1d6bbf6798d295f43923ba02a8b3f7f72ec9。

方向/范数spec openspec/changes/confirm-latent-angle-radius-20261001/spec.md，SHA631e76042a523f598e9135de7ff9bf328f3656eabc800a0aa85f38f66bc01252；CAL geometry sidecar runs/recovered_20260930/independent_validation/calibration_geometry_20261001/calibration_radial_geometry.json，SHA800d0a0d040df2337249958c907b28eb13ef75173392778ce7c674eed1fce17e。q00/z0和q11/z1 exact；q10=z1*r0/r1，q01=z0*r1/r0，F64后castF32。320行共同RAW-N A/支持域，freshfalse，k3/c18，20k PCG64等权40×2的99CI。四主门为q10−q00、q10−q01各C/Canchor下界>0。比例极值/范围外比例诊断不裁剪不筛选。runner latent_angle_radius.py SHAcdc2f922903cfbb35cc9e8d57e24a3963c590a8d8d72b99f93a9cf476406ceb0，静态review通过，GPU尚未起跑；soleGPU agent将做资产preflight后立即启动。

女性：官方15,582,913,665B包MD5 2f494334227864a8a8fec932999db9d8、SHAa4a0313cde0a933e0e01a451f77de0a23d6c942f4694af5bb7f40b9dc38143fe全量通过。20F S0124–S0143×2，40Qwen成功exit0。provider24k RIFF头全有声明数错误但EOF有效；canonical16k精确2:3且头正确，原provider保留。preflight_v5/qwen_female_full_eof_preflight.json SHA fcbf4da41fe0d81310f3c3635e9757e439b76de33a03cde0f10d75b49fa7ef58。remote inputs/female_eval_20261001/extracted/：N40/T40，portable input SHA199d6a2ebcdf54b6acd19f39e4604fd88cbb10c0e9d8ee4fcbaa173e5f9ba9c5。female_replication.py最新候选SHAed5dbcf9f81cfc98525ae2dc5f2ff684c3e431c3803486f10cc14294a3e527a5，静态review通过；独立40/20/primary640行，不改旧evaluate硬编码。需封存实际执行SHA。

WORLD CAL bundle runs/recovered_20260930/remote/world_phone_transfer_20261001_v2/calibration_review_20261001/，manifest raw297d3c138fc2de75e86f32db43f7d42f94476e3275c9ef9b72235c1880af07cc、plan raw2c5d26b09c0bd3fea4812c16612387ec5c96308082c9da1e2a823ee1f6c9d8df、QC rawa7a93f3bf591b85ff337f4fb295a058563697885548d3f2b0ffc4270c5fe6016。20pairs/40dirs/360waves内部0fail，repeat40bitexact、envelope identity40zero、RMS最大偏差4.01e-11。Fullmanifest embedded75b47…先前QC内941444…是加入calibration_waveform_qc之前payload，两值root独立重算匹配，不能混为rawSHA。Conditional AP joint-voiced+active差约0–0.006，与fulltrack −0.17明显不同，提示UV占比混合；仅CAL声学解释。EX_EM/SP_EM局部能量残差p95约0.15/0.09dB，E近似能量移植饱和/残差须报告。producer7a6661af…729b2，execution-bindings annex cde66fa11f140866fc87f07b7a39ca68ddefc403ff7839b239987549e9f7e47d。独立全数组/波形QA由reviewer处理，EVAL音频尚未生成。

Male MFA原160grids资格79/80：S0064W0121的蚀字OOV，官方冻结词典无单字蚀但有侵蚀。新2音频probe仅把侵+蚀合成侵蚀，不改音频/文本/词典/模型，两条件结构合格0OOV；原资格不改。待local包哈希复核后root冻结physical-only overlay，原source-axis视觉primary仍用原79/80，修复后视觉只能单列sensitivity。新WORLD EVAL producer可接受overlay，CAL既有执行源与产物不修改。

Visual主视频8片640文件全SHA通过。640/640 MediaPipe轨迹已提取约294MB，尚未读aperture对比；reviewer先按原N-phone与共同有效8轨迹冻结availability。CAL event metric −5frame控制失败，时序仅探索；phone separation与zero-latency/AUC分别按既冻结规则。短T S0099W0123 guard20空，guard15/full支持单列，不改完整80/40分母。

Fragile machine state: bridge exec session13803、loopback端口8228为临时内存服务，重连需用户原凭据运行时输入；TCP newline JSON不是HTTP，client.py单传输≤8MiB。不要把token/key写笔记/文件，不输出process CommandLine。远端host root@connect.weste.seetacloud.com:12349已在server-access持久记录。E盘约20GB空闲，保留已验证官方源包；大归档下载前检查E盘。未tracked脚本与runs仍本机/服务器状态，需要选择性提交并完成产物本地归档。

## Changed Files
- openspec/changes/confirm-latent-angle-radius-20261001/spec.md：root新增并提交fcc4438，EVAL解读前冻结。
- independent_consumer.py：dirty，null CI安全处理；independent_validator.py：dirty，random float32 cast界与level残差QA，实际执行源需封存。
- 未tracked resampling_sham.py、cudnn_flag_engineering.py、latent_angle_radius.py、female_replication.py、female selector/EOF preflight、WORLD/MFA/compound probe脚本。
- BM已新增编码器方向与范数实验，更新task、女性/WORLD当前状态及cuDNN决策；主实验笔记的core正在封存文字待最终结果清理。

## Verification
Root直接读取配置、Router/实验指令/schema、Git状态/HEAD、GitHub canonical repo/PR[]/pushed baseline，检索两种session类型与legacy字段并readback旧checkpoint确认同agent/session身份。Root核CAL几何、WORLD QA摘要/条件AP与fingerprint关系、female进程exit0；未运行新测试。子代理报告主preflight全301/6311通过、640视频hash与640轨迹完成、sham320exit0、female全EOF/身份通过、复合词probe双条件0OOV、norm/female静态review通过；后续独立科学摘要尚未验证。Attempt05严格相等错误已诊断，attempt06全量重核仍在运行。

## References
- [Canonical repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 当前HEAD fcc4438e9eafa17c78dc49e205c418461bd0aa54为本地未推送。
- 主source spec、WORLD spec/annex、female spec、visual tone spec及新norm spec均在openspec/changes/。

## Observations
- [result] 主2560行、sham320行、female40 Qwen输入、视觉640轨迹完成；这些是执行完成，不替代独立科学结论。
- [decision] 所有后续四格含新q00以统一实际forward false；原EVAL无per-forward状态不追认，2CAL bitexact敏感性只覆盖已测域。
- [decision] 在读EVAL前冻结方向×范数四格，直接判别angular/radial生成响应，禁止径向变化归因仅凭CAL几何。
- [decision] OOV修复只用官方词典已有复合词的输入overlay；原资格/原grids不改，physical单列overlay绑定与新源hash。
- [blocker] 主科学摘要须attempt06全行核验和完整支持；WORLD须独立CAL QA和overlay；新GPU norm/female仍待实际启动。
- [next_step] 先获取attempt06全2560行校验、N-axis支持与四主门/随机特异性/unit/null完整统计，并将结果与随后norm、sham、WORLD、视觉、female实证串成机制结论。

## Relations
- continues [[Codex checkpoint - 2026-09-30T19-43-21Z - Sealed source experiment and physical controls]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[TTS 来源方向与声学机制确认]]
- relates_to [[TTS 机制确认分离生成端与评分端]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]
- relates_to [[女性说话人上的 TTS 来源方向独立复核]]
- relates_to [[编码器方向与范数对生成响应的分离]]