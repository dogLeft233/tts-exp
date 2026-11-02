---
title: Codex checkpoint - 2026-09-30T22-02-00Z - Qualified main analysis and mechanism
  followups
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-09-30-t22-02-00-z-qualified-main-analysis-and-mechanism-followups
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 47202515138ee3f5595cecb0918d5b1a52be1ce7
started: 2026-09-30 22:02:00+00:00
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

# Codex checkpoint - 2026-09-30T22-02-00Z - Qualified main analysis and mechanism followups

## Summary
主实验完整独立分析已成功产出；重采样与视觉结果已读取，随机剂量修正、方向/范数分离、WORLD和女性复核继续执行。

## Story
用户要求恢复并持续推进TTS增强TFG的因果机制研究，允许服务器、Qwen API、网盘和下载。Root负责主线与推理，三个gpt-6-luna/max代理负责实验实现、本地输入和独立核验。当前已完成新男性40×2 EVAL、10×2 CAL的Qwen生成与双脸2560行主实验；科学结论应分开生成端、评分端、视觉运动与事件时序。

上一checkpoint之后发现旧随机操作在float64达到剂量公差后cast float32，导致51/22428活跃帧、37 clips实际存储剂量略超冻结1e-6公差，最大1.022595779e-6。所有80 clips的orth±各7476帧独立KKT/nonnegative/inactive验证通过。保持公差不变，改独立消费者为每arm依赖：随机无效不剔除同clip合格baseline/orth±主门。07错误地拒绝1184行并只有43clip；root未读取其效应。08保留所有矩阵/结构校验和随机失败，现已完整完成，root只读status/keys，尚未读主效应。

Root在新norm、WORLD annex和随机修正spec冻结后首次读取sham与视觉EVAL。重采样四格80clips/40speakers的raw guard20 ΔC=-0.015428，99%[-0.042820,+0.010330]；anchor=-0.017175[-0.044277,+0.008807]。格式往返在本cohort不能解释大的正向增益；不宣称等效检验或中介比例。视觉640轨迹全部有效，但原N-phone定义只有52 eligible clips、16完整配对speakers；完整80/40主视觉支持不合格，16subset的orth+−baseline frozen-latency separation=-0.000117[-0.003126,+0.003903]，zero-latency=+0.000275[-0.002343,+0.002900]，AUC=-0.00104[-0.06632,+0.05972]。旧随机视觉比较须标out-of-spec sensitivity。CAL event-control失败，事件时序不能升级为确认端点。

## Working State
主artifact目录 /root/autodl-tmp/tts-research-20260930/independent_validation/source_axis_consumer_attempt08/output/：independent_consumer_summary.json 11990969B，SHA55988932b41c0410f10acdb87db43ac30d21315d9ef1a5e06f423462c5403f1c，status complete、artifact_failures0；execution_record.json SHAfcba6d567fbc5f9b5309283971068b6dc231baf55b23a7dcf76bf9f4e3fa6226，2026-09-30T21:53:57Z complete。CPU无模型forward；source consumer897c94a…ca1cb、validator17f68f…b20、writer85ec47…b6e524，完整6311成员preflight。下一步root读取支持、四主门、G/E/I、raw/unit/null。producer/source protocol/plan仍为上一checkpoint封存值。

随机诊断本地 runs/recovered_20260930/independent_validation/source_axis_dose_audit_20261001.json，SHAed16182dd06279adeeb9a88fd8d17c8ae933bad879991f332077a308e47dc16f。新aux spec openspec/changes/confirm-random-specificity-20261001/spec.md SHA679b11c8a67b18ac397d63b95900d55865b6dbf1ad7e8662bf3a5b2e915c8c79，本地提交5b2e716；80×2脸×freshorthplus/3correctedrandom=640，同原vectors/seed/dose，scalar求根每候选cast F32后检查实际存储剂量，保持1e-6、64步/上限64、下端tie。新操作单独hash、不改旧ops，女性首跑可绑定此修正；尚未GPU运行。

Norm spec631e760…01252与CAL geometry800d0a…e17e不变；所有旧02/03/04/04_gpu都是启动失败，未GPU forward。04 preflight实际写80context/160文件，不能当readonly目录复用。04_gpu因缺PSF退出1。现已仅复制原syncnet env纯Python PSF到隔离目录（tree fef27638b8c63926890f244daefcfe16a830275253648623da45e17fccf576a7），不引入numpy2环境。代理完成2CAL模型/S3FD/audio/video/SyncNet/JPEG/MFCC真实QA8.25s、forward false，report SHA7d1dd09ed010147e71b93be36ad1dcc969e0f5e17d37df5a1fbb44a3a20b8c28。新attempt05 module SHA93f7b600d83e205194a78e9c15f53dab9d2a18380a7df79b443fd3ee7ebb0a36，本地HEAD4720251已绑定执行源；预检目录与GPU空输出目录分开。尚未确认05实际首行。

WORLD physical-only overlay openspec/changes/confirm-world-source-filter-20261001/alignment-overlay.md SHA4822cf2dbaa1733c93447ef92446cc28cb5381f6c3c3d8d783cca5c477251b7e，本地072c260。S0064W0121将侵+蚀改官方词典复合词侵蚀，仅输入LAB分词，publisher归一文本/音频/词典/模型不变，0OOV N/T Grid合格。原visual79/80资格不改，其余158grids仍旧seal。probe_v2_review archive15bedb…a1，manifest2ce7ec…77f6，Ngrid05e5da…46bd/Tgrid0b1e08…997f8。CAL原20pairs/360waves内部QA通过，独立全数组/outside-copy/WAV QA仍由mechanism_audit处理，EVAL无音频。local_recovery新world_phone_transfer_20261001_overlay.py尚未tracked，应绑定actualSHA并封存input后生成80×2×9=1440。

Female40Qwen/40N输入已封存，沿用男性d/u/k3与dose，不refit。正式MFA05 PID675491，inputs/female_eval_20261001/mfa_20261001_cpu_attempt05/output/，80WAV/40pair预检合格。旧01–04均启动适配失败、无alignment。05用正确audio-root extracted/（relative audio/...）和原male TMPDIR；protocol440b29…8419、adapter2ae55e…6f6f12。pathmap portable_input_manifest_sha256是embedded349446…c2b8而非raw199d6a…ba9c5；分别绑定，不重写旧文件。女性GPU640尚未运行，等correctedrandom操作。

Sham local summary runs/recovered_20260930/independent_validation/resampling_sham_20261001_attempt02/independent_sham_summary.json SHA9548fca660d340623ebb648fac202df693f0e6971864d52bf65b8b3117003d83；consumer e519943…91775，80/320/0fail独立cache、全PCM长度与四格检查通过。Visual local eval_visual_20261001/eval_visual_tone_analysis_20261001_v2.json SHA7b6f0e7b7f32083605630e92a4141518eaacbe23057f0d8454f86505b72c7d59。其normalization_removes含旧normalizer的aspiration/length等加新tone，先核旧/新增源码再修正旧BM“仅tone其余保留”绝对表述；physical phone LCS仍严格exact NFC/strip、不移除IPA差异。

Core transfer manifest98350b…dd73、301片6311成员在服务器已全部验证，本地最近132/301，单SFTP downloader session52357，E盘约19.78GB空。Fragile bridge session13803/loopback8228为TCP newlineJSON，token只运行时内存，不进BM；单下载<=8MiB，主summary约12MB需split或远端压缩，不重开4stream。凭据从原用户授权消息运行时取，服务器root@connect.weste.seetacloud.com:12349已另存无密码reference。不要dump CommandLine。

## Changed Files
- 新physical overlay annex与random auxiliary spec已选择性提交072c260/5b2e716。
- 代理五个新runner和CAL工程脚本fb4bf81；norm attempt04/05修复269bf46/4720251，都是本地未推送。
- independent_consumer.py、independent_validator.py、independent_sealed_run.py dirty；untracked female MFA/selector/preflight、compound probe、source audit/MFA、WORLD producer/AP summary/overlay、independent sham consumer。
- tests中存在代理改动；root没有跑测试，并明确要求停止额外pytest，继续静态检查与真实产物QA。不提交scratch、runs或无关agent修改。
- BM已更新WORLD/任务/随机剂量/norm typed notes；norm误报RUNNING已纠正为启动失败。fresh sham/visual结论和主08结果还需写入typed experiment与最终报告。

## Verification
Root直接读取BM配置/Router/实验指令/schema、查询同agent/session的两种checkpoint类型及legacy（has_more=false），读回F确认exact身份。GitHub只读确认canonical repository、分支PR[]和已推送c4a1998链接；当前4720251本地未推送。Root读回sham数值与hash、visual支持集/数值/hash、random独立QA摘要；读取08顶层status/hash/keys但未主效应。PSF隔离及CAL2真实forward验证为子代理报告，后续下载report核验。没有把failed controls写成科学阴性。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 当前HEAD47202515138ee3f5595cecb0918d5b1a52be1ce7为本地未推送，无PR。

## Observations
- [result] 重采样往返raw guard20 ΔC=-0.015，99%[-0.043,+0.010]；正向上界较小，生成/评分器距离变化会互相抵消。
- [result] visual full80/40 primary缺完整phone support，16paired subset方向干预的separation/AUC区间覆盖零，事件端点CAL未合格。
- [decision] 保留原随机实际F32公差与全部失败；按arm依赖评估完整80主门，再新640同剂量对照，禁止放宽公差或选43subset主结论。
- [decision] 此后新假说应注明已读sham/visual EVAL，不能称原预注册；norm/random/WORLD overlay在本次效应读取前冻结。
- [blocker] Norm尚无EVAL forward；WORLD等独立CAL QA；female等correctedrandom与MFA完成；本地完整主归档未下载完。
- [next_step] 读取主08完整80/40资格与四主门、G/E/I、unit/null，把实证结果接上norm与WORLD的正向机制检验。

## Relations
- continues [[Codex checkpoint - 2026-09-30T21-01-48Z - Independent verification and expanded mechanism experiments]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[TTS 来源方向与声学机制确认]]
- relates_to [[TTS 机制确认分离生成端与评分端]]
- relates_to [[随机对照的实际 float32 剂量复核]]
- relates_to [[编码器方向与范数对生成响应的分离]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]
- relates_to [[女性说话人上的 TTS 来源方向独立复核]]