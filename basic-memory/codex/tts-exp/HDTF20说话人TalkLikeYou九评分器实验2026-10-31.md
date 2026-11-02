---
title: HDTF20说话人TalkLikeYou九评分器实验2026-10-31
type: task
permalink: tts-exp/codex/tts-exp/hdtf20-说话人-talk-like-you-九评分器实验-2026-10-31
status: paused
description: 按用户授权用qwen-audio-3.1-asr-flash转录40段HDTF，qwen-audio-3.1-tts-plus自克隆，192上TalkLikeYou推理和九评分器；以20说话人为统计单位。
assigned_to: codex
current_step: 3
started: '2026-10-31'
steps:
- 云ASR与同文本克隆及ASR回读QC
- 上传并冻结RMS24输入/同人图片/协议，运行TalkLikeYou
- 九评分器与真实视频SyncNet参考
- 来源级统计、独立核验、下载证据与报告
---

# HDTF20说话人TalkLikeYou九评分器实验2026-10-31

## Observations
- [status] paused
- [description] 用户明确授权云端ASR、Plus声音克隆以及新数据TalkLikeYou推理评分；API key仅保留运行时内存，不能写入本笔记或任何磁盘文件。
- [current_step] 3

## Steps
1. [x] 云ASR与同文本克隆及ASR回读QC（40段、160请求，自动QC非人工）。
2. [x] 上传冻结输入/同人图片/协议，80/80 TalkLikeYou生成及媒体验收通过。
3. [ ] 九评分器与真实视频SyncNet参考。
4. [ ] 说话人聚合统计、独立核验、下载与报告。

## Context
本机数据data/hdtf_diagnostic20_20261031：20具名说话人×2个10秒固定窗=40段，原片/图/音/视频已PASS。40自然ASR、40Plus自克隆、40TTS回读ASR均完成，模型精确为qwen-audio-3.1-asr-flash / qwen-audio-3.1-tts-plus，API key仅getpass进内存、运行产物全扫描未落盘，进程已结束并清空key。云目录runs/hdtf20_cloud31_20261031_attempt01，cloud_complete.json PASS、160API响应；WER最大.1923077，无>.20或时长[.5,2]x之外标记，非人工核听。40N+40T，每段1draw，self-reference该自然段；固定10秒可能截断词句，保留边界不声称完整句真值。

重要预评分修订：最初统一RMS−24构建失败，JoeKennedy seg1 N峰值将1.1725137（削波），DanKildee seg1 N峰值.9487286超过−1dB余量；在任何TFG/scorer前向前改全批统一−27dBFS、同−1dB峰值上限，无压缩、裁剪、来源替换或新云合成。旧protocol.json的−24原样保留、normalization_amendment_rms27.json记录修订，失败server_bundle保留。新bundle server_bundle_rms27构建中；远端run名已改talklikeyou_hdtf20_rms27_20261031_attempt01。

本轮代码scripts/experiments/talklikeyou_hdtf20_20261031：prepare、worker、generate、controller、batch_adapters、analyze、audit_precision_replay、verify_runtime、orchestrate、plot、package、build_bundle。共享评分体从旧中文包装复用，只改新绑定；关键：worker逐行face，生成alias按audio+face，cogenav --sources40防截断，统计先两段均值再20人。primary九评分器80媒体，另外40真实视频前245帧/同自然RMS27完整音轨作SyncNet匹配支持参考，避免拿真实视频错配TTS。控制沿旧6项，SyncNet/PEAVS/SyncReward本队列不重测时移控制。bootstrap10000/signflip100000 seed20261030/Holm9；QC清洁说话人补充，未按分数筛选。

192手册本次已实际通过浏览器读取：原wiki连接关闭、Feishu MCP UAT过期；备用公开正文https://l70p3ytk5k.feishu.cn/docx/G1qgdkcydowp2sx2BFhcVfQLnM2 标题1048服务器手册、03月26日修改；读取个人目录/共享conda/GPU索引/后台运行/注意事项。已登录whoami/个人区/home/data2/wangjiajun；2.3T共享盘余量；UUID3464 A6000空闲。仅本人目录/进程，GPU顺序，线程4/CPU0–3/显存60%/nohup。

SSH helper /tmp/tts192_remote.py，Python /tmp/tts192-client-env/bin/python；密码运行时从本机全局Infra凭据读、不回显。TFG /home/data2/wangjiajun/TalkLikeYouDeploy，B=/home/data2/wangjiajun/av-evaluation；新DATA B/datasets/hdtf20_cloud31_20261031。尚未上传/启动GPU。临时code tar /tmp/talklikeyou_hdtf20_code.tar.gz须重打包（RMS27代码修订后），包内有test_contracts.py远端3测试。当前本机5合同测试PASS（云2+波形/聚合3），compile通过。下一步等builder，打包代码和数据SFTP上传，解压个人区，测试后nohup运行orchestrate；它顺序prepare→runtime→smoke→全量→九评分器+GT→analysis→audit→plot→package。所有科学退出写stages，失败保留不盲重跑。

## Relations
- uses [[dataset-reference]]
- relates_to [[79-talklikeyou-chinese20-nine-scorers]]

## 验收续接 repair02（2026-10-31）

数据bundle RMS27 PASS：137228164 bytes，SHA2862876e328accc5bc4c1d56baf6c1850f4ca0322bbf2fac8f61f09d7037eccc；已上传B/datasets/hdtf20_cloud31_20261031，80输入RMS范围[−27.000008682,−26.999991118]，峰值最大.830078125；17权重和129源码fresh SHA通过，服务器3合同测试PASS。首orchestrator PID3831817成功完成runtime，首N模型前向PID3832173 PASS，245 feature frames、250pose、5.758GiB采样显存；但旧generate.validate固定512断言拒绝原生816×816合成画面。这是验收器假设错误，不是TFG前向失败。

未重跑首N模型。已修validate按源图尺寸验收，允许worker_result PASS且既有raw/ALAC齐全时只重做验证/receipt；失败脚本全部保存在RUN/failed_validation_source_attempt01，原失败log/stage/flags保留。新continue_validation_repair02.py顺序续接smoke→full→scores→audit→plot→package。原GPU3464在初次选择后被其他作业占用，候选GPU9a也占用；实际续接改空闲A6000 UUID GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37（非并行，首条0卡/剩余2卡），资源修订另存validation_resolution_repair02.json。未操作他人进程。已重打包并上传repair02代码；远端compile和3测试PASS。

新driver nohup启动命令 /tmp/hdtf20_start_repair02.sh；stdout B/logs/hdtf20_rms27_orchestrator_repair02.log，PID文件B/experiments/talklikeyou_hdtf20_20261031/orchestrator_repair02.pid。轻量探针 /tmp/hdtf20_probe.sh（helper file模式执行），已改看727卡和repair02日志；RUN/orchestrator_status.json为最新，旧orchestrator_failure.json/generation_failure.json仅历史，不据此重启。若再失败先检查具体日志/退出，不能重跑已成功科学单元。

## 当前运行与GT时钟核验

repair02 driver PID3838400运行中，80/80 TalkLikeYou生成及媒体验收已通过（epoch1793418299），已转入九评分器阶段，SyncNet初始2/120（80生成+40真实同支持参考）。具体实时进度以/tmp/hdtf20_probe.sh为准，GPU727卡。新原生816验收回归已加入测试：本机合同4+云2=6项PASS，服务器4项PASS。

额外GT诊断媒体预检发现首个旧GT容器标称245帧但实际244；PCM已逐样点一致。只重建40个GT参考为明确25fps CFR（trim first245, setpts=N/(25*TB), -r25 -fps_mode cfr），无任何TFG/N/T输入或输出变化，无评分发生。新本机runs/hdtf20_cloud31_20261031_attempt01/real_baseline_clock02，40条全部实际245帧、16k160000样点PCM逐值一致PASS。修复归档SHA1de7ff11707472d4e908dc23e959fbc9660fe6284d2482e19e99e7f6753f0ddd。远端DATA/real_matched_cfr02亦40条ffprobe245帧/0起点25fps PTS通过。

RUN/inputs_before_GTcfr02.json保留旧输入，inputs.json仅real_rows视频路径/SHA改为新GT；80个主TFG row序列的SHA在前后同为b0ba6d0cc769abef84ea39faaa7e460973d616067e54f2d21957abe1aad5d2e3。GT绑定receipt RUN/real_baseline_clock02_binding.json：new inputs SHA1c28fab51dfc023d33091e2d9053608ee3cd747c83a8ea2eeee8175d604fee04，scoring_not_started=true。老GT/原bundle不覆盖。修复实现脚本scripts/experiments/talklikeyou_hdtf20_20261031/repair_real_baseline_clock02.py；未来builder也已修CFR，这两个纯准备脚本均已上传远端用于最终source snapshot（不需重跑builder）。

本机cloud_quality_summary.json：WER中位0、最大.1923076923，32/40精确token匹配，0阈值标记，TTS时长比.736–1.16；非人工核听。该摘要已上传RUN纳入证据。最新探针/tmp/hdtf20_probe.sh显示driver存活、80生成进度、各评分行数；不再打印历史验收失败旗标。下一步继续监视生成到80及所有评分退出，不以已后台启动替代完成。

## 本机最终复核入口

结果预览目录results/talklikeyou/hdtf20_rms27_20261031_attempt01/previews/first_N.jpg与first_T.jpg已下载并看过，为同一人物起/中/末帧，未见空帧或静态输出，不是人工同步判定。本机已写独立CPU交付复核脚本 results/talklikeyou/hdtf20_rms27_20261031_attempt01/verify_delivery_local.py（未运行，等待最终archive/receipt），将逐成员SHA核验/安全解包到evidence，再独立按speaker/arm聚合两段、复算bootstrap/signflip/Holm9，校验80 worker PASS和首成功前向SHA未变。使用参数 --archive <local.tar.gz> --receipt <downloaded_delivery_receipt.json> --output <result_root>。

云转录已另导出runs/hdtf20_cloud31_20261031_attempt01/transcripts/*.txt、tts_transcripts/*.txt与transcripts_and_tts_qc.csv。全部云调用已结束，不需API key继续任务，严禁把用户消息中的key写入笔记/脚本/日志。

最终仍需：9评分器完成→analysis/audit/plot/package退出0→下载delivery_receipt和归档、本机脚本核验→另写terminal_delivery_receipt（归档打包时driver状态可能仍RUNNING；旧首验收失败旗标是历史，不能据此重启）→标准AAC80播放文件导出→正式编号报告、研究地图、索引/HANDOFF及本Task done。不要把后台启动或部分进度当完成。正式报告须说明RMS27与中文RMS24、40clip/20speaker/1draw与旧中文20source/3draw、每段同人图与固定3.png、变量原生分辨率和HDTF训练重叠/非论文test split等差异。

## SyncNet完整生成队列初步均值（其余评分待完成）

epoch约1793419530前后，SyncNet已完成80生成视频并进入40 GT（总84/120观测）。从80逐条原始JSON独立读得：N40 C均值6.380293440818787、D均值7.96166821718216；T40 C均值6.823768353462219、D均值7.8052179455757145；ΔC+.443474912643432。每人恰2片段，所以总体均值与20人等权均值相同；CI/主Holm9尚未计算，不能先称校正显著。已在commentary向用户明确为完整队列均值、等待其余评分器。

最新driver PID3838400仍存活，stage scoring/syncnet；所有80模型/媒体已PASS。不能因初步均值结束任务，仍需GT40及其余八评分器、分析审计打包下载。Task active/current_step3。

## 评分可测性续接 support03（必须按此继续）

SyncNet120/120完成：80生成N均C6.380293440818787/D7.96166821718216；T均C6.823768353462219/D7.8052179455757145；40真实GT均C7.143080532550812/D7.40228967666626；同音轨/帧数GT−生成N均ΔC+.762787091732025。仅完整均值，其他8评分器与Holm9尚未完成。SyncNet_complete flag已在RUN/stages，不可重跑。

repair02 driver在CoGenAV第13输入WRA_BobCorker_000_seg1_N触发原有MediaPipe检测覆盖<.95门槛而退出（epoch1793420124）。前12标量和24 NPZ（12原生+12真实移音）均完整保存。不是CoGenAV网络或TLY失败：该输入在嵌入前被拒。原脚本/日志/失败stage未覆盖，另存RUN/failed_scoring_source_repair02/。

新私有cogenav_support_adapter.py仅把原有95%覆盖/全无检测的拒绝记UNMEASURABLE并继续；不改阈值、cropper、模型/权重、前向、移音、归约或协议hash。原B/legacy_snapshot脚本保持原SHA6e8301c16aefa4a7f43e7333ea3218575ef395b8a8a3075a7b396a6234d0b11a，原模型数学连续源码span断言不变。初始BobN拒绝直接保留在cogenav/unmeasurable_faces03.json（原检测计数未保存，不能杜撰），不重新检测以求通过。其余新不可测记录实际帧数/检测数/比例。已算12条用缓存重建，不做重复网络前向；结束后support_adapter03_audit.json要求24旧NPZ与12旧score dict逐位相同。

AVS_u原来也有同一95%MediaPipe门槛，batch_adapters.py只将该前处理门槛转换为专用NativeFaceSupportError→UNMEASURABLE；无检测也明确记录。其他断言/模型错误仍FAILED，不宽泛吞异常。源码/权重不改，其他3类评分器不改。以每个指标实际完整支持集统计，任一片段缺失则该说话人在该指标整人剔为不可测，不用剩一段冒充两段、不插补/换样、不当阴性。CoGenAV/AVS_u可能n<20，报告须明确。

新continue_support03.py只续scoring→audit→plot→package；控制器跳过已完成SyncNet120，不再运行任何TLY生成。GPU仍727，4线程/affinity0–3/显存60%。私有CoGenAV通过flock原scorer.lock+原gpu_guard+原cogenav环境执行，未改共享run_scorer/legacy/score*.py。旧失败flags保留为历史。

新driver PID4052927，PID文件B/experiments/talklikeyou_hdtf20_20261031/orchestrator_support03.pid，stdout B/logs/hdtf20_rms27_orchestrator_support03.log，阶段scoring_support03。探针/tmp/hdtf20_probe.sh已更新读新PID、只展示本轮新failure，并统计CoGenAV有效行/不可测数及5适配器状态。当前CoGenAV重新加载并复用12缓存继续；最新error文件orchestrator_failure_support03.json（若有需查）。本机7合同测试PASS，服务器5合同测试PASS；新95/94边界和不吞其他模型断言的回归已覆盖。

package.py已补齐九评分器source/weight manifests、verified receipts和SyncNet上游Python源码，并在打包前重做源代码/五环境比较，要求与开始时SHA一致；不含新模型前向。runtime_inventory.json已记录ffmpeg/ffprobe二进制版本和SHA、硬件driver和实际CPU affinity；cloud_prepare_source.py已上传实验源码目录纳入snapshot，cloud_cpu_media_environment.json保存本机媒体工具环境。其余完成/下载/正式报告步骤仍必须执行，不能以此初步均值结束。


## support03已验收端点覆盖

CoGenAV与AVS_u都已完整遍历80输入，各79 PASS+1 UNMEASURABLE；同一缺失为WRA_BobCorker_000_seg1_N，按原95%人脸检测覆盖门拒绝，不放宽/换样。CoGenAV support_adapter03_audit.json已PASS：原12 score字典逐值不变、24 NPZ SHA未变、模型数学未改。本机checkpoints已下载回执与两个缺失清单。预计这两个端点统计为39完整clip/19完整speaker（仍须最终summary核对），其他端点继续。当前MTDVocaLiST运行中，未结束/未重跑任何已成TFG或SyncNet。

另已准备/tmp/hdtf20_terminal.sh，仅在最终driver COMPLETE且进程退出后执行，核对最新必需阶段exit0和九完成flag，保存单列terminal_delivery_receipt；不要提前运行。局部probe_compact /tmp/hdtf20_probe_compact.sh仅去掉大日志尾。标准AAC与配对页导出脚本results/talklikeyou/hdtf20_rms27_20261031_attempt01/export_and_index.py已写、compile通过，等待最终归档本机核验PASS才运行。独立核验脚本另加入WDA/WRA描述性分组检查（无子组假设检验/因果主张）。


## 四端点完成后的续接入口

MTDVocaLiST已80/80 PASS，完成flag为stages/mtd_complete.json；至此SyncNet120、CoGenAV79+1、AVS_u79+1、MTD80均完成。support03 driver PID4052927仍运行，当前VocaLiST21/80 PASS（driver elapsed01:00:10的观测），之后两分类器、SyncReward、PEAVS、analysis/audit/plot/package。VocaLiST当前约每分钟2条，不是卡死；探针GPU持续约80%利用，无错误，不应重复启动。

本机verify_delivery_local.py已扩展额外40GT的原始31列距离FP64核对及20人GT−generatedN独立CI/signflip，另存local_real_video_audit.json；九指标本机独立统计仍保留。两个本机交付脚本和/tmp/hdtf20_finish_report.py compile通过。后者是最终正式报告生成器（尚未执行，依赖最终evidence/summary.json、delivery_receipt和local_statistics_audit），生成/tmp/hdtf20_final_report.md，拟编号81需完成时再检查编号冲突。报告静态段已包括160云请求、RMS27预评分修订、训练重叠、不同原生分辨率、三类技术修复、19人可测支持、11/11统计谬误检查、非完整前向复现实质边界。正式写入须用BM并读回；先更新研究地图，再索引/HANDOFF/Task done。作者论文https://arxiv.org/html/2610.06658v1 本轮web实际读取，表1 8.73/7.97及第一阶段HDTF训练已核；只作不同协议背景，不能当直接差异检验。

所有云/80生成均已完成，API key不需再使用。仍不能以已后台启动、四评分完成或SyncNet初步均值结束任务；完整后继续终局凭据/SFTP归档/本机SHA统计复核/80视频导出+配对index/正式报告和记忆维护。


## 用户明确暂停评分（权威状态，2026-10-31）

用户在质疑数据泄露和TalkLikeYou Sync-C有效性后明确要求“先暂停评分”。已按当前已知driver PID4052927精确识别仅本实验进程树（4052927、4052928、4116090），停止控制器/编排器防后续阶段启动并终止本轮评分进程释放GPU。没有操作他人作业。RUN/user_pause_receipt.json记录epoch1793426060.2152216、原状态、停止PID/命令和完整行数；本机results/talklikeyou/hdtf20_rms27_20261031_attempt01/user_pause_receipt.json已下载。后查三个PID均不存在，UUID727 GPU回到2MiB/0%。orchestrator_status.json已标PAUSED_BY_USER；旧status.json RUNNING是保留历史，不应据此自动启动。

暂停时SyncNet120完成；CoGenAV79+1、AVS_u79+1、MTD80完成；VocaLiST保存76 PASS/80，最后在途条目可能只有部分文件，未来仅凭PASS receipt计完成。Synchformer/SparseSync/SyncReward/PEAVS及最终统计/审计/打包/正式报告均未完成。不得继续评分、补最后四条或启动其他GPU评分，直到用户明确恢复。Task status已paused/current_step3，不标done。原已授权云160请求、80TFG完成仍有效且产物保留。

本轮已下载syncnet/scores.json及real_scores.json至本机checkpoints，并用标准库statistics独立重算80生成：N40均6.380293440818787、T40均6.823768353462219、T−N+.4434749126434321，20人每人2段等权相同；GT40均7.143080532550812。已向用户答复均值/差值和HDTF属于TalkLikeYou训练来源但具体40段训练划分未核实的边界，不能据正差证明或排除泄露，未做九项Holm判定。

用户当前只读问题：刚部署PC-Talk是否与本项目使用同一官方标准评分。部署笔记docs/deployment/pc-talk.md与本机runs/pctalk_deploy_192_20261031均确认仅I2V部署验收，未跑SyncNet或九评分器。远端PC-Talk commit a0392744cbac857d96db1b99ceab70cf7b5f6c83的git ls-files和git grep已实际核查，没有SyncNet评价器/评估CLI（只有无关face detector的metric参数）。官方论文arxiv2503.14295v3 §4.1实际web读取：评价使用SyncNet LSE-C/LSE-D，同指标家族；25fps/16k/512，10秒跨身份测试，作者称train/test不重叠，但未提供本轮可逐项验证的评估脚本/权重hash/crop/offset协议，不能称与我们完全同协议。当前回复不得启动PC-Talk评分。