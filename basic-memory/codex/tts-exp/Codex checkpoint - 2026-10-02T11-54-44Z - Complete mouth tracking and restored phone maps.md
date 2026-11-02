---
title: Codex checkpoint - 2026-10-02T11-54-44Z - Complete mouth tracking and restored
  phone maps
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-02-t11-54-44-z-complete-mouth-tracking-and-restored-phone-maps
model: gpt-6.1-sol
cwd: E:/Documents/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
username: dogle
branch: codex/tts-mechanism-20260930
agent: codex
repository: dogLeft233/tts-exp
git_sha: c24d506a2dbf4fa24cd4bcbbaf241dc5a4f268cf
repo_root: E:/Documents/tts-exp
trigger: compact
hostname: LAPTOP-E133GSLE
started: '2026-10-02T11:54:44.4399749+00:00'
project: tts-exp
capture: deliberate
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-02T11-54-44Z - Complete mouth tracking and restored phone maps

## Summary

新800视频嘴部追踪已完成且0提取失败；外层日志解析exit1保持原样，160原始音素映射已实际恢复到本机，等待ROOT完整核验后运行本地统计重建与渲染。

## Story

用户要求从本地BM/Git/历史数据恢复并持续基本探明TTS增强TFG原因，尤其局部声学特征；授权服务器、云TTS、数据下载和重跑。ROOT负责机制推理、主线、BM与Git，三名gpt-6-luna/max代理承担实验代码。研究已建立增益→log-mel→编码器方向→嘴部运动链，新分解进一步固定自身F0/AP/UV/时间轴/逐帧SP功率/全局RMS，拆分音素平均log谱形mu与中心化内部轨迹rho，以SELF_GRID控制两次插值。该控制不等于全部局部PCM包络固定。

800V/1600score与11776共同20000×40说话人99%maxT统计族已完成并独立复核。现阶段以新800独立嘴部测量/54联合视觉端点解释局部谱形对生成动作的影响，同时完成本机全统计重建、8图表、中文报告25与可恢复归档，不能在准备/启动时结束。

## Working State

### 可信统计与完整恢复

权威consumer19/family11 namespace16、远程Recon09原数学保持不变。10304可估计/1472短Tg20不完整，40规划分母不缩为39；FULL/VALID2944精确重复不计为独立重复。联合临界值4.9553070962292765；endpoint raw c47cb2e0fb5b6eb014a54ae9b13fc86774cf8ce983753495e8a020e5a819de1c。Natural→TTS RAW/raw/g15/Cfixed mu−.149375761[−.279986046,−.018765475]、rho−.200588822[−.345675441,−.055502204]、joint−.283170331[−.520124362,−.046216299]。rho−grid−.191935408[−.336451484,−.047419332]，joint−grid两方向负向；rho−grid时间超额RAW/RT×raw/unit全部负向。直接mu−rho和交互56可估计Cfixed区间均含零，不按显著性数量排名、不推断无交互。

FULL局部35独特负向全在Natural→TTS：q10 a˥˩4、q01 n12、q11 n10/ŋ7/早tɕ2。q01 n whole G−.0175857846[−.0337551216,−.0014164476]、excess−.0142291936[−.0269053261,−.0015530610]，begin/middle亦负；全部g15phone区间含零。约350ms指五视频帧/16mel列输入块的查询窗口并集，含邻音和共发音，不是全网络精确waveform RF、孤立鼻音或特定共振/反共振证明。理论指向recipient谱形、发声时间骨架、固定A之间对应；TTS供体不是通用提升。

`runs/recovered_20260930/independent_validation/world_phone_shape_mean_dynamic_consumer_attempt16_full_output_restore_attempt01/` 已实际恢复712files/145072993B；clean consumer_output646=645产物+manifest。ROOT已全部SHA/集合/641NPZ5929arrays/0inf/231336设计support NaN核验，外部 `root_full_restored_disk_and_array_readback_attempt01.json`。PACK60401739B/9chunks SHA9dfae880381c8589ece8b7e36ccd81eb663f4d0daa8b0decb2a9cfdfa164db26。完整ledger `C:/Users/dogle/.codex/tts-exp-artifacts/20261002/world_phone_shape_mean_dynamic_consumer_attempt16_output_transfer_attempt02/full_restore_and_manifest_pass_ledger.json` raw0021163908165cba748d0a6fbde5bfaa7030798e935756fe628c8cec6b79c8db。Lossless tables02全部33868cells精确join已ROOT通过。

### 新800追踪已提取完成，外层失败待独立确认

08 CPU实际PID30275 exit0 PREFLIGHT_PASS；ROOT独立17stage/9controls/6modelseals/160×5时钟/source/runtime完整核验凭据 `world_shape_visual_preflight08_actual_readback_attempt01/root_readback_receipt.json` raw34bc294c17d3f258a3816709c5fb82a0175517201b08810237817edfd421e88a。ROOT tracking release raw09ca0731613c59fc9cbbd4082fd74133e8bbc84079bc7fe5a27de5db73df22e1/payloadc4c37a54ba0b6a5d6ac38e89a8abec1a903d9b5cc438a9f4884bfc6d8b41bf5c。Guard01候选三个schema/pin错误在执行前发现并保留；fresh guard02 db2381412ff6935b8ed10e6b2322c25997eb46dc6a9aef15c0cde90e85ec29d4 即时全hash gate后实际启动tracking PID40813/tick751203604。

实际新快照 `.codex/scratch/shape_visual_tracking_attempt08_guard_attempt02/04_root_readonly_extraction_status_snapshot02_request.json.response.json`：800 complete/0failed、800 features+800 availability、PID已退出、wrapper exit1、无extraction.result。extractor07原样每20视频输出progressJSON共40行及最终COMPLETE，08worker要求stdout仅1行，故外层解析失败。保持原800产物、worker/source/exit/log不改，不重提取。stdout2451B b538654cc886b17ab297e7dbda8ca4b683ff3b1ebadd118069b23575b372c865；stderr3612B 8c69e56cc14e1fccc8eb31098accfdcd71bfe2ff8f44ae761def92d93c61ce5b。新的metadata-only独立validator b00101a6dfcf79eea16cf0641c488321107fc8b02f6306e3be402266b27343dc/packet `.codex/scratch/shape_visual_tracking_readback_attempt01/request_index_attempt01.json` f417024d47e217ee8e7b7cd369d44bc5d9a6c02e097757198a57b8c4df7f2724刚准备、尚未ROOT审阅/执行；只读raw hash、timestamps/valid、seal/manifest/inventory/进度，禁止geometry效果读取。

OUT08 `/dev/shm/tts-research-20261002/independent_validation/world_shape_visual_attempt08/output`，CODE/JOB08详见packet。Extractor07 aa86f4011e4a93e345471142ce29d6fcb739d1d4d89254f95f7685d2bf7fb41f；plan07 raw13537f3836668de1539c3faf66870d5832ca340f2d1791a448a2bdaeb8ea996f/payload26d44618a58077717d4f66d1051ac89b8f311fc5ee9937fe4aec71d6bf906f6e。原visual freeze0394781ac97369b1dc8d3eaf79ec43f61b888d52cac8ff585eff4f0a14f896c8/source correction cfea922c2cba344b757005ab5f838ac6808db2c70808b3665da177b099e65de6不变。Availability-only freeze及ROOT精确geometry release尚未执行。Primary54 candidate/独立NumPy verifier02 da2c30b504172434f3f9d3464b103a745d675d6a326ac2fded7260e47955a427尚未最终冻结/运行，no new geometry read。54=9contrast×2direction×aperture/absolute-aperture-velocity/open-vowel-minus-bilabial；phone完整pair mask/ratio保存全部20k denominator，任何0denom失败不换seed/dropdraw。40规划/原MFA S0064=false/CAL−85ms/共同有效邻帧不变。

### 160原始map实际恢复与明确RF pin更正

更正前AU checkpoint内RF raw hash两字符笔误：实际且一直与远程、本地、原645manifest相符的正确值为 af90f774cfc79d7087ff3b048e14b529d73e2640e3acc40129419f199daccb95/843320B。错误cfd…cc95只影响候选pin，没有改RF数据。Collection02实际00 pinFAIL、01–04未执行，保留空namespace。Fresh03 corrected pin全数学不变。其05 status候选误用旧inspector pin，ROOT在执行前发现；fresh09 readonly corrected request实际通过，均保留。

Collection03实际PID47783/tick751256846/exit0：160 raw maps/80clips40sp、beforeafter相等、wire/disk/archive核验。Archive526059B 101f05bc6f7f9b17a4b078660fab9a30cd0bf40f1ea185025550d19e27261128；inventory144603B 51f01d460ff852a3bb34d8327af72d9cf729499b0c63568eeb52bcf6ef767ce5。位置 `runs/recovered_20260930/independent_validation/world_phone_shape_mean_dynamic_phone_maps_attempt03/`。Agent实际restorer03 6346bea829ae1f07e41a98eb6a11721058010e0e146be233d00ff86269a46ac5/Python dd5f8d19f6755d6491ee7c4bef2fe35ddd521334cc3ca3ed8fc93ebcadf135d0 PID283660 11:38:51.611–11:38:54.374Z exit0，mirror160/exactset/allSHA/selfhashPASS。Manifest raw2da33e8b67cd4483256ae7781bd3132f5b7161c866f6adc665ece7366c6ec9b3/payload890046373714c6c535d342cd06b54a381084f774c880dc79ca7b066c1e1eb8e1；execution receipt be5ee20a846819b3cc592c72bce7c747f178131f715d9ddc8d4e3e95279c92a6。ROOT尚需独立全160身份/顺序/sha/文件集合核验后授权finalizer04。

Recon11 source6a907bb226830bf14979b3284245986741267fb4baf887d664fbf4c0e2901205严格mirror，原09全部数学不变，仅正确RF pin和10parent lineage。Finalizer04 f3f735443c4a77eabba3b7978d270d71617a0ee4b945c9625a24936e848362cf已ROOTdiff和关键gate审阅但未执行，freshworkdir02/freshMPL02。Renderer03 a7093a1585c88cf86fbdda634d5e800315077797ae4bd04c82803be27de5b2bd；ROOTfigurebinding03 rawf95142b36ae82363f5d0841408818fc749184a1d5b960828a13e313f94305211；现有runtime全3042files已核验，不安装。目标8资产：11776/3456/8192/128 CSV、9页globalforestPDF、32页phoneheatmapPDF、144rowguard15PNG/SVG。Render verifier01 03afa1761173b028c54b3d6c682110e39e0c51614ae46399d7bff3298abeb560需receipt放output外。旧localfinalizer02/Linuxpaths失败workdir01保持。Draft05最后2cdfb6e4ff555f74e75fb72a0b8bc8afad4835f06e07caa9b37a82177d547b1c/12965B为中文草稿，54仍待。

### 本机与持久状态

BM Task实际6627chars active step4/phase statistics_restored_independent_visual_tracking_running，已替换历史膨胀内容为权威指针，尚未跟进最新800完成和实际镜像。ADR0044299chars open、Experiment10957chars implemented/statistical_family_verified_independent_visual_pending。HANDOFF46lines12820B e1f2f298ccd19d0a60219f64c16a62c712e28f2da0c1deba2848e55cbdcaf892待此次实况更新；CONTEXT91lines原dirty保留。

Branch codex/tts-mechanism-20260930/HEAD c24d506a2dbf4fa24cd4bcbbaf241dc5a4f268cf localunpushed/noPR，9exactSHA源/视觉freeze已scoped commit。scratch untracked、runs/BM/HANDOFF ignored，最终explicit小段封存及actual portable restore未做。私有PTY50274/bridge05PID217452/port2133运行凭据仅ENV。server root@connect.weste.seetacloud.com:12349、hostpin SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0在BM server-access；密码/API不写文件。所有请求response保存，truncated为两个bool对象；不重新dispatch已成功operation。

## Changed Files

新metadata-only readback checker/packet、map03/restorer/Recon11/finalizer04、ROOT trackingrelease/guard02、中文draft05仍未提交或本机文件。所有最终科学源/运行控制/实际图和报告/BM必须明确SHA选择封存；不能stage-all原dirtytests/config。

## Verification

继承已实际通过完整712恢复/645manifest/5929array/33868报告cells、远程665600重建和11776bootstrap。此次实际08CPU及ROOT全部元数据PASS、release/guard启动并完成800提取，保留outerexit1；collection03与本机restorer160实际PASS但ROOT160独立核验尚待。没有运行pytest或unit tests。此checkpoint先读取Router、实验指令、CodingSession；新同chat/legacy搜索、AU direct身份、Git/GitHubcanonical/pushedbaseline/noPR只读核验。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local unpushed HEAD `c24d506a2dbf4fa24cd4bcbbaf241dc5a4f268cf`.
- `.codex/scratch/mechanism_synthesis_verified_factorial_20261002.md`：已验证主线推理，最后报告需更新160镜像实况。

## Observations

- [result] 800新视频提取完成且0失败；160原始音素映射实际恢复，已保持未修改的原RF索引。
- [decision] 原wrapper失败/错误候选pin与来源全部保留；通过新metadata-only验证完成产物，不重跑800、不改旧exit/source或科学支持集。
- [blocker] 新54视觉族、local全再算/8资产、报告25/BM结案/scopedGit/finalsmallsegment实际portable restore仍需完成。
- [next_step] ROOT核验160镜像后授权finalizer04，同时审阅执行新800metadata-only completionvalidator，再冻结availability并完成54独立联合重建。

## Relations

- continues [[Codex checkpoint - 2026-10-02T11-04-45Z - Full statistics restored and fresh independent mouth tracking]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[音素平均谱形与内部轨迹的因果分解]]
- relates_to [[004-canonical-float32-score-field]]