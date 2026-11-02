---
title: Codex checkpoint - 2026-10-01T11-16-47Z - Local spectral factor design and
  full WORLD verification
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t11-16-47-z-local-spectral-factor-design-and-full-world-verification
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 01876da9c33329bbdab3e3bf5a4ee2cf3ac20229
started: 2026-10-01 11:16:47.312357+00:00
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

# Codex checkpoint - 2026-10-01T11-16-47Z - Local spectral factor design and full WORLD verification

## Summary

局部声学路线已细化为音素平均log谱形与内部轨迹的独立移植；WORLD全量重跑最后确认106/160无失败，完整独立验证与新的五臂后续实验待执行。

## Story

用户要求从本地BM/Git/网盘恢复TTS增强TFG的机制研究，并自主取得全面明确结论。root负责理论、BM与主线，代码实验由gpt-6-luna/max代理承担。最新重点是局部声学特征，因为音量/能量已经做过。已有独立男/女性、RMS真实encoder四格与mel分区结果支持输入表征方向轨迹的主要作用；本轮进一步固定受体时钟、句级RMS、F0/AP等，通过WORLD谱形移植定位可操作的局部原因。

CAL描述发现映射后的音素内谱形较平滑，但双向插值本身也平滑，native whole-clip差仅−.009767861 natlog、13/20低，不能把映射诊断当成TFG因果结论。因此新实验分开音素均值与内部残差，再并列SELF_GRID双网格往返对照。方案在读EVAL效应前冻结，报告25仍待完整WORLD、局部响应、独立视觉及该针对性后续结果。

## Working State

持久源与协议：当前本地未推HEAD01876da9c33329bbdab3e3bf5a4ee2cf3ac20229提交了local05、WORLD science05、wrapper5、consumer04的逐字节核验源。前一e4e6eee96025bb26ac4273ed939790de4ed1a5e7冻结 `openspec/changes/factor-world-phone-shape-mean-dynamics-20261001/eval-design-root-freeze.md` SHAe17d257e43a78af6d7a291641ade6825b61c430ba51b8052383cab54dbea31fb。160 contexts/80 clips/40 speakers，五个全新视频格q00/q10/q01/q11/SELF_GRID，固定RAW与受体RT评分音频，总800视频/1600完整scoreNPZ；3456全局与8320局部端点共同11776 max-|t|，20k×40共享speaker bootstrap，100 full-A shared permutations，同固定完整支持。short guard20保持计划40/80且INCOMPLETE，不救援39。

新CAL实现已按root静态纠正WORLD manifest raw身份与全部五格域外recipient shape。最终source `scripts/experiments/world_phone_shape_mean_dynamic_cal_20261001.py` SHA4f4f1fe5d59565c87706cedf5531c3f29c649819d7a1c61413f0dee8508ed88e；spec01456c701f63863342820d5a3a87f98e7e3c93fa534aee79d977a3ca26f096d9；plan raw594592fe09a763634b3f757220096b0f08c038923aec418816442fc99b0879e7/payload85b94f1f7ccdb2a7c7dbceadf72d4e0f298019c64122514df9249a2d9442c65a。q00/q11全WORLD帧formula检查后才拷贝RT/SP原数组；mapped ambiguous帧必须原SP==recipient，否则该方向失败。机制代理独立静态PASS。尚无CAL数值执行，local代理正准备request bundle及prospective EVAL schema；mechanism代理因网络断线已followup恢复。

旧局部响应consumer05 SHA766d7141037475c80bf70e28ef17ae88ecf86adff4e6bdfd5489b316ee1f6add已实际上传，source/base spec/inventory/addendum04/worker/metadata inspector所有阶段0/PASS，bundle `.codex/scratch/world_local_response_attempt05/bundle_index.json`。其16640端点冻结不变。v5只读metadata探测159 contexts全支持==V且A较长，158有效q00/1failedstub/1missing，SHA17c1e9003aa62571d81c9f0e8aaea9c22a4846625706acc5a5cf4e5d05b4846b，未读距离效应。完整v6必须再次运行15/16全部160metadata gate，随后才06 preflight与07 analysis，且先要求consumer04 PASS。

易失运行：WORLD v6 `/dev/shm/tts-research-20261001/world_phone_transfer_gpu_attempt07_v6/run`，JOB `/root/autodl-tmp/tts-research-storage-20261001/job_control/world_phone_transfer_gpu_attempt07_v6_tmpfs_worker03`。worker9429/runner9466，最后真实读回106/160、0失败、688954 Module forward、五flags正确/hash mismatch0，tmpfs约30.628GB free。运行中status键plan_sha256，最终execution_plan_sha256。完整要求160/12160/5760/1600/0；all_files_sha256为flat relpath→SHA且不含自身。仍需最新状态与全成员独立封存，未读取WORLD EVAL效应。

WORLD source05 SHA275a3169c7c7531a01d5942bd06278843faae70724f0e9ae5b7cf8c90ac7a6ac、wrapper5 708a33d67a1e734f3247ffb3cccd6605a0a596621fac6adff818bf61cd623afa、plan rawbba8bf428a1e5bc0a7e4eafa203976ea2c988d4332d2f82a02d2015e90f81d6d/payload1410aface9e20f23421b42ec871f00344c424d56ea4a0d87c921b670bd4b11bc、physical QAe78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d均保持。consumer04 source9feaf39eb6cad01c9dc76d86535fae4ce95379a8ed775a2e70e0bc93286eb056。

consumer操作packet02在root静态审查发现五flags字典含版本而整dict比较、缺set-e、无人创建launch_gate，故未执行。fresh03 `.codex/scratch/world_gpu_attempt07_v6_consumer_attempt03/request_index.json` SHAe223996327170707cf44c632f81883cd77834919602911a4cd1ad1c5c504204b已准备未派发，gate7f62f9de.../worker5a9f988d...；五键投影另pin版本，set-euo，异步全成员metadata-only验封后写gate。保持绑定output `/dev/shm/tts-research-20261001/independent_validation/world_consumer04_attempt02/output`，新code/control03。root需审全source与真实run/external plan一致门禁再派发。

旧v5仍158/160 ENOSPC失败，不能拼成完整run。archive02 hidden launcher212932/child211404仍E+B双副本stream，最新1.744GB/约4.107GB，慢B造成吞吐低；不重启、不删除源。`.codex/scratch/world_v5_stream_archive_attempt02/INDEX.json`与archive_e进度是证据，10796 files/11439 members/4023374101 bytes，完整成员hash+双副本整文件读回+source before/after+E restore后才准精确旧run与JOB退役。retirement02root已审源/构建器、未执行；泛producer进程匹配可能保守阻挡，应待实际状态处理。之前RMS1680files/1.807GB迁移全hash/readback PASS，勿重复。

本地实际F free30656663552B、E15849181184B、C31259271168B。选择F:/tts-research-archives/20261001快速持久化新RAM完整run与reports/新physical/visual，全成员source before/after、tar成员hash及本地读回/restore probe；旧v5 E+B归档继续。新归档方案尚待具体packet和BM decision补充，不能把RAM当持久成果。

SSH桥03/port11639存活但transport inactive；原3019亦失效。root持有PTY50274 runtime token。未执行draft04自动重连字符串替换漏定义，保留为失败草稿；fresh05已由AST/static检查确认password仅daemon内存closure、每operation前若inactive才重连、绝不模糊重试mutation，host ED25519 pin SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。入口root@connect.weste.seetacloud.com:12349；credentials只getpass/运行时ENV、永不BM/文件/argv，启动与最新远端状态待实际确认。

视觉仍未执行：remote代理准备独立CPU MediaPipe0.10.21环境、1600全视频（10/context，含V_FIXED+全部9 arms），冻结aperture/velocity与−85ms exactphone映射，不能混为新五臂800。旧native/female视觉与分数分开解读。CAL报告635a7907.../payload a4ec6bb4...已root验，RMS真实方向主效应+.161456621[+.094297571,+.232449024]、方向减半径+.141622955[+.076967955,+.212289098]，16observed/null支持，三个existingPNG已root与agent视觉审查。

## Changed Files

四个已合格WORLD/local源已提交且staged blob==worktree SHA，冻结shape EVAL design已提交；CAL源/spec/plan未跟踪待真实CAL QA。原dirty independent_consumer/sealed_run/validator/rebind与tests及旧失败attempt保留，未stage all。CONTEXT91行与ignored HANDOFF尚待完整结果更新；BM新shape实验spec_only已实际写/readback、主Task与WORLD note也有实际读回，最后状态不是最终结论。

## Verification

真实v6最后106/160及forward侧车已读；local source与metadata上传实际PASS；v5shape-only159结果完整SHA读回；CAL source/spec/plan py_compile/embedded hash/LF/scoped whitespace和独立静态review PASS，无CAL执行。Git root/branch/SHA/status/username/host与BM22samechat/0legacy/no-more/前V身份直接核验，Startup Router/实验指令/Coding Session完整加载；canonical repository/current noPR/pushed baseline只读API核验。没有新增或运行pytest。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 本地未推HEAD `01876da9c33329bbdab3e3bf5a4ee2cf3ac20229`；冻结设计commit `e4e6eee96025bb26ac4273ed939790de4ed1a5e7`。
- `.codex/scratch/world_gpu_attempt07_v6_consumer_attempt03/request_index.json`、`.codex/scratch/world_local_response_attempt05/bundle_index.json`。
- `runs/recovered_20260930/independent_validation/world_cal_phone_spectral_detail_20261001_attempt03/`。

## Observations

- [result] 新平均谱形×内部轨迹及SELF_GRID协议已先验封存；修正CAL工程源已独立静态复核，尚无数值结果。
- [decision] 五臂后续共11776端点使用一套共享40speaker bootstrap/max-|t|，完整支持与计划分母固定。
- [decision] 利用实际F盘30GB空余持久化后续RAM成果，全hash读回后才释放源；不打断旧双副本归档。
- [blocker] WORLD完整seal/consumer04、local05数值与视觉、新shape CAL/EVAL及完整持久归档尚未完成。
- [next_step] 恢复凭证内存SSH桥，取得v6最新状态并审核执行consumer03全成员验封gate。

## Relations

- continues [[Codex checkpoint - 2026-10-01T09-57-39Z - Phone-local spectral controls and guarded WORLD rerun]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[音素平均谱形与内部轨迹的因果分解]]
- relates_to [[WORLD 音素局部响应与声学剂量]]
- relates_to [[研究服务器缓存清理与可恢复归档]]