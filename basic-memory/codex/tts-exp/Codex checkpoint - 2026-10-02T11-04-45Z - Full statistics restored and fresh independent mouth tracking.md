---
title: Codex checkpoint - 2026-10-02T11-04-45Z - Full statistics restored and fresh
  independent mouth tracking
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-02-t11-04-45-z-full-statistics-restored-and-fresh-independent-mouth-tracking
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
project: tts-exp
hostname: LAPTOP-E133GSLE
capture: deliberate
trigger: compact
repo_root: E:/Documents/tts-exp
started: '2026-10-02T11:04:45.9848455+00:00'
username: dogle
repository: dogLeft233/tts-exp
model: gpt-6.1-sol
agent: codex
status: open
cwd: E:/Documents/tts-exp
git_sha: c24d506a2dbf4fa24cd4bcbbaf241dc5a4f268cf
branch: codex/tts-mechanism-20260930
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-02T11-04-45Z - Full statistics restored and fresh independent mouth tracking

## Summary

11776联合统计族的完整712文件已实际恢复并独立核验，800新视频的独立嘴部测量正在完成CPU预检；本地再重建需要已定位的160原始音素映射镜像。

## Story

用户要求恢复并持续基本探明TTS增强TFG原因，尤其局部声学特征，已授权服务器、历史数据、重跑与云TTS。ROOT维护理论/BM/主线，gpt-6-luna/max代理执行科学代码和实验。当前证据已建立能量→log-mel→编码器方向→生成嘴部运动链；新分解固定自身F0/AP/UV/时间轴/逐帧SP功率/全局RMS，分离音素平均log谱形mu与中心化内部轨迹rho，另设SELF_GRID两次插值控制。控制不等于全部局部PCM包络固定。

800V/1600score已全部完成。9对比×两方向的global、phone、overall端点组成11776共同20000×40说话人99%maxT族；10304可估计、1472短Tg20不完整，保持规划40分母。FULL/VALID2944精确重复不可计为独立重复。独立原始RF重建665600比较、全部11776bootstrap已经在远程通过。下一步独立800嘴部追踪与54联合几何端点将区分评价器对应变化和实际运动变化，然后合并正式报告25。

## Working State

### 合格科学结果与完整本地恢复

权威Consumer19/family11 namespace16与Recon09仍保持原定义、原浮点公差、原完整支持集。`runs/recovered_20260930/independent_validation/world_phone_shape_mean_dynamic_consumer_attempt16_completion_readback/` 与 `world_phone_shape_mean_dynamic_root_reconstruction_attempt09_full_readback_attempt06/` 已完整源码/日志/状态/封印读回。Endpoint c47cb2e0fb5b6eb014a54ae9b13fc86774cf8ce983753495e8a020e5a819de1c，联合临界值4.9553070962292765。

Natural→TTS RAW/raw/g15/Cfixed：mu−.149375761[−.279986046,−.018765475]、rho−.200588822[−.345675441,−.055502204]、joint−.283170331[−.520124362,−.046216299]；rho−grid−.191935408[−.336451484,−.047419332]、joint−grid−.274516916[−.512036806,−.036997026]。TTS→Natural joint−grid−.197170395[−.381172437,−.013168352]。Natural→TTS rho−grid时间超额在RAW/RT×raw/unit均负向joint99%。mu−rho与交互的56可估计Cfixed区间均含零，不能以显著性数量排名或推断无交互。

FULL局部35独特负向全在Natural→TTS：q10 a˥˩4；q01 n12；q11 n10/ŋ7/早tɕ2。q01 n whole G−.0175857846[−.0337551216,−.0014164476]、excess−.0142291936[−.0269053261,−.0015530610]，begin/middle亦负；全部g15phone区间含零。约350ms mel RF包含邻音/共发音，不宣称孤立鼻音、特定共振机制或native中介比例。理论综合 `.codex/scratch/mechanism_synthesis_verified_factorial_20261002.md` 指向recipient谱形/发声时间骨架/固定A对应；TTS供体不是通用提升。

PACK07/output_transfer_attempt02实际60401739B/9chunks/712members SHA9dfae880381c8589ece8b7e36ccd81eb663f4d0daa8b0decb2a9cfdfa164db26。ROOT4metadata+9chunk wire/disk/combined全部PASS。实际全712恢复145072993B；clean consumer_output645产物+manifest=646文件，641NPZ/5929arrays/197420863B已allow_pickle=False打开，0inf、231336已设计support NaN，全部SHA/精确集合PASS。独立645manifest再验PASS。实际恢复PID277780 10:30:05.897–10:30:19.906Z exit0；manifest verifier PID274612 10:31:17.395–10:31:19.908Z exit0。完整ledger `C:/Users/dogle/.codex/tts-exp-artifacts/20261002/world_phone_shape_mean_dynamic_consumer_attempt16_output_transfer_attempt02/full_restore_and_manifest_pass_ledger.json` SHA0021163908165cba748d0a6fbde5bfaa7030798e935756fe628c8cec6b79c8db。恢复目录 `runs/recovered_20260930/independent_validation/world_phone_shape_mean_dynamic_consumer_attempt16_full_output_restore_attempt01/`。外部 `root_full_restored_disk_and_array_readback_attempt01.json` 保存ROOT独立全读回。

Lossless report tables02的36primary/2048FULLphone/35negative convenience subset全部33868投影cell已ROOT精确join原端点通过，没有重估。`world_phone_shape_mean_dynamic_report_tables_attempt02/root_exact_join_readback_attempt01.json` 是外部报告表检查凭据。

### 本地重建移植错误已定位，镜像准备中

Renderer03 source a7093a1585c88cf86fbdda634d5e800315077797ae4bd04c82803be27de5b2bd，现有Python3.12.14/NumPy2.3.5/Matplotlib3.11.2 runtime3042files已ROOT核验。全恢复后ROOT冻结figure binding03 raw f95142b36ae82363f5d0841408818fc749184a1d5b960828a13e313f94305211/payload7549adfe4f4e23f1a189441a16f9ca8d4ba7ff243081820a7b16ead93b224df1。固定原11776族生成8CSV/PDF/PNG/SVG资产，尚未实际渲染。

Localfinalizer02已实际执行并在render前FAIL：Recon09直接读取RFindex中的Linux absolute phone_map_path，Windows不存在该路径。不是统计失败，也不能将原远程PASS当作新本地实际再算PASS。失败workdir `.codex/scratch/world_phone_shape_mean_dynamic_local_finalize_attempt01/` 保留。RFindex raw af90f774cfd79d7087ff3b048e14b529d73e2640e3acc40129419f199daccc95 已ROOT实际核对。代理准备只读160raw maps collector/restorer，ROOT两源码全读过；collection packet `.codex/scratch/world_phone_shape_mean_dynamic_phone_maps_attempt01/request_index_attempt01.json` 尚未dispatch。collector685873baf7e32ed4939959dbe615164d461468b4d9b6761fc3b60424799f69b6；worker048ade707df3ef9799b448dd0f6f9ebb02cc5d451ce05efd566d431bf652063d。freshRecon10 4a7a3fdd2dd8db368a0a661304735e36683fa4242d9bae3a416f3e3c20457c42准备并待ROOT审阅：仅新增严格镜像解析器，保留09所有数学；镜像160原路径/身份/字节SHA封印，禁止改RFindex或dirty clean consumer_output。随后freshfinalizer03/新workdir02实际全部再算与8图资产核验。

### 新800视觉预检真实失败与08新启动

正式ROOT视觉freeze0394781ac97369b1dc8d3eaf79ec43f61b888d52cac8ff585eff4f0a14f896c8及source-pin correction cfea922c2cba344b757005ab5f838ac6808db2c70808b3665da177b099e65de6保持不可变。原始freeze在score完成后、ROOT解读前；执行helper拼写修正发生effect解读后/新geometry前。helper实际2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4。q00官方时钟/每ctx五cell25FPS/framecount/duration/CAL−85ms/原MFA资格不变；S0064 original=false不能救入。Availability不含geometry，先ROOTseal，再54=9contrast×2direction×3metric共同20000×40族；phone pair ratio存所有denominator，0denominator失败，不dropdraw/换seed/按geometry选资格。Primary54consumer及独立NumPy-only重建代理正在写，尚无冻结源码/实际效果。

Extractor07 aa86f4011e4a93e345471142ce29d6fcb739d1d4d89254f95f7685d2bf7fb41f和plan07 raw13537f3836668de1539c3faf66870d5832ca340f2d1791a448a2bdaeb8ea996f/payload26d44618a58077717d4f66d1051ac89b8f311fc5ee9937fe4aec71d6bf906f6e保持不变。Stage17实际PASS。07 CPU extractor实际PREFLIGHT_PASS全部160/800 clock/model seals/resource，但outerworker exit1：比较tracking_plan_sha256时错用raw argv5而不是payload argv6。ROOT全7原日志/17source/800clock读回确认wrapper错误；输出目录不存在/无geometry extraction。`world_shape_visual_preflight07_actual_readback_attempt01/root_readback_receipt.json` 保留完整真实失败，不写首次成功。

Fresh08仅命名空间07→08与worker比较argv5→6，源码/plan/rows输入不变。Packet `.codex/scratch/shape_visual_tracking_attempt08_packet/request_index_attempt08.json` raw8c6c66081d58c9a3fca79e044497dba3a0eb375abe92ea1e6ba18c2ab35c58c4；worker1be932016b5361c5ac34d377d15f398867deb31b85319f6d6458c7246ca1c9df。ROOT完整19requests/17sources/gzip/plan逐项核验PASS，实际00–17全部wire成功、STAGE_PASS17，CPU实际PID30275/tick751001169已经启动，未知终态。18只读status尚未dispatch。Extraction尚未授权/运行；ROOT需全预检读回后新exactrelease和guardedexec即时重核源/预检raw/plan/model6seals再启动800，不能仅依赖exit0或缺少modelseals复查的原extractor。Worker必须gate COMPLETE800/0fail。新track全部完成和availability ROOTseal之前不消费新geometry；旧1600tracks不能替代。

### BM/Git/易失本机状态

BM Task9397chars active step4、ADR0044299chars open、Experiment10957chars implemented/statistical_family_verified_independent_visual_pending，尚未结案。HANDOFF46lines11729B已记录全恢复但仍需把local02真实FAIL及visual07 wrapperFAIL/08CPU启动更新；CONTEXT91lines原dirty保留。Branch codex/tts-mechanism-20260930，HEAD c24d506a2dbf4fa24cd4bcbbaf241dc5a4f268cf localunpushed/noPR，9exact-SHA源/视觉freeze已scoped commit。scratch未跟踪；runs/BM/HANDOFF ignored，最终explicit小段封存和actual portable restore待做。

私有PTY50274/bridge05PID217452 port2133继续有效；runtimeENV仅该PTY，不能重新dispatch已成功00–17。Server root@connect.weste.seetacloud.com:12349及ED25519pin在server-access；密码/API key只走runtime不写文件。Bridge response request.json.response.json，download size/sha256；truncated是对象，必须核对stdout/stderr两bool，不能把对象当true。ROOT曾错误bool判定，随后真实18只读读回已保留，未造成stage mutation或科学失败。旧所有namespace/cache/失败记录保持。

## Changed Files

新未提交Recon10、电话map collector/restorer/packet、visual08控制包、ROOT full restore/arrays/tables/binding工具；Git已提交9源于c24d。所有新代码和实际run/BM/图都需最终exactSHA显式选择封存。报告25draft03/新8图/54geometry未完成。

## Verification

已实际通过712全文件恢复、645manifest独立验证、641NPZ全部打开/所有文件SHA集合、33868报告cell精确join；07真实CPU160/800预检通过但wrapper失败，08实际stage17通过并CPU启动，未冒称新geometry通过。Local02实际FAIL源于外部Linux maps缺失，未放宽分析。未运行pytest/unit tests。Router/实验指令/CodingSession重新完整加载，46samechat/0legacy及AT direct身份已确认，GitHubcanonical/noPR/pushedbaseline与Gitcurrent再次只读核实。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local unpushed HEAD `c24d506a2dbf4fa24cd4bcbbaf241dc5a4f268cf`.
- [Wang/Bao/Chen 2000 Chinese visemes](https://www.isca-archive.org/iscslp_2000/wang00e_iscslp.pdf): 单说话人195单音节/495记录/静态且未建模共发音，Table2 d/t/n/g/ng同类、b/p/m另一类，只用于解释声学RF响应不等于独特视觉类别；不用于新classification或音素原因归属。

## Observations

- [result] 全统计产物已可本地完整恢复；内部轨迹替换在固定功率/全局RMS下仍有联合族支持，n的局部RF响应集中于begin/middle/whole。
- [decision] 维持原11776族与新54视觉冻结设计；所有移植和wrapper修复保留失败、使用新namespace，不改源支持集或公差。
- [blocker] 新800追踪/54几何、160map镜像后local完整再算/渲染、报告25/BM结案和最终小段实际portable恢复仍需完成。
- [next_step] ROOT读回08 CPU完整终态并即时sealed release启动800追踪，平行完成160raw map镜像与local再算渲染。

## Relations

- continues [[Codex checkpoint - 2026-10-02T10-21-51Z - Verified factorial effects and complete transfer package]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[音素平均谱形与内部轨迹的因果分解]]
- relates_to [[004-canonical-float32-score-field]]