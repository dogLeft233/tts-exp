---
title: Codex checkpoint - 2026-10-10T15-40-46Z - Fourteen updates and warmstart boundary
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-10-t15-40-46-z-fourteen-updates-and-warmstart-boundary
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-10T15:40:46Z'
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

# Codex checkpoint - 2026-10-10T15-40-46Z - Fourteen updates and warmstart boundary

## Summary

ROOT已独立核验唯一coordinator26/session09的14/20次UPDATED和原执行身份，第15轮正在运行；恢复入口同步至14，显式11参数备选的旧系数继承边界已保存，继续冻结的完整数学门与四中心选择。

## Story

用户最初要求用本地Basic Memory、代码、Git和历史数据恢复TTS增强TFG的研究路线，随后要求持续产出适配多个TFG、跨未见说话人且超过固定RMS的自然音频增强头。增强PCM只驱动视频，完整原自然PCM及原时钟用于评分与交付。ROOT维护理论、冻结协议、BM和独立产物QA；代码与实验继续交给原GPT-6 Luna/max代理。一次方法失败会进入明确的新方法迭代，不结束研究。

上一checkpoint保存11次更新与事件身份纠正。本轮ROOT实际核验第12、13、14次持久化tell的事件链、来源和进程，最后一次读回为15:13:41.623646Z、87events/14UPDATED；第15轮plus已正常启动。新证据已同步Task、Experiment、HANDOFF和CONTEXT，没有重启训练或重算部分训练。事件链与报告terminal计数资格尚不等于原始数组数学复算，全部20轮真实终局后统一进行。

本轮还核对了9参数direct-dB谱形向11参数tanh坐标初始化的源码和代数边界。可继承的是系数，径向投影、共享剂量及局部功率补偿改变了最终映射。采用旧Fit结果开发下一方法是可行的明示路线；下一方法须在效果运行前冻结自己的坐标、初始化与身份，Dev/Confirm继续封闭。当前没有选中θ、数值scale/epsilon或任何11参数训练、应用与评分。

## Working State

**持久入口。** Active Task=`tts-exp/codex/tts-exp/开发跨-tfg-的自然音频增强头`（active）；Experiment=`tts-exp/experiments/跨-tfg-自然音频增强头的训练与原音换轨验证`（implemented）。当前phase=`native_joint_spsa_pair15_running_after_fourteen_updates`。Task第一段已注明frontmatter context/Observations/HANDOFF为当前恢复入口，后面的长实施材料是历史依据。旧immutable checkpoint保留不改；本次通过同chat实际搜索、有效started排序和直接身份读回链接前一条。Git `codex/cross-tfg-audio-head-20261003`/`5688784dd5c6c1272074f0c4b6c5ff79f117c907`为local/unpushed，branch PR实际[]，没有commit/push。

**冻结路线。** initial720→20 joint SPSA→tell5/10/15/20四个full120中心→final720→portable→sealed Dev/Confirm，2160唯一科学上下文，失败物理尝试另计。Initial四候选NOT_ADMITTED，step060仅minviolation初值，不是方法胜出。package12、原自然PCM/clock、共同支持、cohort/policy01/ADR031及全部统计门不变。中心全部在20轮完整数学资格之后执行。9参数完整路线未过门就继续显式11参数及后续合理迭代。

**机器态：恢复先只读。** 唯一训练机`wjj@192.168.50.2:22`，公钥BatchMode/StrictHostKeyChecking=yes，V100 `GPU-d12b3309-c299-da83-064d-a3cd619bd560`。下列绝对路径是远端机器路径，SHM/活动PID均易失：

- JOB=`/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/job_control`。
- 原执行cwd/repo-root/source_root QUEUE=`/dev/shm/fit40_queue_remote_package_attempt02/project`；dispatch=`/dev/shm/fit40_joint_spsa_execution_clean_rerun_attempt03`。
- session09=`JOB/fit40_durable_spsa_session_attempt09`，sessionID=`fit40_spsa_bfa027430723_6800c380298d`。
- coordinator26 ID=`fit40_twenty_pair_a2438432284847ada11020287e82f2b4`，PID785358/start_ticks10166461；HOME源码`/home/wjj/tts-audio/tts-exp/scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_twenty_pair_coordinator_attempt26.py`，97564B/raw `93dddaf8e6e760f0c5e3c689657bff74a0296792ee1e5268f7fe66d0701f34cc`。start receipt raw `aead2e0a9e1bbd00f1554cfb694d605ad4e7143b9d647f9574d452c8d994f6c4`，self `0a41aadeb99fcf3e4e526e33381c7502e2e7b9dee31d5bcc16030f0ee1e8f01a`。
- lexical Python=`/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01/bin/python`，合法symlink，勿resolve至MFA。HOME只用于控制/存储，执行与重放坚持原QUEUE。

旧AutoDL westd18625 DirectD0、旧session07未知进程均不重新launch，不清正常writer.lock。`/root/dev_method_resume`为唯一活动监控代理，沿用原Luna/max；Signal原代理完成边界复核后parked，ASR原代理已接安全归档方案的source-only准备，未执行归档。自动化tfg已存在且ACTIVE15min，不创建重复自动化。密码/API key不写文件、argv、BM或日志。

**本轮实际事件链证据。** 各目录包含完整remote readback与ROOT自校验receipt，均为machine-local `.codex/scratch/`，尚未成为Git持久产物：

- `.codex/scratch/root_coordinator26_twelfth_tell_eventchain_actual_qc_attempt01/root_actual_qc.json`：14:12:55.805734Z，75events/12UPDATED，self `094285b50b82dcd425992cfb6a33da6cf15ec32b2f5d0fc6b827e12379bf4216`；event72 raw `48431ec3ee05a78317a573cb5e156ed03548950f0179242c0773dc9879b26408`。
- `.codex/scratch/root_coordinator26_thirteenth_tell_eventchain_actual_qc_attempt01/root_actual_qc.json`：14:42:45.302397Z，81events/13UPDATED，self `f4722169df413891194c75946516447dad294a8674517c0357a9d4d932b32ea3`；event78 raw `0e8d5f232cf25393ae8420e82a0615684aa8d8971e6f3149252adc0374dbe503`。
- `.codex/scratch/root_coordinator26_fourteenth_tell_eventchain_actual_qc_attempt01/root_actual_qc.json`：15:13:41.623646Z，87events/14UPDATED，self `b8feedb4edbb840f5e5bd7a4fa410fa2d1b1ed9da6ff5d54ba32cf0aef222280`；event84为837466B/raw `891709f48e5ec73cdab84c197faa5e260b391cae139c5316168223fac8d8aa6e`/self `e11c22bfef1394572b03d7a8e6bd0d791b60ec49fd790726a4649fead138edc6`。

第14次QA核实events81/83是source iteration13的k14 plus/minus，各三模型COMPLETE8/PASS8、0failed/incomplete，共48报告PASS；event84 source13→tell_result/controller_after14；events85/86为source14的k15 ask/plus intent。原7events逐byte不变，原PID/ticks/source/start/QUEUE/package12/session09/SHM03/recovered05均一致，SHM可用13,441,122,304B。Dev最新实际15:34:42.044269Z只读报告89events/14UPDATED，latest88为k15-minus/source14 intent，W2L COMPLETE8/rc0，Muse真实child运行，Ditto尚未启动；同一coordinator PID/ticks，无tell/error。该minus阶段为独立ROOT14 QA之后的代理观察，Task计数仍只取ROOT资格14，终态需再次核对。

**用户最新的休息需求。** 用户说要睡觉、GPU噪声大，询问最早什么时候可关机；这不是立即shutdown指令，也没有授权杀当前进程。Dev15:30:45Z实际读取最近5个完整pair的plus-intent→PAIR_TOLD耗时：36:14/35:29/35:23/34:56/35:40，均35:32；结合k15已到minus，20轮自然结束估计北京时间10月11日02:45–03:05，属于估算。ROOT早先按相邻QA时间估的02:15已向用户修正，不再使用。coordinator26源码确认20pair后退出、不调用中心，所以GPU会空闲；现无已资格的更早优雅pause/resume边界。停机前须完成内存盘恢复闭包备份，不能仅看到瞬时GPU 0%就关机。

Dev实际容量观察：当前dispatch 173852 regular files/5,141,108,428 logicalB，终局预计outputs约7.3–7.5GB，活跃runtime/source额外约5.7GB；远端root仅4,535,058,432B free，本地B盘3,755,865,747,456B free，SHM13,250,129,920B free。计划使用SSH流式归档至B盘，避免远端root暂存大包；必须保存原QUEUE、package12实际package11 runner08、runtime/model roots、原PCM/NPZ/media/receipt/source及symlink/模式恢复映射，并全member/hash核验。ASR代理正在准备可审查方案，未执行、未宣布备份完成或可关机。完整数学仍在中心之前，归档与关机时间需真实终局和备份回执再确认；用户尚未选择具体停机安排。

**监控纠正。** Dev先前口头第10轮物理目录偏一，第13轮又误把plus terminal说成minus。磁盘/科学迭代没有改动：实际event75=k13-plus terminal、76=k13-minus intent、77才是真正minus terminal，后者已由ROOT13 QA核验。监控现在从persisted最新POINT_DISPATCH_INTENT的proposal_id/point_id/pair_id/source iteration/side派生唯一目录，并在报告计数前核对各receipt的完整身份和captured UTC；不按宽glob或相对轮次口头标签找目录。第10轮纠正receipt `.codex/scratch/root_coordinator26_tenth_tell_eventchain_actual_qc_attempt01/root_iteration_point_mapping_actual_qc_attempt01.json` self `af5654c360a94c004b28bba28991f2fbc7b94a2a054051392bcf8d87afa52971`保留。ROOT只在实际独立读回后推进Task计数。

**全20数学门准备仍未执行。** 权威reader `.codex/scratch/qa_fit40_session_math_actual_attempt05.py`，113414B/raw `366b2fa59ad610421e8b09f65d9f0ad5d2a202b0b6ed15a9f2413baa3c13ac7a`；远端`JOB/root_independent_math_qa_attempt05/同名`0444已完整读回。snapshot01实际CPU1重放前2tell/17events/192unique NPZ，rc0、数学schedule误差0，PARTIAL_N_TELLS_QC；ROOT实际比较receipt `.codex/scratch/root_two_tell_math_reader05_actual_qc_attempt01.json` self `def9c922cbf94e580896c8e7d5e3db53b78e3012e951915a0ef946a877fbfe32`。第3–14tell没有新增原数组数学重放。

Full20快照源码 `.codex/scratch/fit40_session_event_snapshot_attempt03.py`，20194B/raw `c4c62e09cc3a419f898ec9dd5d056fc6b3b617d4a136873fba066a286892a94f`，已ROOT+ASR源码接受，UNSTAGED/UNRUN。真实20终局/121链式events/20UPDATED/双向COMPLETE/clean iteration20/no pending/no halt/writer.lock前后缺席后，才stage一次并建立fresh `JOB/fit40_session09_math_snapshot_attempt02`。复制manifest/events create-only、全byte读回及source前后不变；NPZ仍引用原pin路径，并非自包含归档。ROOT资格后reader05 max20/CPU1/原QUEUE执行一次，独立核验7个原始数组、F32官方归约/支持/PCMclock、固定initialθ regularizer、PCG64 asks、全部SPSA更新和journal链接后才seal journal。recovered completion05 raw `19d55ebd2b65af77c0bff26670020ce337822ed881d926a456945a4b15f612cd`/self `5a69b9e6143425721d91c17f66c80f920928b6b59654d9885f0675839dad45f5`保留，首tell旧rc2已按prefix-recovery修复完成凭据，不重跑ask/tell/native。

**四中心/选择runbook。** `.codex/scratch/fit40_final_center_execution_plan_source_only_attempt03.json`（21845B/raw `99bd1032a4233f1e9cd3118f8f04bdbcc99a2f98f1a50310c2fc3fedd6e41639`）及commands03（10001B/raw `1fa7f0148980bdbd452398adb86d2e5cb4195c1c74a1bd62a354f85a63921cbb`），ROOT `.codex/scratch/root_center_runbook_source_actual_qc_attempt01.json`为权威，SOURCE_ONLY_ACCEPTED_NOT_EXECUTION_RELEASE。14源码/5session inputs/实际runner全文与路径已资格；执行前fresh remote pins/capacity再核。bootstrap17/center12从history[k−1]/journal.entries[k−1]取中心。package12实际runner08仍从package11绝对路径`/dev/shm/fit40_spsa_execution_package_attempt11/project/scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_point_execution_attempt08.py`加载，73539B/raw `b1f0d863a1f8d3675fe0fc708e84a7804701f0b01eb759c2f8fe10f335ed8490`；不copy进QUEUE或改包重封。package12 manifest raw `fd805e8b7063d9b7478e699167c6cc12577ce886094c082a8f3da423ba59ae05`/self `bf86bda9b8361e5a5b0d8c45743bc6996fda6f8fab30b421bce594023703fd1d`。

按原QUEUE seal-journal→逐个prepare/execute k5/10/15/20→四terminal index→finalizer03 final720→实际ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV后portable06动态selected_step。journal=`JOB/fit40_durable_spsa_controller_journal_attempt01.json`；packets=`JOB/fit40_spsa_center_packets_attempt01/kNN_attempt01/`；中心dispatch=`/dev/shm/fit40_joint_spsa_full_centers_attempt01`（ROOT释放后create once），terminal=`dispatch/centers/fit40_d0_spsa_center_kNN/center_terminal.json`；index=`JOB/fit40_final_center_stagea_ingest_attempt01.json`；receipt=`JOB/fit40_final_center_selector_receipt_attempt01.json`。COMPLETE只读资格，UNKNOWN/partial先识别，不能换freshdir绕开。runbook内部逐步门由ROOT按既有自主授权释放，不是新增用户审批。

**显式11参数备用的source-only契约。** head=`scripts/audio_enhancement/shared_recipient_local_power_spectral_head_v1.py`，25881B/raw `8f22db628ec84f4b67d01e6e9c1bce12edf9280cca88e52cfc55d32d7b94dcfe`。固定向量消融receipt `.codex/scratch/root_signal11_fixed_vector_source_contract_actual_qc_attempt01.json`；四cell `rms_reference/envelope_only/shape_only/full`必须同完整拟合向量、共享κ=min(1,3/(2A+B))及一次源STFT Parseval shape补偿，移除支路不重算κ。9个tanh谱形+2个tanh宽带包络坐标，形状补偿后STFT帧功率与最终重建/RMS后40ms波形功率分别记录。3dB界仅为全局RMS前构造STFT mask界。固定向量功能消融不是分别重训练的因果factorial；Yfull−Yshape−Yenv+Yrms只称功能非加性。11须独立family/schema/有序11F32 hash/config/binding/apply/plan/portable，现有9参数validator不放宽。

新增 `.codex/scratch/root_signal11_warmstart_source_actual_qc_attempt01.json` self `874384dc18dde71c1f61209956a071f7dbdd6a65e7965bf705e5b159ff01fbea`：ROOT实际读旧V3（32233B/raw `77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87`）及新head，做源码/代数审核，Signal独立边界审核一致，均未import/apply/test。Task小节《旧9参数到11参数的系数继承边界》是完整契约：

- 旧θ直接dB到新`s_shape*tanh(z)`无符号翻转；实数精确系数逆`z=atanh(θ/s_shape)`仅当全部|θ|<s_shape，两个env坐标置0。到达/超出scale没有有限精确逆；若采用clip，ε和dB残差必须预声明并报告。实数恒等不等于F32逐位相同，后续需有序11F32/hash与实际系数恢复误差。
- 旧mask为r(A)S，r=tanh(A/3)/(A/3)、r(0)=1；新zero-env为κS+C_t，κ=min(1,3/(2A))。相同basis/config/gates时κ=r且C_t≡0足以使mask相同，C_t=0单独不够。全env时κ=min(1,3/(2A+B))。
- C_t=0不是最终归一化波形相同的必要条件：理想线性完美重建、无peakcap时，整个时频平面的常量dB差对应正全局缩放，目标RMS会消去该缩放。通常C_t随帧变化，且F32/残差STFT不保证逐位等价。
- 旧Fit选中θ可用于明示的Fit-dependent method development，不自动降为diagnostic-only；新方法完整协议、scale/ε、坐标步长、warmstart身份须先于该方法效果运行冻结，Dev/Confirm封闭。旧direct-dB SPSA步长不能直接搬到tanh坐标，导数s*sech²(z)随坐标变化。当前未选择数值θ/scale/ε或映射。

**机制已有结果范围。** `.codex/scratch/root_first_pair_acoustics_actual_qc_attempt02.json` self `d2fc402bbc8c95b8e8f3d8b33672b0f1d7cd9424f76d21bec957c6a6b7d55ac5`：128WAV/3544帧/64数组实际独立复算支持整段RMS相等时，近静态谱形通过局部频带组成改变时变条件特征。该结果不代表全部TTS的因果比例或新候选泛化。机制BM《近静态谱形通过局部谱成分改变三类 TFG 的时变条件特征》与Task保留方法细节。11的谱形补偿与宽带包络支路将进一步区分功能路径，不反馈改变当前9参数训练。

**Dev/Confirm仍封闭。** Fit实际准入后先修复draft protocol06与builder05/analyzer04/nativeadapter03固定旧引用，封存真实候选/runtime/base360/input来源，生成最小后继protocol07/consumer06/05/04并保留cohort/policy01/ADR031及所有统计门。新540行=20clips×3models×3seeds×3methods，不复用旧360、不提前读Confirm80/LatentSync。FULL Cbest需每模型对两参考正、等模型比RMS≥+0.050、GUARD20正、D0/Canchor双支持/双参考不负，ASR/visual/speaker-bootstrap门不变。

候选ASR producer仍缺：实际selected generator F32 WAV→20 mono16k PCM16 payloads→固定qwen3-asr-flash-2026-02-10/zh/ITNfalse/no prompt/不重试去重20calls→COMPLETE_20_CALLS_PASS summary。builder外部键receipt、内部receipt_ref；plan.spsa_binding.application_receipt绑定fullpath/bytes/rawSHA/self，generator_condition_input raw/self键为application_receipt_raw_sha256/application_row_receipt_sha256。禁止用score_audio原音替代候选。旧60基线只在新540每clip九行RMS F32 digest与旧rms_intermediate逐位相同且lineage合格时复用；路径不可读不自动重打API。readiness=`.codex/scratch/root_candidate_asr_binding_readiness_attempt01/`。Dev20源ZIP已ROOT核验，Fit准入后stage到fresh wjj目录，不用sudo/旧不可读root路径。该producer与后继冻结尚未实现/执行。

**持久性边界。** 旧session07的12877文件已在本地与B盘完整逐body归档PASS，远端原件保留，百度云同步未知，入口见HANDOFF。当前SHM仍约13.44GB可用，无需现在清理。大量新源码、BM、runs和scratch是dirty/untracked/ignored本机状态，不能把Git SHA当作这些产物的完整备份。旧get_goal额度状态为blocked不代表实际研究暂停；不要为绕开该状态创建新goal或标记完成。当前研究仍active，遇到完整9门失败继续新方法。

## Changed Files

- `.codex/scratch/root_qualify_coordinator26_{twelfth,thirteenth,fourteenth}_tell_eventchain_attempt01.py`及同轮`root_coordinator26_*_tell_eventchain_actual_qc_attempt01/`：实际只读QA和完整raw readback；没有训练源码改动。
- `.codex/scratch/root_sync_{twelfth,thirteenth}_tell_actual_attempt01.ps1`、14的sync/completion revision输出目录：Task/Experiment/HANDOFF/CONTEXT同步与直接读回证据。14 attempt01在Task写入前遇SQLite lock，attempt02 Task全文读回PASS后Experiment写入遇lock，attempt03同一Experiment写入仍lock，失败均保留；`.codex/scratch/root_complete_fourteenth_tell_sync_attempt04/`最终实际完成并读回全部入口PASS。未停止未知本地BM进程。`.codex/scratch/root_prepare_next_tell_memory_sync_attempt01.py`只生成下一轮本地同步草稿，不执行草稿或修改远端，14草稿已人工审核。
- `.codex/scratch/root_signal11_warmstart_source_actual_qc_attempt01.json`与`.codex/scratch/root_record_signal11_warmstart_source_attempt01.ps1`：source/symbolic资格与Task追加读回，旧9训练不变。
- `HANDOFF.md`/`CONTEXT.md`（≤100行）及上述BM活跃入口：当前14次更新，checkpoint指针在本次创建/readback后更新。历史checkpoint全部不可变。

## Verification

ROOT实际SSH只读取得12/13/14事件链全body，检查原7event byte、self/hash链、controller iteration、双向point和child terminal报告身份/计数、PID/start_ticks/source/package/recovery/cwd一致；14次最后读回PASS，未把代理口头标签当终态。未新增原数组数学复算、模型运行、API调用、候选选择或Dev/Confirm读取。

14次BM同步实际readback通过：Task context/Observations与Experiment phase、HANDOFF及CONTEXT精确相符；warmstart小节先前独立读回PASS。ROOT warmstart源码/代数审核与Signal独立审核均仅资格边界，不是数值映射或科学效果实验。没有新增或运行实施测试。

本次先读Startup Router、实验指令、Coding Session/Task schemas与BM配置；canonical GitHub仓库/branch PR/HEAD可见性及same-chat coding/codex/legacy全页搜索在checkpoint wrapper内重新只读核验。CLI已核对其write_note参数传overwrite=False/output_format=json；本次只接受action=created，再直接全文读回校验正文、agent/session/model/trigger/started，更新HANDOFF指针并读回。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；当前SHA `5688784dd5c6c1272074f0c4b6c5ff79f117c907` 为local/unpushed，不构造未解析GitHub commit链接；branch PR实际[]。
- `tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement`：Fit联合原生目标；`HANDOFF.md`：实际当前执行和历史归档入口。
- `.codex/scratch/root_full20_snapshot_and_candidate_asr_source_actual_qc_attempt01.json` self `598eaa0ba66739276cd0fe2908edf8359b6c682563d99ba1ba733d28f38b3926`：full20 snapshot和候选ASR source-only资格。
- `.codex/scratch/root_center_runbook_source_actual_qc_attempt01.json`及plan03：完整数学之后的四中心/选择权威步骤。

## Observations

- [result] ROOT15:13:41Z独立核验87events/14UPDATED，同一coordinator26 PID785358/ticks10166461及原执行闭包；第14轮双向三模型48报告PASS，第15轮plus/source14运行中。Task/Experiment/HANDOFF/CONTEXT实际同步读回通过。
- [result] 旧9到新11仅具源码/代数继承资格：atanh映射继承系数，径向投影/共享κ/时变补偿使波形通常改变；当前未选数值初始化，也没有11效果运行。
- [decision] 监控路径由最新persisted intent的完整身份唯一派生，并核对child receipt身份及captured UTC；旧口头目录误报保留纠正证据，不重跑科学产物。
- [decision] 原QUEUE、package12、原自然PCM/clock与所有门不变；20轮全数学先行再四中心/final720，完整9门失败进入独立11方法闭包，Dev/Confirm继续封闭。
- [blocker] 全20原数组数学、中心选择、实际候选portable、Dev冻结后继与候选ASR producer/540新Dev尚未完成；当前是活动训练阶段，尚无新的跨TFG胜过RMS结论。
- [next_step] 沿用原Dev代理只读跟踪当前唯一coordinator26至真实20次UPDATED终局及writer.lock缺席，再由ROOT释放full20 snapshot03与reader05/CPU1完整数学核验，合格后推进原QUEUE四中心/选择。

## Relations

- continues [[Codex checkpoint - 2026-10-10T13-39-39Z - Eleven updates and event identity alignment]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]