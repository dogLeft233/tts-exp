---
title: TTS 时间辨识与同文本实例交叉实验 Spec
type: research
permalink: tts-exp/research/tts-时间辨识与同文本实例交叉实验-spec
status: spec_ready
date: '2026-09-17'
protocol: tts_time_instance_v1
execution_status: complete
tags:
- tts
- syncnet
- mechanism
- temporal-identifiability
- instance
- spec
result_note: tts-exp/experiments/tts-时间辨识与同文本实例交叉实验-2026-09-17-结果
run_id: tts_time_instance_20260917_v1
---

# TTS 时间辨识与同文本实例交叉实验 Spec

## 任务与完成边界

本 spec 面向下游 GPT Luna 等实现模型。实现两个小实验，并分别交付结果：A 使用历史 LeapTalk/SyncNet 特征检验时间辨识；B 用同文本两次独立 TTS 合成检验实例适配，并附带上下文负对照。当前文档状态 spec_ready，已执行；结果见 [[TTS 时间辨识与同文本实例交叉实验 2026-09-17 结果]]。

科学问题：TTS 的 Sync-C 优势是否伴随更好的局部时间辨识？控制音素身份与相对位置后，生成视频是否仍偏爱自己的合成实例？上下文匹配是否解释部分同音素实例差异？

A 是既有原生优势的回顾性诊断；B 是新的 Wav2Lip 小实验。B 的模型与历史 LeapTalk 不同，必须报告自身原生优势，不能称为旧 LeapTalk B 阶段的补全。本轮不训练增强头、不运行参数搜索、不建立通用实验框架。建议一个 runner、一个独立复算脚本、一个测试文件即可。

执行顺序：输入审计 → A 无音素分析 → A 音素事件分析 → B 音频和对齐 → B 生成与评分 → 独立复算 → 报告。A/B 独立完成；某个依赖缺失时保留已完成结果，不将其解释成科学阴性。

## 1. 历史核对：什么已经做过

搜索覆盖项目 BM 的 Experiments、Research、Tasks、旧 docs/experiments，以及本地 openspec/changes；关键词包括可辨识、单位范数、上下文、phone-context、same-phone、instance、specificity、共适应、triphone、同文本实例。无命中项也搜索了全局 main。以下“未见”仅限可核验项目记录，不是对一切历史的绝对断言。

| 假设 | 已有实验与实际结果 | 本次真正新增 |
|---|---|---|
| 时间辨识 | [[MFA-linear 轨迹增益的匹配项与背景项分解 2026-09-16]]：自然/TTS 轨迹剂量都提高 C，背景项主导；[[LRS3 TTS 原生优势的分数分解与曲线诊断]]：原生优势、曲线与边界分析；[[TTS 原生增益来源的生成端与评估端交叉诊断]]：单位范数 180 行、错误内容检索 396 对已经完成 | 保持错位距离可比，比较正确时间的局部排名；再区分同一音素 occurrence 内错位与跨音素事件错位。不能把普通 C/B/D、单位范数、整句错误内容检索再次当新实验 |
| 上下文发音 | [[LRS3 masked TTS trajectory-specificity diagnosis]] 已有 wrong-instance 与 reversed；原模型重建证据不确定，加入 hard-negative 训练后 paired-over-wrong 的 C 中位增益 +0.548、paired-over-reversed +0.363；是已见数据、训练后模型，不是原生 TTS 机制。[[AISHELL-1 phone-context residual transition experiment]] 只做边界 crossfade 音频 QC，未跑下游。最新 P/R 实验功能干预未获支持 | 同一中心音素、同一句话内，按左右音素上下文构造不同实例负例。此项只定位评价器的上下文敏感性，尚不构成对生成器上下文机制的因果识别 |
| 实例适配 | 旧 tts_tfg_mechanism_report 已有 N/T 2×2、交互 +7.290，但 off-diagonal 只有全局时长对齐；多个 strict replacement 失败。它们含局部 timing 与声学差异，不能全部归为音色。现有归因 B 尚缺正式生成数据 | 同文本、同 TTS 模型、同参考音色的 T1/T2，完整双向交叉；音素 occurrence/phase 控制，平方距离交互消去独立音频/视频范数项 |

结论：三个宽泛方向均非全新；两次独立 TTS 的双向实例交叉，以及按音素事件分层的时间排名，没有找到已完成的等价实验。对同音素控制的旧阳性必须保留，不能继续写成“从未测过实例结构”。

## 2. 冻结输入与实现入口

仓库根为 [redacted-local-path]

固定 cohort：sample_id 151…162，共 12 条、12 个 source groups；来源以以下历史清单为准，不用 speaker_key=lrs3 代替 group。保留全部 12 条台账，缺失项明确记录，禁止按得分替换样本。

- openspec/changes/disentangle-tts-native-gain/input-bindings.json
  SHA256 72ae4fcd3e3a4a0d14cba00142edc68c8f40b0ae24ecdae2266f6016318570e2
- runs/lrs3_tts_gain_mechanism_review_v15/00_audit/cohort.json
  SHA256 e2b511c3ef9f500739a37232bac139c3110e6ec55b40f1138240098ae22772eb
- runs/tts_native_gain_attribution_completion_20260916_v1/02_fixed_video/a_manifest.json
  SHA256 e11c26b0e5f1ca9bd28d8ec15cde207136caad1e76e0b7a67f4f9d2343f9b364
- runs/tts_native_gain_attribution_completion_20260916_v1/05_analysis/summary.json
  SHA256 54f4b270767dc79cb279cfd6c8a93da052929e5b0c2a8c2d7089039b6c1313e0

A 的 feature root：
runs/tts_native_gain_attribution_implementation_20260915_v1/02_fixed_video/features/

每个 id 读取：
- audio/N/{id}/ORIGINAL.npy 与同名 json；
- audio/T/{id}/ORIGINAL.npy 与同名 json；
- visual/V_N/{id}.npy 与同名 json；
- visual/V_T/{id}.npy 与同名 json。

上述 48 个特征文件已在编写时逐个核验存在且匹配 metadata.sha256。音频路径由对应 json.audio_path 读取，PCM hash 必须匹配。A 只用 ORIGINAL，不把 A0、GAIN、NOISE 等混入 N/T 原生比较。续跑目录会引用原始目录，按 manifest 解析路径。

复用入口：
- scripts/experiments/tts_native_gain_attribution/syncnet.py：SyncNetEngine、extract_audio、extract_visual、distance_matrix；
- 同目录 analysis.py：参考官方曲线口径和已有诊断；
- scripts/tts/faster_qwen3.py：本地 TTS provider；
- third_party/Wav2Lip/inference.py：B 的冻结生成器。

写新代码前按 AGENTS.md 使用 CodeGraph。若索引未覆盖近期文件，再读取上述精确路径。本 spec 不要求重构旧包；独立验证器不 import 新 runner 的统计与采样函数。

输出：runs/tts_time_instance_<run_id>/，新 run 不覆盖历史。inputs.json 记录 spec、实际代码、cohort、音频、特征、checkpoint 的文件 SHA；无需给每个中间 JSON 建一套自哈希框架。

## 3. 共同统计与状态

统计抽样的随机数固定为 bootstrap：numpy.random.Generator(PCG64(20260917))，20,000 draws。先得到每条记录一个值，再 source-group 等权；当前每组一条。对同一可用组集合的全部对比共享 bootstrap 索引。禁止把帧、音素、seed 当独立样本。

四个主统计预先固定：A_rank、A_event、B_instance、B_transfer。各报告普通 95% 区间与 Bonferroni 98.75% 区间，分位数 linear；缺失某项也不缩小四项检验家族。报告组数、每组值、正向数、效应均值。Sync-C 表格保留 3 位小数，机器文件保留原始精度。

每项独立标记 POSITIVE（校正区间下界>0）、NEGATIVE（上界<0）、INCONCLUSIVE（跨0）、NOT_ESTIMABLE（覆盖不足）。工程状态与科学状态分开。没有显著差异不等于等效；不以扩大样本、改阈值或换模型“疏通”阴性。

基线原生 ΔC 是 effect_present 参照，报告 95% CI，不并入四项机制检验。若未复现优势，仍计算可用机制统计，但结论写“当前协议下的匹配响应”，不能声称解释了已复现的 TTS 增益。

## 4. A：时间辨识与音素事件

### A0：重放与坐标

A 无新 TTS/TFG/SyncNet forward。用缓存特征重新构造官方 float32 pairwise_distance（eps=1e-6）的 31 列矩阵，逐 cell 对照历史 ORIGINAL 矩阵；max_abs_error<=1e-4。复现历史 INTERIOR 的 C/B/D 作为输入校验，并报告本实验支持上的原生 ΔC。

列 j 对应 audio index i+j-15；令 k=j-15，则 official_offset=-k。在新自定义距离中，用 float64 的严格欧氏距离，无 eps；与官方分数分开命名。不能把平方距离、自定义 rank 称为 official Sync-C。

对每条 N/T arm 分别令 F=min(len(v),len(a))。所有 A 主分析使用 U=range(30,F-30)，保证校准 k∈[-15,15] 后再加 ±5 不越界。若 U<20 行，保留该记录为缺失，报告原因。A_rank 至少需要8个N/T均有效的source groups才出主区间；否则NOT_ESTIMABLE。

固定交叉校准：按 floor(i/25)%2 将 U 分成两折。对一折的待测行，使用另一折估计唯一全局 k0，使校准行平均原始欧氏距离最小。tie 先选 |k| 小，再选 k 小。所有 raw/unit/event 分析共用该 k0；禁止逐帧、逐类别重新找最低点。每折校准和待测均需>=10 行，否则该 arm 不可估计。

该校准是消除固定 lag 的探索性办法，不是已知真实同步真值。按连续秒分折仍存在时间相关；统计单位始终为 source group。

### A1：尺度不变的局部时间排名（主统计 A_rank）

对每个待测 i：
- 正配距离 d0=||v[i]-a[i+k0]||；
- 错配 delta∈{-5,-4,-3,-2,+2,+3,+4,+5}，d_delta=||v[i]-a[i+k0+delta]||；
- 每对胜率 w=1 if d_delta>d0；0 if d_delta<d0；严格相等时0.5。使用float64，禁止加入固定绝对tie阈值，否则正仿射变换会改变近tie分类；
- R_arm=所有 i、delta 的 w 等权均值。
- A_rank_i=R_T-R_N，正值表示 TTS 对 80–200ms 局部错位更可辨识。

补充报告 ±1（40ms）胜率，但不纳入主统计。对单位范数 v/a 重复计算 R_unit，作为预注册次要诊断，不新增主检验。raw rank 对全曲线正向仿射变换不变，但未排除逐帧范数或其他几何影响，不能称为完全去除声学域偏差。

必须保存每行 k0、d0、全部 d_delta、raw/unit w。若原始 C 高而 rank 不升，只能说“未见局部排名改善”；不能直接断言所有增益来自尺度或评价器被欺骗。

### A2：音素事件分层（主统计 A_event）

对与缓存 audio 对应的原始 PCM 做英文 MFA；优先复用输入 PCM/hash 完全相同的已存对齐，否则用本地固定版本 MFA、英语词典和声学模型各对齐一次。记录模型版本/hash、命令、TextGrid。转写使用父清单绑定文本，先核验它与选中的 crop 音频覆盖一致。父音轨被裁剪时按原 crop 时间偏移取词，不能给局部音轨直接套完整文本边界。MFA 不可用时 A1 照常完成、A2=MFA_UNAVAILABLE；不使用整句均分假边界。

音素 occurrence 编号按 TextGrid 的 phones tier 时间顺序；词典 phone label 大小写原样保留，空白/sp/sil/silence/spn/<unk>（不区分大小写）排除。区间左闭右开，零时长不参与。

缓存 MFCC 为20个10ms间隔、25ms窗，其第 u 个输入块的名义声学支持是 [0.04u,0.04u+0.215)，中心 c(u)=0.04u+0.1075。时间相对缓存 PCM 起点，不凭视频中心代替它。若实际前端与此不符，先记录真实坐标并修复兼容层后重放，不能静默沿用。

对 A1 的每个(i,delta)，检查 c(i+k0) 与 c(i+k0+delta)：
- SAME：两个中心属于同一个非静音 occurrence；
- CROSS：两个中心属于不同 occurrence，且 phone label 不同；
- 同 label 不同 occurrence、任一静音/未知/越界：单独列账，主事件分析不用。

这只是“窗口中心的音素事件标签”；SyncNet 窗口约215ms，可能跨多个音素。不得称为隔离了纯音素核心。另存每个窗口覆盖的 phone 序列和各 phone 重叠比例供审阅。

每条记录按 h=|delta|∈{2,3,4,5} 分层。同一 h 只有 N/T×SAME/CROSS 四格均>=5 对时才可用；先每格平均 w，再：
E_i,h=(R_T,CROSS-R_N,CROSS)-(R_T,SAME-R_N,SAME)。
A_event_i=该记录全部合格 h 的等权均值。报告所有四格计数、保留 h、phone 分布与 N/T 时长比，禁止把所有跨界样本与所有不跨界样本直接混池。

有合格 h 的记录>=8组才计算主区间；不足则 NOT_ESTIMABLE 并保留描述性值。该 observed-support estimand 与 A_rank 全队列不同，报告两者分母，不能声称覆盖全部12组。

A_event>0 表示 TTS 的排名增益更集中于跨音素标签事件，属于事件相关证据。MFA测量误差、语速、上下文和原生视频差异仍存在，不能直接叫事件的声学因果效应。

### A 控制

1. 对任意矩阵应用 d'=3d+7，rank 应完全不变；C 应按3倍缩放。
2. v/a 复制输入应使 N/T 对比为0。
3. 合成 one-hot 时间序列，音频移位5行；校准 k 的方向与预期一致。
4. phase/label 边界玩具例验证 SAME/CROSS，单独测试同label不同occurrence。
5. A1不依赖MFA；A2缺失不能把A1全部标成失败。

## 5. B：同文本 TTS 实例的双向交叉

### B0：为什么另做与资源预算

旧 N/T 换音轨无法区分 TTS 通用表达与原配实例细节。B 固定同一句文本与同一自然参考音频，新生成 T1/T2；以同一张脸分别驱动 N/T1/T2，保存3×3音视频匹配。

固定仍为151…162，不按A结果挑样本。使用本地 faster_qwen3 0.6B ICL、两个 seed 42/43；模型、文本、reference PCM、语言与sampling参数均一致，仅seed不同。两臂均新生成，不把旧seed42与另一个版本新seed43混搭。固定 model_id=Qwen/Qwen3-TTS-12Hz-0.6B-Base、strict_backend=true、language=English、max_new_tokens=4096；ref_text=text。其余sampling参数使用所绑定本地版本的默认值，在首条生成前展开并存入inputs.json，后续保持完全相同；不允许静默fallback到qwen_tts。冻结实际本地配置并记录版本/hash。每次调用用独立子进程或局部 RNG 作用域；不修改 provider 的全局种子契约。不切换云API或其他TTS模型补失败。

预算：24条新TTS；36个科学视频；ID151的N/T1/T2各独立重复一次，共39个TFG render；39个原生SyncNet cell加ID151三个来源的±200ms延迟控制6个，共45个原生评分。108个phase交叉量由36组缓存音视频embedding在CPU计算，不能计为108个新模型前向。不训练，不用vocoder，不做波形phone warp。

若本地TTS权重或Wav2Lip环境缺失，记录具体缺失，B=DEPENDENCY_BLOCKED；A仍完整交付。本 spec 不依赖缺失的LeapTalk配置。

### B1：生成与原生参照

N 使用父绑定自然PCM。T1/T2用scipy.signal.resample_poly，up=16000/gcd(sr,16000)、down=sr/gcd(sr,16000)，固定默认Kaiser窗；以numpy.rint(32768*x)后clip到[-32768,32767]并转int16，保存重采样后量化前clipping fraction；不调响度、不人工修复波形。保留原始TTS与canonical版本、采样数、peak、clipping fraction及hash；同一记录T1/T2若PCM完全相同则为DEGENERATE_INSTANCE，不重新抽seed。

文本为父资产绑定的同一句文本；自然参考使用同一N。为N/T1/T2分别做固定MFA，所有speaker/aligner配置相同。按“词的文本和出现次序，再按词内phone次序”做三方exact-label occurrence匹配。插入/缺失/OOV只列账，不用最近同音素替代。silence不参与phase匹配。

冻结Wav2Lip官方wav2lip_gan.pth，参考图固定取本记录自然输入crop的首帧；N/T1/T2及repeat均用完全相同的图与固定face box，25fps静态模式。用每cell独立cwd/temp；输出保留各driver自己的原生时钟和完整音轨。不用原始动态视频作为三臂不同长度的姿态输入，避免循环/截断造成视觉混杂。checkpoint、运行命令、图/box、seed记录在manifest。

冻结官方SyncNet V2；每记录三个生成视频使用同一评分ROI策略，ROI只由固定参考图确定，不按分数重新选crop。提取完整a/v embedding及31-lag矩阵。原生C/D使用同一INTERIOR定义：每arm的F=min(len(v),len(a))，support=range(15,F-15)，先按support对时间均值，再在31个lag上取D=min、B=median、C=B-D。N/T1/T2各自时间轴与各自support，不用逐窗口一一配对冒充等时钟。

原生参照：
G_i=0.5*(C(V1,A1)+C(V2,A2))-C(VN,AN)。
报告G的95%组CI，以及T1/N、T2/N。G区间跨0则native_advantage=UNCONFIRMED；继续交叉分析，但不把其结果用于解释此设置中已成立的增益。

repeat应满足相同输入下raw矩阵max_abs_error<=1e-4；否则报告生成重复噪声、B结论INCONCLUSIVE，不以新seed替换。延迟+3200samples应使最优k增加约5帧（容许1帧），负延迟相反；以共同有效支持验证，并分别保存控制结果。

### B2：共同音素坐标，避免波形拉伸

三方匹配得到有序K个非静音occurrence。前ceil(0.2K)个仅校准lag，其余作evaluation。每臂在校准occurrence中心对应的原生行上，按A0规则确定唯一k_s∈[-15,15]；不使用evaluation距离找lag。对每个校准phone的center取i=floor((center-0.1075)/0.04+0.5)，去重，仅保留所有k∈[-15,15]都不越界的行。校准>=5个不同窗口，不足该记录为ALIGNMENT_INSUFFICIENT。把这些visual窗口以及搜索过的全部audio窗口的真实时间支持取并集，evaluation查询的audio/visual插值邻点窗口与该并集的间隔均至少0.24s；N/T1/T2任一不满足则共同删除。

每个evaluation occurrence在N/T1/T2中分别取phase {0.25,0.5,0.75}：
t_s=start_s+phase*(end_s-start_s)，u_s=(t_s-0.1075)/0.04。
a_s(t_s)在线性插值位置u_s读取；v_s(t_s)在位置u_s-k_s读取。这个索引关系来自d(v[i],a[i+k])，不要再额外加减80ms。对a/v都先插值原始embedding，再各自L2单位化。任何插值邻点越界、零范数、非finite，删除的是该查询在所有9格的共同支持，并记录原因；不外推补零。

phase并不保证声学事件或嘴形完全相同；这是MFA occurrence/phase条件下的比较，不能称为精确articulatory alignment。所有9格都采用同一插值流程，避免只有off-diagonal有插值。

每个查询z有v_N,v_1,v_2,a_N,a_1,a_2；定义：
d_ij(z)=||v_i(z)-a_j(z)||²。
同一记录所有9格使用完全相同查询集合。先3个phase在occurrence内等权，再occurrence等权，再source组等权。只有三个phase均有效的occurrence参与主分析；至少8个evaluation occurrence/record、至少8个完整record才能估计B主区间。其余报告缺失和observed-support范围。

该d是单位范数embedding的自定义平方距离，不是Sync-D，不能转换为Sync-C。另存未单位化平方距离作次要对照，避免把单位化后的阴性一概归为范数机制。

### B3：两个主统计

记d_ij为上述每记录均值。

B_instance：
I_TT=0.5*(d_12+d_21-d_11-d_22)。
正值表示T1/T2视频平均偏爱自己的合成实例。双向交叉能抵消某一音轨普遍好评的加性主效应。

逐查询独立校验恒等式：
I_TT(z)=(v_1-v_2)·(a_1-a_2)。
raw平方距离也应满足相同恒等式；独立音频/视频范数项在交互中相消。不将所有正交互归为“伪影”，残余局部时序、韵律和发音实现也可能贡献。

B_transfer：
X= d_NN - 0.5*(d_1N+d_2N)。
正值表示在共同phone-phase坐标和同一自然audio embedding下，TTS驱动视觉更匹配自然音频。这是“音素坐标下的迁移”，不是播放原始N音轨的strict replacement；因为坐标映射参与了评价，不能宣称可部署replace增强。

次要量：
- I_NT1=0.5*(d_N1+d_1N-d_NN-d_11)，T2同理；
- 相比N/T交互，I_TT有多大：报告两者及差值，不用接近零分母算比例；
- T1/T2的phone duration差、phase sampling舍弃率；
- 用phase={0.4,0.5,0.6}重复一次描述性敏感性分析，不另挑最有利phase。

读数规则：
- I_TT正、X不正：支持实例相关匹配；尚无可迁移视觉优势证据。
- X正：支持在此音素坐标下的视觉迁移线索，需另做真实自然时钟验证。
- I_TT不确定：不能宣布实例可互换。
- I_TT正且G不确定：能说明实例匹配，却不能说明它导致TTS原生增益。
- 任一结果均不等于人工同步或音质改善。

## 6. B附加：上下文匹配负例（预注册次要，不阻断B主分析）

本项补“同音素但上下文不同”的缺口；不重复把所有wrong-instance混为一类，也不训练hard-negative模型。它是固定视觉embedding、改变评价音频窗口的干预；不能直接识别生成端因果。

在每条记录内，只用B evaluation occurrence作为query/donor。context=(left_phone,center_phone,right_phone)必须取自各臂原始完整TextGrid phones tier的直接相邻occurrence，而不是删除静音/未匹配项后重新接邻居。N/T1/T2的左右邻居标签须分别完全一致；句首尾、任一邻居静音/未知、或三臂邻居标签不一致的occurrence不参与。

候选r必须是同句另一个occurrence、center_phone相同、与q在N/T1/T2所有时间轴的中心都相隔>=0.4s。定义context_match_count=左右邻居分别相等的数量（0/1/2）：
- r_plus：match_count>=1；
- r_minus：match_count=0。
不从其他source借donor，避免混入speaker与录音条件。每个q须同时有plus/minus，否则只计coverage。

从所有plus/minus组合中，依次按以下键最小选一对：三臂内plus/minus的中心phone log-duration差绝对值之和；-plus的match_count；plus occurrence序号；minus序号。选择过程只读labels/times，不看embedding或分数。另报告两donor对query的duration差，duration仍可能混杂。

在相同phase读取两个donor的a_s；query使用v_s(q)，s=N/T1/T2。所有s用完全相同q/plus/minus/phase集合；定义：
K_s=mean_z[||v_s(q)-a_s(r_minus)||²-||v_s(q)-a_s(r_plus)||²]。
K_s>0表示相同中心音素下，更匹配邻居上下文的窗口更容易与query视觉匹配。
K_T=0.5*(K_T1+K_T2)，报告K_N、K_T、K_T-K_N及95%描述CI，不计入四主统计、不做独立“机制已证实”判定。

至少每记录3个query、至少6组才出组CI，否则CONTEXT_COVERAGE_LOW，仍完成B主分析。给出2/1邻居匹配各多少、donor复用数、词身份差异。此比较还混入词位置、重音和约215ms窗口内容；它只能检验上下文相关匹配，不能证明“上下文一致性导致原生增益”。

## 7. 实现、审阅与交付

建议脚本 scripts/experiments/tts_time_instance.py，子命令 audit / a / prepare-b / render-b / analyze-b / validate / report，或等价简洁实现。无必要引入服务、数据库、调度器、插件系统或训练代码。

必需产物：
- inputs.json：冻结输入、版本、cohort、实际可用/缺失状态；
- A：replay.json、time_pairs.csv（可压缩）、phone_events.csv、a_summary.json；
- B：audio/video/features manifest、3×3逐query距离、每条记录9格矩阵、lag_calibration.json、context_pairs.csv、b_summary.json；
- validation.json：逐项实际重算状态；
- report.md：历史重复性核对、原生效应是否存在、四主统计、分母、限制、下一步；
- 两张图足够：A的错位距离—胜率曲线；B的3×3均值热图和逐记录I/X散点。没有人工听检/观看评分时明确NOT_ASSESSED。

独立复算脚本从冻结embedding、对齐、采样清单读入，不调用新runner的采样/统计函数；重新构造A的折/k0/排名/事件标签，B的phone-phase索引/9格距离/I/X，独立重建bootstrap并比较。允许用同一NumPy库，不允许只数文件或读producer填的PASS。

必要测试限定为：
1. 官方矩阵lag符号与合成已知移位；
2. rank对正仿射变换不变，ties处理；
3. 同label不同occurrence、边界及静音分类；
4. phone/phase映射identity和双向映射，插值共同支持；
5. 2×2平方距离交互恒等式、T1=T2时I=0；
6. bootstrap以group而不是frame计数；
7. 缺依赖/缺alignment/coverage低时输出真实缺失，A与B互不伪造成功。

审阅后若发现实现bug，修复并在新run或明确resume中重算受影响产物；不得因结论不好修改seed、phase、lag、cohort或donor规则。文档不足以决定的重要科学改动记录为protocol amendment，再执行，旧结果保持原样。

完成条件：可运行阶段的数据与独立复算通过；不可运行阶段写出具体依赖和未完成数量；四项主统计没有用次要结果代替；报告明确是否存在原生增益。工程实现完整与科学机制成立是两个独立判断。

## Observations

- [status] spec_ready，已执行并完成新实验；结果见 [[TTS 时间辨识与同文本实例交叉实验 2026-09-17 结果]]。
- [question] 原生TTS增益是否伴随事件相关时间辨识，并可在同文本不同合成实例及自然音频表征之间迁移？
- [history] 时间曲线分解、单位范数、错误内容检索、wrong-instance/reversed及N/T交叉已有历史；不能将它们重新包装为全新实验。
- [decision] 本轮新增局部排名/音素事件分析与两次独立TTS的双向交叉；上下文作为有明确边界的次要评价端探针。
- [boundary] phase交叉不等于原始自然音轨strict replacement，评价端上下文响应不等于生成端因果。

## Relations

- follows [[TTS 声学变化重组轨迹与剩余项干预：实现与结果]]
- relates_to [[TTS 原生增益来源的生成端与评估端交叉诊断]]
- relates_to [[LRS3 masked TTS trajectory-specificity diagnosis]]
- relates_to [[MFA-linear 轨迹增益的匹配项与背景项分解 2026-09-16]]
- relates_to [[LRS3 TTS 原生优势的分数分解与曲线诊断]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 核对历史重复性，绑定现有特征，编写两阶段下游实验与上下文次要对照；未执行实验 | September 17, 2026 | user request |
| 自审修正rank近tie尺度问题、TTS后端/语言/重采样契约、校准支持隔离及原始phone邻接定义 | September 17, 2026 | user request |
| 完成实现、重算最终 run、独立复算与 12 项协议测试；结果笔记已关联并标记执行完成 | September 17, 2026 | user request |
| 修正父资产 transcript hash 语义与成功重试的 stale error 标记；最终输入哈希、验证和报告已重新绑定 | September 17, 2026 | user request |
