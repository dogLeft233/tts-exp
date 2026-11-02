---
title: TTS 来源方向与声学机制确认
type: experiment
permalink: tts-exp/experiments/tts-来源方向与声学机制确认
status: implemented
started: '2026-09-30'
repository: dogLeft233/tts-exp
---

# TTS 来源方向与声学机制确认

## 问题与因果路径

解释 TTS 原生评分优势，并找出能在同一自然音频、同一时钟下产生生成器收益的分量。根问题分为声学波形与前端、Wav2Lip 生成器、SyncNet 评分音频，以及时序/错配距离几何四条路径。变化后的 Sync-C 只有在固定评分音频时才可归入生成端响应；独立视觉端点另行判定。

root 维护理论与冻结设计；remote_recovery 使用 gpt-6-luna/max 实验和代码并唯一串行管理 GPU；local_recovery 恢复网盘；mechanism_audit 独立审阅。服务器见 [[TTS 机制研究服务器 weste 12349]]。

## 待检验的正向解释

H1：原始波形强度通过 Wav2Lip log-mel 的偏移/截断及 SyncNet MFCC 能量项改变视频与评分，句级电平是可操纵路径，局部动态范围需单独测试。

H2：TTS 在 Wav2Lip audio-encoder 的分布方向可向自然语音移植。保持每帧原自然表示的变化和自然评分音频，仅加 cal-only TTS−N 来源方向，若在独立说话人上向正方向提升、反方向下降、随机同范数方向不复制，则可确认生成器表示路径。再使用独立声学 level 轴的正交分量检验它是否超出电平。

H3：声源（F0/voicing/aperiodicity）和谱包络可分担波形层来源差。WORLD 的自重合成与双向交换必须共用recipient时钟和固定RMS；先量化重合成桥接，再归因交换分量，避免把 WORLD reconstruction 算成 TTS 效应。

H4：剩余 C 优势可能源于错配背景距离和配对相关性的变化。C=median31−min31；固定lag与独立置换用于区分搜索收益、距离尺度和真正时序关联。B/D 的算术分解不称为因果贡献率。

## 现有证据与恢复顺序

9/26 实际报告已从 B:/experiment-archive/sync-20260929 恢复：Wav2Lip/云Qwen74对同PCM原生 +0.935，主差分由 B+0.784、D−0.151组成；自然音频替换下降，原生高分依赖具体配对。报告位置 runs/recovered_20260930/local/reports/tts_pcm_residual_20260926/report.md。

9/27 源码显示来源方向用20条cal/10speaker等权拟合，holdout80/40speaker做speaker-disjoint断言。parent primary 是 global_plus 对baseline和global_minus在raw guard20 C/C_anchor的4个99%CI下界全正；strict_orth 延续测试level轴外分量。当前这是代码中的冻结判据，不是已经恢复的真实数值。

9/28 WORLD 网盘目录有 audio/features，先找producer/protocol和可独立重算的缓存；本地 stats/scoring 仍是草案。

## 当前执行状态
正式input固定于runs/recovered_20260930/local/qwen_source_axis/hf_selected/input_manifest.json：100自然/100Qwen、cal20/10speaker、eval80/40speaker，官方AISHELL revision bbe295d530192a4cd41644b711c9aecd087df653。cal-only d/u和k=3/column18独立重建合格。

双脸主2560行全部封存、服务器6311/6311成员通过；attempt08独立CPU验证2560/2560、缺行0、artifact failures0、N来源primary240必要行与80共同支持组全部合格。summary本地 runs/recovered_20260930/independent_validation/source_axis_consumer_attempt08/independent_consumer_summary.json，SHA55988932b41c0410f10acdb87db43ac30d21315d9ef1a5e06f423462c5403f1c；execution_record SHAfcba6d567fbc5f9b5309283971068b6dc231baf55b23a7dcf76bf9f4e3fa6226，complete于21:53:57Z。root首次读取主效应在该完整核验之后。protocol8ba27fcb…cf39、plan19f805…9dab不变。

完整N80/40、raw guard20四主门eligible=true但conjunction=false：orth+−baseline C=-0.019，99%[-0.048,+0.011]；anchor=-0.019[-0.048,+0.011]；orth+−orth− C=-0.010[-0.040,+0.017]；anchor=-0.013[-0.041,+0.013]。primary没有确认统一来源方向可迁移。full+为预设secondary：C=+0.023[-0.001,+0.045]、anchor=+0.023[+0.002,+0.045]；不得用单个secondary阳性替换四主门。

自然波形句级FIXED RMS对生成端（同RAW N评分）的G路径得到正向响应：primary raw C=+0.192[+0.111,+0.276]、anchor=+0.189[+0.110,+0.272]、D=-0.207[-0.304,-0.110]；secondary face C=+0.135[+0.071,+0.207]。unit geometry仍有primary C=+0.027[+0.016,+0.038]与secondary +0.016[+0.008,+0.025]，故这一生成端响应不完全由SyncNet feature norm尺度解释。评分端E在主脸raw C=-0.016[-0.033,-0.002]，B=+0.241/D=+0.257会互相抵消，不能把B变化直接当唇形改进。

一条TTS S0099_BAC009S0099W0123共同窗口40、guard20空，T四格与native gap intended80/40不完整：39完整speaker的raw主脸原生T−N C=+0.421[+0.254,+0.582]，FIXED双端gap=+0.213[+0.063,+0.360]，均仅支持可测subset。G路径差T−N=-0.214[-0.312,-0.128]，E=+0.008[-0.014,+0.026]，I=-0.002[-0.012,+0.009]；禁止拼不同分母成中介比例。guard15/full/valid原数据已齐，预设敏感性聚合待独立supplement，不升级main。

旧随机存储F32剂量有51/22428帧略超1e-6公差、37clips受影响。独立QA所有80 orth± KKT/nonnegative/finite/inactive合格，随机失败按arm保留，不使同clip primary失效。原specificity仅43clips/10完整speaker，intended80确认不合格；新640行实际F32匹配对照按[[随机对照的实际 float32 剂量复核]]执行，不放宽公差。原08消费者search_gain错误赋为C，B/D/C/anchor独立数值不受此项错误影响；新不可变supplement需正确D0−D及null search诊断，原08字节不改。temporal-null主contrast raw C observed−mean null=-0.018，99%[-0.045,+0.011]，one-sided p=1.000，属于次要诊断。

视觉640/640轨迹有效，按原N-phone/all8轨迹冻结availability有52 eligible clips、16完整paired speakers；全80/40视觉primary支持不完整。16speaker secondary中 frozen-latency separation orth+−baseline=-0.000117，99%[-0.003126,+0.003903]；AUC=-0.00104[-0.06632,+0.05972]。CAL event-rematching未合格，不写时序确认。v2是在旧剥ː/ʰ/ʲ/ʷ/̪规则前增加五tone-bars剥除，最终classifier不是保留所有非tone标记；物理identity仍NFC/strip exact保留IPA符号。

重采样独立80/40四格ΔC=-0.015，99%[-0.043,+0.010]，支持格式往返的正向幅度上界较小。接续[[编码器方向与范数对生成响应的分离]]、[[WORLD 声源谱形与局部能量的双向移植]]和[[女性说话人上的 TTS 来源方向独立复核]]，把已确认的电平生成路径与局部动态、声源状态/谱形进一步区分。新假说注明已读main/sham/visual效果。SyncNet与独立视觉/感知端点分别判断。
## Observations
- [status] implemented；主2560行独立核验完成，四来源主门未通过、电平生成路径已确认；guard15/full和search_gain次要分析supplement待完成。
- [protocol] 固定PCM/face/clock/support；四格G/E/I、八条件cal-only来源方向/level secant、global q10 anchor、40说话人等权99%四项联合门槛见冻结spec。
- [result] cal20/10speaker运行72.98s：||d||=.302893460935、||u||=.079444045363、cos=.576333695076；仅为几何，不称贡献率。
- [verification] 20/20 cal NPZ已恢复并独立重建d/u、q10 PCM/MFCC/掩码与anchor，valid=true；JSON runs/recovered_20260930/independent_validation/calibration_artifact_validation.json。global k=3、column18、official offset=-3。
- [result] 校准T−N原始RMS平均+7.620dB（19/20正向），固定RMS后仍有静音和局部动态范围差；120个gain对照C0增量符合2ln(gain)，shape项近似不变。
- [result] cal-only WORLD20对：TTS全帧voiced比例高.1906、低频AP低约.17、归一SP局部变化低.7605dB；RAW/FIXED差稳定。需活动/voiced/音素条件域复核，描述不单独定因果。calibration_world_20261001/calibration_physical_descriptors.json SHA d03ff74233284e14c485f7358382f1f3c2ec4d65f0b026c617707c956cec177d。
- [verification] 官方与中文pilot PCM身份/direct decode抽样/官方特征/延迟控制通过；独立visual+validator18项pytest子代理执行通过。MediaPipe两条fresh视频QC JSON见 independent_validation/pilot_mediapipe_qc.json。
- [verification] attempt1因CUDA边界zeros构造错误写0评分行；旧run保留。retry2修复后运行，双脸2560预定行manifest SHA 19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab，protocol SHA 8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39；所有效果汇总等待封存。
- [next] 完成eval80所有条件/双脸生成特征封存，独立score/null/门槛；并推进MFA视觉、WORLD own-synthesis桥接及物理source/filter双向移植。
## Relations
- advances [[解释 TTS 的 TFG 与 SyncNet 优势]]
- uses [[TTS 机制研究服务器 weste 12349]]