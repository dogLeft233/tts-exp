---
title: proposal-validation-p0-p1-p2-20261016
type: note
permalink: tts-exp/codex/tts-exp/proposal-validation-p0-p1-p2-20261016
status: in-progress
project: tts-exp
repository: dogLeft233/tts-exp
priority: high
current_step: 3
phase: amplitude_complete_p2_parked
goal_tool_status: paused
owner: ROOT
start_date: '2026-10-16'
tags:
- proposal
- amplitude
- pause
- validation
---

# proposal-validation-p0-p1-p2-20261016

## Current completion

幅度六环已完成并交付 [[46-amplitude-gain-independent-validation]]，本轮无待运行GPU单元。20人/160视频/160评分/160口部条件全部完成，0删样/缺失；FIXED主+0.049、95%[−0.005,+0.106]、14正6负，预定+3dB+0.067、四剂量同时95%[+0.013,+0.121]、15正5负。全部四剂量和full敏感性按冻结协议保留，不重选最佳点/主支持。口部开度与速度可复制地改变，评分不是准确性/感知指标。旧40/15数值复核与新20独立结果分开。

canonical分析`runs/p1_gain_dose_response_20261016_attempt02_readback/final/analysis_results_attempt02/`，summary SHA7b9ef0a1…；joined input5567f039…。原score29865f15…不变。full截断错误以160数据-only数组/384尾部cell修正，primary逐位不变；首join副本错误导致mouth未进入attempt01分析，attempt02已修复160行并重放数据分析，primary/dose exact相同。原错误文件及addenda保留，不重模型。ROOT最终实际QA52a62bf9…/30f974e1…/8ef77f6e…通过全部主与full矩阵、视频哈希、口部公式、CI与28组件。四图已目视，报告/地图/索引/同一proposal E16已更新。

本Task仍in-progress只因原P2停顿独立待办；幅度子目标完成，不等于原P0/P1/P2总体完成。原goal工具paused/旧objective保持。P2原M/8段标注未完整冻结，不从F推回，不自动恢复head/训练/扩模型；下一研究安排以该明确边界为起点。

## Current scope and acceptance

最新授权优先完成幅度六环：纯x*g波形资格→固定原自然评价音的生成响应→Wav2Lip内部路径→完整距离曲线解释→电平对原生TTS分差的影响→新增多档增益响应。旧P2停顿保留独立待办，本轮不新增MFA/训练/模型或语言扫描。完成以有效检验与实测交付为准，不要求阳性、最优增益或感知提升。

P1仍20新speaker×1句×RAW/FIXED×2次=80视频；另同20人的四非零统一相对dB档×1次=80视频，RAW0复用P1 repeat1，FIXED单列实际dB。本轮160计划生成，已合格单元计入，不承受则交完整较小先导并标覆盖。峰值/静音/来源资格与统一安全档位、新主端点/曲线多重比较在新分数前冻结，不逐句限幅或挑最高分。

内部机制复用已接受40人/80句MEL interior-boundary与真实RAW→FIXED direction-radius；rms_consumer07才是实际RMS路径，source_axis_consumer08不得替代。15人原生TTS电平四格从9/27两份实际原报告/数组复核，FIXED和配对中点LEVEL分开；三模型同机分解与40人队列分开，旧148eval视频有意不保存不得称现存。C/B/Dmin/D0/baseline固定anchor/Canchor/bestlag、开度/速度分别报告，不称中介质量百分比或感知收益。

交付正式幅度报告、原生分差控制图、内部机制图、完整距离曲线图与新增剂量响应图、逐样本/说话人结果与源/失败成本；更新BM地图后精准回填同一proposal。当前proposal SHA d968dd7f6b9ed9371f43c8c24d6d9ed609721840c4e33b459729a3d77945fdb7，必须保留3.1/附录E新增跨TFG/TTS/数据集内容，不能用旧入口覆盖。

已有无预算goal保留原P0/P1/P2 objective，get_goal实际paused（上次中断后）；工具不能改objective或resume。按最新授权继续兼容工作，不虚报旧目标完成或建重复goal。ROOT理论/冻结/BM/一次关键QA，原GPT-6 Luna/max代码实验；必要tests已授权。

## Frozen execution and actual evidence

冻结方案 [[幅度六环与新说话人增益曲线 SPEC 2026-10-16]] raw SHA `fb84fd8d0622c189130cb0a0867fc6f32957e30b3ae503ad2bd6fca2f3bc069b`。20新speaker为S0102–S0111及S0144–S0153，10M/10F；原发布者核验AISHELL包只读复用，0下载。source manifest raw SHA `f3cc98a1598a22f0bd84af77d3e0bf802c5ad62ef3c76c5461e091e46b1535f7`；其内部canonical fingerprint另为9e44024f…，两者不能混用。ROOT实际20原WAV QA036e3fa0…通过，统一−6/−3/0/+3/+6dB全部无削波；FIXED按句实际dB单列。

唯一新执行根 `/dev/shm/p1_gain_dose_response_20261016_attempt02`，packet raw SHA `cc3b293f1d04c1fb343a88bf3fd3aace1283ab3a20ddeca2e4c074b604a1239e`。首S0102计划内8视频/8score已技术通过，primary101q/full113q/31lag/encoder绑定完整；120唯一音频已全部实际encoder前向。首8口部观察100%共同支持，145frames/144edges。当前原代理继续余152render→score，不按分数决定扩跑、不重复launch。最小readback `runs/p1_gain_dose_response_20261016_attempt02_readback/smoke/`，receipt9d0ba7e2…；首score81b46de9…为PARTIAL原件，非全量终局。完整源/身份见最新同chat checkpoint与实际receipt。

attempt01缺log parent在0child时失败；attempt02第一次encoder因远端缺既有helper在0forward时失败，三exact模块窄恢复后只重试encoder。两失败保持原件；不重render/no新安装。代理目标tests17/17、mouth8/8通过。ROOT下一只做一次saved scalar/原score数组QA，余152并行推进，不新增审批框架。

ROOT实际新120音频/首8score QA已完成，`root_new20_smoke_actual_qa.json` SHA `a1ef9d5059ad4319f068fb95eec4d722aa9a6ab10409ace14ac1fe4787e38635`：20原PCM+100保存scalar逐样本exact float32，FFT归一化形状/RMS/clock/无削波合格；guard20独立距离重建最大3.815e−6、全部curve/metrics误差0。仅full敏感性发现engine先截断A到min(A,V)再padding，首8每条3个尾部cell替代了真实A；原full保留，review代理已获授权从saved完整A/V做create-only数据修正overlay/derived analysis JSON，0模型forward/不改q/primary/原score。该修复不阻塞当前render，不再走新资格循环。源文本含替换符的console疑点经actual codepoint/原transcript核对为显示问题，0U+FFFD/20文本exact，无需metadata overlay。

两旧分支已实际复核关闭：40人2560RMS/2560MEL/320motion行QA401a37a0…零误差，编码direction C+0.161（radius+0.020、joint+0.190）、mel interior+0.171/boundary−0.009；15人71合格句原曲线QAc30304c3…零误差，原生gap+0.438→FIXEDq11+0.254，G-only gap变化−0.101，LEVEL仍单列。controls360既有QA只复用；新图native_gap_control_attempt02已ROOT目视通过。旧证据不称新独立复现。

用户基线/覆盖问题已澄清：报告45的5/24（20.8%）可执行配对来自2人、完整来源0/4；+1.770～+5.495是固定视频local评价音轨相对affine异实例音轨的恢复，不是超过自然自驱动/自评价基线。本轮幅度主比较才是`S(G(gN),N)−S(G(N),N)`，重复先按speaker平均；必须报告完整20人受益/下降/未完成分母及重复波动，不能只报阳性子集。

## Original objective (superseded schedule)

完成最新版中文proposal第5节及统计折叠附录的P0/P1/P2，交付实际数据、图2–4、正式报告，并将结果回填原proposal。完成指实验和验证完成，不要求阳性。用户在proposal会话明确要求交给本会话持续完成；本会话已创建无token预算的active goal。跨TFG增强头训练、广泛模型/语言扫描仍暂停，报告45评价音频时钟恢复不能称生成增强。

权威方案：`basic-memory/Research/tts-tfg-resynthesis-proposal-zh.md`，本次入口SHA256 `95be37e11eb2421c47b5f1a56f3ecc2cf15fdfce9ebf6ff8d9135a8beeda7aee`。ROOT理论/冻结协议/Basic Memory/一次关键实际QA；沿原GPT-6 Luna/max代理执行具体代码、恢复、实验与必要测试。不得逐阶段重复索要确认。

## Original P0/P1/P2 schedule (current scope above takes priority)

1. P0恢复已有幅度与MFA证据，核验输入/模型/参数/产物身份，复用已接受QA；仅按图2–4指定端点进行数值核对，0新生成/评分，不把复核当新独立证据。幅度consumer与301归档包已定位；报告33目录现存、36预期目录仍缺，继续定向查找。
2. P1冻结独立于幅度规则调参及既有机制分析的20位新speaker、每人1句；原音/RMS各2次实际独立生成，80计划视频。目标RMS0.0376838172675746、同脸/冻结Wav2Lip/原自然PCM/共同支持/原裁剪；先音频静音、削波、长度与身份桥接。每句平均重复后按speaker配对统计95%区间，记录C/D/固定lag、开度/速度与重复波动。尚未生成或评分。
3. P2恢复6已见样本a1_070/072/076/082/094/096及8段原音频标注（这6句来自4speaker）；原M、仅正确停顿P、等时长发声对照C同批18视频。20ms边缘、固定seed避真停顿；非目标采样精确一致/等长、记录RMS但不重新归一化，同原自然PCM。F不是P，缺原M不可从F反推。尚未冻结具体资产或生成。
4. 整理全部计划分母、失败/排除/支持不足/预定比较、实际生成及重试成本；关键图表标baseline、n和区间；正式报告→BM地图→proposal实际回填→Task done。保持中文短句、Abstract/Method/Experiments模板、流程图和折叠附录。

## Original execution boundaries (current 160 amplitude budget above takes priority)

P1/P2共98个计划生成单元；用计划内单元工程smoke计入全量，工程失败重试另计，不无限重复确定性失败。不按分数选样、调gain/阈值/支持、扩大队列或模型。来源不独立只称重复验证；P1独立复现限波形层，不称内部四格机制也独立。评分/运动不等于感知，负结果收窄主张。资产恢复阻塞不依赖分支继续；缺失不作科学阴性。

LAN wjj@192.168.50.2:22、公钥SSH与现有qualified环境；先查真实进程、容量，不重复launch未知进程。科学原件只读、derived create-only。秘密不文件/argv/BM/log；不动CloudDrive配置/DB/进程、sudoers或旧拒绝删除。源/输出只做本轮必要恢复与清单，不重做旧巨大资格审计。

## Observations

- [status] in-progress
- [decision] 幅度六环优先，160计划生成；P2独立待办/旧head暂停，完成不以阳性为条件。
- [result] 幅度六环160/160全部完成、实际QA/正式报告46/四图/proposal E16回填完成；FIXED14/20正但主CI跨零，预定+3dB15/20正且同时CI为正。
- [next_step] 复用报告46完成论文主张收敛；P2保持独立待办，当前不启动新的GPU实验或恢复旧head。

## Relations

- implements [[从 TTS 高分到可控驱动：唇形生成的输入机制与迁移边界]]
- relates_to [[研究交接总览]]
- extends [[25-tts-tfg-mechanism]]
- extends [[33-mfa-linear-content-pause-prosody-mapping]]
- extends [[36-mfa-linear-heldout-speech-residual]]
- implements [[幅度六环与新说话人增益曲线 SPEC 2026-10-16]]
- delivered [[46-amplitude-gain-independent-validation]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 建立已授权P0/P1/P2持续执行任务与成本边界 | October 16, 2026 | user（proposal会话授权，经Codex转交）；ROOT |
| 最新范围改为幅度六环优先、160计划生成、P2停放，保留旧goal及proposal新3.1 | October 16, 2026 | user（proposal会话授权，经Codex转交）；ROOT |
| 冻结新20人与剂量；首8技术通过、余152继续；澄清报告45比较对象与覆盖 | October 16, 2026 | ROOT（沿既有用户授权） |
| 幅度六环完成、完整分母/统计/实际QA/图/proposal已交付；P2单独待办 | October 16, 2026 | ROOT（沿既有用户授权） |