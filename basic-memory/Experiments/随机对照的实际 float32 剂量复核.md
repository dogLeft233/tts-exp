---
title: 随机对照的实际 float32 剂量复核
type: experiment
permalink: tts-exp/experiments/随机对照的实际-float32-剂量复核
status: implemented
---

# 随机对照的实际 float32 剂量复核

## 问题与证据
原独立consumer07因把random资格扩到整context，拒1184行，只43/80 clips；保留失败且不读取部分cohort效应。08改为各端点只依赖必要臂，完整自然80/40方向主门可核验，随机控制失败独立保留。Root已读完整来源主效应，不再声称“尚未读任何fresh EVAL”。

独立剂量audit rawSHA ed16182dd06279adeeb9a88fd8d17c8ae933bad879991f332077a308e47dc16f：22428活跃random帧中51帧、37clips超1e−6实际F32剂量容差，最大absolute误差1.022595779e−6，超界2.25958e−8。所有80 clips的orth±各7476活跃帧KKT/nonnegative/finite/inactive identity通过；baseline端点与400非random臂/每脸均合格。旧等剂量specificity只有43clips/10完整speaker，不能当全80确认。

代码根因：原ops按float64投影norm搜索，最后才cast F32，metadata actual_norm也来自cast前。规范约束实际decoder F32 displacement，不放宽容差；新source对每候选cast F32后再算实际norm并单独封存。旧male run及ops不改，数值控制错误不当模型阴性。
## 冻结修复与新实验
权威spec openspec/changes/confirm-random-specificity-20261001/spec.md，rawSHA679b11c8a67b18ac397d63b95900d55865b6dbf1ad7e8662bf3a5b2e915c8c79，本地commit5b2e716未push。原main4门只依赖baseline/orth_plus/orth_minus；修正validator的依赖范围，各非random行用自身资格，random逐臂逐帧保留控制失败，禁止整clip删掉。全部原2560计划行seal仍必要，原80/40分母、threshold/axes/k不改。

新实际F32-dose操作版本保留旧ops源：每次候选KKT结果cast F32后从savedarray以F64算norm，与exact原orthplus实际位移匹配；同[0,64]/doubling/64bisection/cap、zero/inactive exact。分别存F64与F32 norm/error、actual constraint cast bound，不放宽容差。Female第一次forward之前采用同修复并绑定新源hash。

新独立aux640行：80male×2faces×[fresh orthplus,3corrected random]，exact原z0/zorthplus/mask/d/u/3directions/seeds。全部四臂与freshRAW N A统一post-detection false，记录actualflags，不与旧未知runtime视频相减。reuse k3/c18、common domain、raw/unit/full/valid/guard20/guard15；完整A/V/z/MP4/cross/zero/lag存档，独立实际dose和cache复算。

## 统计与状态
实际F32操作QA CPU attempt02完成80 contexts/240 random arms/22428活跃帧，0failed/capped/miss；各臂max实际dose误差1.924e-8/1.085e-8/8.628e-9。旧51越界帧与失败源保留，容差仍1e-6。

Aux v2 source SHA792e0522572f5d02f476519d85139849af472348f723bd3c730a090a530a43e7只修写/hash顺序并增加materialize-only；数学/helper/directions/seeds/masks不变。helperc9af12003cd970899cb76d3147201b896352eb8a050389199e6d476f090027f4。CPU-only materialization06 exit0，80contexts/640refs/880arrays、0integrity/dose/cap失败；QA实际64位SHA487bedd831e00012ebbf2524e5897454d57c098322868a35ac09c412948bae2b，inventory4fdd955d6e9b759211035bb030bd474a3799a43ed0233426e1b5ad8e1e0ebd46。前immutable checkpoint有65位抄写错误，实际产物未变。

GPU aux attempt03现已实际完成并由root读取状态：80samples/640scored rows/640visuals+videos+scoreNPZ、0failed、worker exit0。全部640行actual flags cudnn_benchmark=False/cudnn_deterministic=True/deterministic_algorithms=True。权威run /root/tts-research-system-20261001/source_axis/random_specificity_aux_20261001_attempt03/；materialized contexts在RUN/samples/*_latent_context.npz，结果evaluation_results.jsonl。LF-only launchv6 SHA976b65c4fcbe9f31cc0b8bd22b90fab7e9624038507c3d5e4ac174d8b89051ea；此前所有failed launcher/worker原样保留。纯PSF private parent的3py source treea419b997acfd21da8f4c7324e5be22e32357d085e37ac3376165e676c1e67c17，W2np1.26.4不混np2。

完整run完成仍需独立consumer：核source/spec/plan/result/status/file/lineage seals、实际F32剂量、逐forward flags，重建raw/unit四支持域五指标及100 full-A null，再对orthplus−mean3random用80clips/40speakers、20k PCG64/99CI。尚未读取aux效应，不把640完成写成specificity正/负结论。
## Observations
- [decision] 规范约束始终落在decoder实际使用的float32数组；修搜索实现而不放宽冻结容差。
- [decision] 端点只依赖必要臂，secondary随机dose失败不删除otherwise-qualified整个clip；原primary分母与seal不变。
- [result] 修正F32剂量QA和80context材料化完整通过；fresh aux03已80/640完成、0failed、全部640行实际确定性flags通过。
- [next_step] 对aux03封存完整产物做独立剂量/矩阵/统计消费，报告80/40的specificity 99CI与raw/unit/full-A null。
## Relations
- relates_to [[TTS 来源方向与声学机制确认]]
- relates_to [[编码器方向与范数对生成响应的分离]]
- advances [[解释 TTS 的 TFG 与 SyncNet 优势]]
- applies [[TTS 机制确认分离生成端与评分端]]

## 完整实际剂量与独立评分复核

Aux独立consumer04 actual exit0/COMPLETE80clips/40speakers/640score/0failed，源f0d9bfd56be17b34af1f26f6ce175e39ba929b530a8acf1dea380194278ae375。报告`runs/recovered_20260930/independent_validation/aux03_consumer_attempt04/independent_aux_summary.json`120012B/SHAdc73acddc9f745c6c4a8e87ec635a9abc21b1920406ad99a3eb63ec40744867e，root已下载SHA核对与读回。完整allfiles路径/哈希、640 visual、80 context、880 latent array绑定、3450 actual forward旗标、完整V×A与PCM单位均通过；raw最大误差3.815e−6/unit3.576e−7。

独立KKT重建三随机方向的实际float32剂量：每方向7476 active frames均PASS、0miss/0capped、80eligible，max误差1.924e−8/1.085e−8/8.628e−9。此前metadata inventory的0clips是字段读取问题，不能充当剂量结果。producer visual NPZ只存lip，consumer03假定z字段而失败；04将context z_key的SHA与visual JSON/result latent_sha256对齐，核row-keyed decode事件。事件没有latent-array hash，其输入路径由sealed producer源码绑定；不夸大为事件内容自身证明。

预设specificity=orth_plus−mean(random1/2/3)，80意向/40均权、raw/unit四支持集、20k99CI。主脸guard20 C+.004635[−.015626,+.024254]、anchor+.002079[−.019686,+.023034]；unit C+.001133[−.002130,+.004342]、anchor+.000997[−.002355,+.004310]。guard20 100 full-A置换time-excess raw C+.004514[−.015429,+.023923]。此主门槛不满足稳定正向来源特异性。

支持敏感性不能略去：第二脸FULL/VALID raw C+.017428[+.005068,+.030210]、anchor+.017386[+.004848,+.030209]、search+.017519[+.004133,+.030828]；guard15 raw C+.018537[+.002722,+.034765]、anchor+.018582[+.002767,+.034839]；FULL/VALID unit C+.001732[+.000198,+.003356]、anchor+.001719[+.000188,+.003363]。完整主脸与guard20两脸主要CI跨0。现consumer null仅guard20，其余支持域null需新addendum，不能把第二脸观测正值称为已确认时间特异机制，也不能说所有域阴性。

## 全支持域时间置换复核完成

独立补充source `independent_random_specificity_null_domains_20261001.py` SHA85781c20444497c905401a3b1468e78252c86c4354abc0d439096b05c5d8ad0e实际PID328893/exit0，无stderr。root下载并读取report1,883,904B/SHA9afa5f1db74adb232523acee7d60d95e35f58e493deef05be002aa478425109c，位置`runs/recovered_20260930/independent_validation/aux_all_domain_null_attempt01/independent_aux_all_domain_null.json`。7680clip rows留远端，detailSHA714f069607544c8c19f1885b6688da7bae9136d984b5c9ecfebcb4c958d14f5e4。80clips/40speakers、2faces、raw/unit、FULL/VALID/guard15/guard20共96summarycells。640score SHA、完整A、共同V前缀、支持/距离重建合格；继承aux04实际F32/KKT审计。每clip100同full-A PCG64置换，各臂/脸相同，20k等权speaker bootstrap，99%CI。

primary guard20 raw C observed orth_plus−mean(random)=+.004635，time-excess=+.004514[−.015429,+.023923]；anchor excess+.002791[−.019374,+.024251]。unit C+.001024[−.002065,+.004055]、anchor+.001031[−.002322,+.004337]。主判定未达到来源方向相对随机的联合门槛。

secondary FULL/VALID raw C specificity observed+.017428，time-excess+.018689[+.006227,+.031422]；anchor excess CI[+.005020,+.030000]。guard15 raw C observed+.018537、excess+.018252[+.002620,+.034386]，anchor CI[+.002722,+.034640]。secondary FULL unit C excess+.001806[+.000301,+.003439]，anchor CI[+.000198,+.003281]。guard20及unit guard15/20 CI跨0。

结论：小幅第二张脸、FULL/guard15的来源方向对随机对照相对效应包含正确时间配对成分；正式primary guard20仍未通过。这不能单凭relative specificity称orth_plus提升了未干预baseline，最终报告同时给baseline/orth_minus和主四门槛。来源轴可作为有限条件响应，不是本项目稳定音量路径的替代。旧general消费者search用C的记录保留；本补充按D0−D明确修正并密封。

- [result] 第二张脸FULL/guard15 raw及FULL unit的来源方向随机相对time-excess为正；primary guard20 CI跨0。
- [verification] root实际下载SHA/读summary完整，无以exitcode或代理摘要替代科学复核。