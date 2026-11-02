---
title: SyncTalk 四新身份官方完整训练电平实验 Spec
type: note
permalink: tts-exp/experiments/sync-talk-四新身份官方完整训练电平实验-spec
status: planned
tags:
- synctalk
- multi-speaker
- spec
- gain
---

# SyncTalk 四新身份官方完整训练电平实验 Spec

2026-10-17，IMPLEMENTED / EXECUTION_FAILED（四身份统一准入通过；Brown/seed42/coarse ep5/+6非有限loss门停止，部分epoch无完整state；尚无完整模型/27格，不是科学阴性）。用户授权保存spec、准备4个新说话人每人6–8分钟自然素材、保留同人不同录制测试，先核验，通过后才执行官方完整训练的−6/0/+6 dB配对，并列完整27条目。没有用数据资格尚未通过的素材启动训练。

2026-10-18诊断补充：用户授权保持FP16核查适配与官方环境。本轮旧环境独立复现NaN，scaler2^-128反缩放溢出和原density exp溢出有双环境直接算子证据；原日志同阈值末步支持高置信度终末归因，原失败state和上游因果边界仍保留。详见 [[SyncTalk FP16 非有限训练失败的适配与官方环境核查]] 与 [[核查 SyncTalk FP16 适配与官方环境]]。本spec原FP16/defaultGradScaler政策、失败状态、数据准入和24模型/27格目标不因此改写；独立FP16更新前保护已另立并资格，见 [[SyncTalk FP16 更新前保护资格与密度溢出因果核查]] / [[044-synctalk-fp16-preupdate-guard-revision]]；上游稳定性尚未合格，未正式重训。

## 目的与实验单位

检验报告50的训练电平响应能否在新身份、完整官方训练与同人不同录制测试中重现，并用评价音量轴区分固定视频的评分器响应。沿用英语与AVE，避免同时更换语言或版本。每个人单独训练身份模型；本轮不是训练A后直接生成新B的人脸，也没有额外加入跨声音测试矩阵。四身份为统计单位，两个seed为身份内重复，片段/帧不是新增独立样本。

“新身份”指未参与这批SyncTalk电平身份训练且非May；项目早期音频实验和AVE/SyncNet上游暴露另做核验，不能把身份模型训练留出称无上游污染。

## 数据合同

每人360–480秒不同自然素材，目标420秒；总24–32分钟。优先采用HDTF官方元数据可追溯的自然讲话视频，镜像只作为传输，不冒称官方训练数据相同。源URL、原视频ID、原时间区间、镜像revision/ZIP CRC、下载SHA、许可和改动全部入manifest。

目标监督：train6000帧/240秒、val750帧/30秒、test1500帧/60秒，统一25fps/512×512。剩余素材用于前端上下文与预留，不声称所有原料均进入训练。train/val可同场录制但连续时间留出并保留2秒支持间隔；每段首尾至少1秒仅为音频上下文。test必须来自训练/验证以外的同一个人的另一场原录制。不能仅凭不同文件名或不同上传ID判定独立；须核对官方来源、视频/音频重复、画面和语境。

冻结六个10秒测试clip，不根据后续分数更换。检查真实自然监督、同身份、单人清晰嘴部、帧率/PTS/音轨时钟、静音/峰值/削波、支持不重叠、素材SHA。测试录制仅提供测试姿态/眼部/背景；训练身份形状、背景与相机校准不使用测试监督，跨录制坐标相容性必须验收。独立录制造成的衣服/背景/光照域差异需记录，不当成纯声音因果效应。

## 官方完整训练合同

固定SyncTalk commit43079fb3f349675f6cb33031bb07262f7566cdd5，依据官方scripts/train_may.sh，而非论文另一个100000/25000预算。官方源码网络/loss保持，允许独立split/provider绑定和控制采样/日志的外层适配，改动逐项封存。

每身份seed42、2026；每个identity/seed生成一次完整随机NeRF/条件模块state，三增益strict加载相同state，预训练AVE固定eval，禁止May生成器warm start。共4×2×3=24个完整训练。三增益真实重提AVE，不乘既有缓存；原音频按recording独立保留上下文，float64增益，无归一化/limiter/时移。

第一阶段-O/FP16、65536 rays、max_steps16、--iters60000；第二阶段从各自第一阶段最终完整checkpoint恢复，--iters100000 --finetune_lips --patch_size64，保持官方交替lip与patch、LPIPS、优化器/调度器/scaler/EMA行为。官方按完整epoch运行并恢复epoch/global_step，所以100000不是额外100000步；若6000训练帧，则第一阶段10epoch=60000 attempts，第二阶段到17epoch=102000累计attempts，新增42000。最终data长度确定后在训练前冻结实际预算，不能沿用旧“新增100000微调步”的模糊说法。

同一配对块的帧序、实际ray/patch/rect抽样、occupancy/perturb随机域对齐，验证不改变训练RNG。记录每步实际optimizer update和AMP skip；相同global_step不等于相同更新预算。若跳步导致预算不等，该配对块控制失败，禁止静默改变一档时长、ray数或precision补救。资源无法承受65536时阻断，不以旧4096先导冒充完整训练。

官方new-video预处理依赖face parsing、landmarks、Basel 3DMM、姿态细化与眼部条件；不以未经资格的替代跟踪器冒充官方预处理。官方train_may.sh不包含可选torso训练，本轮采用head+lip+portrait，torso不追加。

## 验证、生成与评分

记录两阶段学习曲线和固定验证支持；固定最终EMA，不按test选checkpoint。每个模型分别以三档测试音频驱动，仅生成九种视频条件；评价三档复用视频和视觉特征。主0 dB评分严格原始PCM/原时钟，FFV1/PCM正式媒体，H264/AAC只观看。

±6评价采用float64原PCM数值单位乘增益后做MFCC，并保存FLOAT64 WAV及明确单位转换，避免新素材峰值越界时削波。0评价仍原PCM16；不得把FLOAT WAV直接cast int16。此项与报告51的PCM16舍入处理不同，明确标记，不混池；有余量时可单列PCM16量化敏感性，不替换主27格。

SyncNet V2严格原权重、FP32、共同source-owned ROI、±15lag，guard20主/full次，保存C/Dmin/D0/B/offset与原距离矩阵。每clip先算，再seed内六clip平均、身份内两seed平均、四身份等权平均。独立重建全部距离/端点，0评价复现原评分、音频编码重复、原视频/特征SHA不变。另保留同身份测试的口部重建误差；Sync-C高分不作感知更准结论。

## 留出预处理与内存实施合同
2026-10-17，在首次新人物身份/焦距/全量光流/BA拟合和训练前冻结scripts/experiments/synctalk_multispeaker_full_20261017/preprocessing_contract_v1.json，见[[038-synctalk-training-owned-geometry-and-lazy-loading]]。官方训练网络、loss、FP16、65536 rays、两阶段epochs预算不变；官方preload0按需读取替代preload1内存预加载，须共同子集像素/ray输入逐值等价，不能直接假定。

训练6000帧运行原版BFM tracker和原版500+8000步联合BA。验证/测试固定训练身份及焦距，以官方landmark目标/调度只拟合表达和姿态；heldout BA先500步固定相机拟合点、再冻结点进行8000步姿态优化。只从训练BFM/BA相机配对估计正相似变换，把heldout pose映射到训练坐标；训练残差、尺度/条件数、测试投影恒等式必须通过。这个heldout conditioning约束是防泄漏实现适配，不称官方默认独立自由重拟合。阈值见机器合同，不按SyncNet结果改变。

光流mask来自原版portrait mask的255-G alpha，三通道相同；旧stage1 face_mask彩色副本弃用，不能交给光流。原NeuralNRT权重/crop/12iters和原Laplacian选点块保留，只优化存储：前2500帧刚性像素与全段关键点轨迹，稠密子集逐值校验。训练背景只用train帧拟合；所有监督合成共享这个背景，heldout原图可作其portrait条件，不参与训练拟合。

## 共坐标实施修订 v2
训练前几何资格发现v1单一BFM→BA相似变换假设失败，Brown旋转p95 6.675°/平移14.896mm超过2°/5mm；旧合同和失败原数保留，不放宽后假称通过。按[[039-synctalk-shared-training-flow-geometry]]修订heldout实现，冻结preprocessing_contract_v2.json（SHA37f0714cecd41cca0f531bb062b6029bad2e96fe5bc19bd3ad0827d61f3dd005）。

训练原BFM/原联合BA/最终相机不改；以其相机和训练flow三角化固定训练参考帧2关键点，heldout使用同一训练参考图与原NeuralNRT匹配同一组点。训练3D点冻结不更新，heldout仅用原flow loss/Adam1e-4/8000步估计姿态，直接处于训练BA坐标；不再拟合heldout点/形状或最终全局映射。训练内固定两个1200帧控制恢复原BA相机，仍须p95≤2°/5mm；landmark5/10px、flow2/5px、条件数1e6保持。v1初始相似变换仅供初值，残差继续报告。跨录制方法适配明确记录，不能称原版默认heldout实现。

Brown训练内控制和val/test各20帧诊断通过，未使用TFG处理效应/SyncNet排序，也未删帧；抽样不代替全部750/1500帧资格。数据门仍要求四人全量通过后训练。预算、初始化、24模型、三档driver/三档scorer与27格未改变。

## 完整epoch执行与恢复合同
2026-10-17，在任何本轮实验TFG optimizer更新前固定execution_contract_v1.json（SHA135b1001717033fed41bf0e77ddecf8520ccf3b74e88b945af238103badd96c7）。完整6000帧/epoch，coarse1–10、fine11–17，尝试预算60000/42000。原Trainer.train_one_epoch与loss/AdamW/scaler/EMA数学不变；EMA.95每1000 attempted steps，原默认bs_area=upper七维，fine lip/patch按原代码交替（lip矩形rays不人为填满65536，patch/粗仍65536）。

同一identity/seed三档逐完整epoch轮换；同阶段重启恢复完整model/optimizer/scheduler/scaler/EMA及renderer非注册iter_density/local_step，验证后stats/EMA collected也恢复。进入fine仍按原版fresh model/Trainer载自己的coarse完整checkpoint，非注册counter reset不跨阶段恢复，首更新使用coarse当前LR，随后原fine lambda。共同逐步frame/ray实际采样SHA验收，真实AdamW post-hook计数区别AMP skips；阶段末实际更新数不等则CONTROL_FAILED，不补步、不改默认scaler政策。

每epoch固定全部750val×三driver，0dB主曲线；原eval_step/EMA/FP16，验证恢复RNG且不改变optimizer/scheduler/model/EMA shadow。仅压缩日志为MSE与三个固定0dB预览。完整checkpoint每epochF逐字节验证后滚动，只退役已被同臂新完整state消费的中间cache，保留各coarse-final/fine-final与显式finalEMA；全部trace/curve/receipt保留。finalEMA在应用EMA后深拷贝完整tensor再restore，避免原best writer别名保存raw。

实际20项Ubuntu runtime单元控制、Windows锁/408模拟epoch恢复两项及跨Windows/Ubuntu持久轮换attempt02控制通过，但不是24模型完成。Brown18个零更新资源probe中原scale65536的lip梯度非有限如实保留；未来AMP skip实际预算仍待阶段末独立验收。四人完整data_qualification才允许新TFG更新。

正式输出严格原test_step、blend_with_mask_cuda、uint8截断，FFV1/原PCM16保真读回；原头部浮点仅留固定帧诊断，不用有损观看版评分。SyncNet实际输入MFCC cast FP32须精确且音频forward逐值相同；float64 MFCC因BLAS末位重复差单列记录。独立float64 NumPy重建原距离（含pairwise_distance eps1e-6）、全部lag与端点，阈值3e-5为计算实现校验，实际四人72个GT矩阵误差<4.151e-6、端点<2.377e-6，offset一致；不能把这称为本轮生成视频结果。

## 录制配对资格失败后的数据修订
2026-10-17，在0本轮TFG更新/0生成视频时按[[040-synctalk-recording-role-fallback]]冻结单次回退规则：已经通过的原录制配对保持；仅未通过冻结预处理门者可反转两原录制的main/test角色一次，保持原四身份、420秒、6000/750/1500和固定六10秒位置。全部原数与失败另存，独立版本重新准备并验资格；若反转仍失败则保留两次失败，不继续扫描片段/按SyncNet选优。这个数据适配须在最终报告披露，结论只针对通过几何资格的保留配对，不能声称随机录制总体效应。

Debbie原main录制0/test录制1的全量v2测试flow median4.48346127047664/p956.69030642173168px失败，阈值2/5不改。F geometry_diagnostics/DebbieDingell_failed_v2_attempt01/full_failure_bundle/failure.tar.gz共61成员/29944792原字节独立成员SHA通过，归档SHA796861f6aaedeabcf168c9d41a262cd22857093b7b2e7a86e47d657518f72734；archive verified不等于geometry qualified。原训练控制/val通过及独立投影复现同时保留，具体原因未定。

候选版本DebbieDingell_recording_roles_v2：源录制1原片290秒作main290，源录制0作test130，合计420。两source下载SHA未变；源录制1ffprobe实际290秒。F dataset_revisions/DebbieDingell_recording_roles_v2/revision_contract.json与selection.json在候选准备前固定，ROOT独立dataset_revisions目录，不覆盖当前失败材料/预处理。新test需要重建source-owned ROI、三评价音频forward与独立评分参考QA；旧测试分数不能继承。当前只是候选，不是训练准入。

原训练初始化、24模型、官方原FP16/loss/optimizer/scaler/EMA、65536rays、完整60000/102000attempts、实际成功更新预算和27格不变。Cathy原配对独立worker继续，尚未得到其全量几何结果。原主stage2/训练及媒体监督器已经因Debbie失败停止，不能把旧排队状态称仍在自动开训。

## 长度不支持反转时的唯一替代测试录制
按[[041-synctalk-eligible-recording-fallback]]，Cathy原配对全量v2测试flow median2.372264813512586/p953.936536358146548px，因median>2失败；旧60文件失败包F独立验证SHA624a46a0f1135e819b7238f1c8282f4f004bdea40b37b8d7b481de475227b9e5。原test录制0实长243秒，达不到main固定train/val/上下文至少278秒，不能强行反转。

唯一下一候选按此前已下载同官方身份标签原录制编号选2（21mNN49Pp34，官方标注00:05–01:55，110秒），保留原main录制1（yDi_pmbanwc，295秒）。实际405秒/6分45秒满足原360–480秒合同，不称420。6000/750/1500、六固定10秒、全量几何阈值、24模型与完整训练预算全不变。新test全部重做；原主录制训练/val状态只有在全RGB/PCM/来源与对应JPEG/FAN/parser/BFM/BA/flow SHA相同后才可复用，仍重跑全部冻结v2恢复/几何、loader/rays/资源与新评分参考QA，不利用测试更新训练几何。

这是旧资格失败后的唯一数据候选，需披露失败与243秒排除，不按SyncNet/TFG效果挑选，不追称最初预注册；若新候选仍失败，保留而不扫描片段。F dataset_revisions/CathyMcMorrisRodgers_recording2_v2/selection.json与revision_contract.json在准备前保存。四身份最终版本必须独立统一准入，旧false worker或旧test参考不能沿用为新通过。

## 用户授权的固定元数据顺序身份替换
2026-10-17，在0新TFG更新/视频时，用户明确要求按官方元数据固定顺序替换说话人，保留失败与原门，四人全资格后自动完整训练和27格；详见[[042-synctalk-metadata-ordered-identity-replacement]]。

Debbie原pair及唯一反转均失败；反转val750 flow median2.0101095402098133>2、p953.938487529220711≤5仍不放行，test shared-reference flow未执行。原pair及48108成员反转失败F完整存档保留，archive SHA da2695fcabf7e6408530759b8be24427a3e3bd5f309d2e79834d64db133e4f45；只回收已持久验证的重复cache，不放宽门、不第三次反转。此前“保持原四身份”条款仅对旧尝试适用，当前由本条用户授权修订为保持四个合格新身份。

身份替换policy_v1 SHA 180603ad4e9d2a5d92edfb6e74a68fb824d1a10c9fe7746e966f1f7440391efd：固定HDTF元数据WDA→WRA→RD文件次序、身份首次出现顺序、原录制出现顺序，不按下载速度/分数/字母另排。排除保留Brown/Patrick/Cathy、已失败Debbie和May。只取原录制首连续区间/镜像首_000 chunk，不拼断续段/扫后chunk。首main连续≥278秒、取最多300秒，首独立test≥64秒且总素材≥360秒，test取min(420−main秒,该连续段秒)，原6000/750/1500与六10秒位置不变。完整数据资格失败者全保留后按封存队列下一身份继续，首全资格者保留，禁止用SyncNet或TFG效果选条件。

首metadata候选BenCardin0(hSvTbV4wW8A)/1(OTg7ngyHFWw)，官方annotation325/284秒，计划main300/test120=420秒；尚待真实下载CRC/SHA/ffprobe与自然素材/独立录制/全几何、整录制AVE、ROI三评分参考、独立QA、持久及90loader/18资源门。Brown/Patrick保持原合格版本，Cathy405秒唯一新test候选继续，不能将未完stage1称数据准备完。全四人实际fresh准入前仍禁止训练，不减为3人。

最终通过cohort的身份、原录制来源、数据版本、全部排除/失败和替换过程须明示；n=4是资格条件化探索，不声称随机总体。24模型、两seed、FP16/65536rays、完整60000/102000预算和实际成功更新控制、原loss/optimizer/scaler/EMA、三driver×三scorer完整27格、原PCM评分与独立QA合同全部保持。统一所有训练/渲染/评分/封存/汇总身份与输入路径后，fresh四人准入通过即自动执行，无需再次请求一般训练许可。

## 固定传输目录资格与当前待核验身份
固定镜像revision的完整ZIP目录SHA5fdcd80d087214ec385c3817d2965ee7a448ed77f29c71d1496c32adae9f268f实际核验：首metadata候选BenCardin缺WDA_BenCardin0_000，第二CarolynMaloney缺WDA_CarolynMaloney2_000，均记录SOURCE_UNAVAILABLE_IN_FIXED_MIRROR（未预处理/未几何/未TFG），原metadata候选和acquisition attempt01失败保留；不是科学阴性。按原封存队列继续第三ChrisVanHollen，两个首chunk齐备；没有因分数重排、换角色或另找后chunk。

Chris源0/1 CRC1a2faaba/479e55ed，SHA18aa6f842e8b2e42474115a0ea43c2a072599f9b782977d35e996945668642a7 / 30b00b54bdfbe73aaa81b474b6bf0cdc30e72b97bf9771ad94308e6d966c36b0均通过。实际ffprobe328/293秒，计划首300/120秒=420秒，25fps；原发布YouTube/Senate Democrats标题与dateText分别2019/06/24、2018/01/19已直接读取并保存原HTML SHA（这是发布日期，不冒称拍摄日）。十抽样帧单人清晰自然讲话、稳定同木背景/相似衣服、灯光和取景不同，身份标签来自官方metadata；相似布景不当成独立性充分证据，后续全量PCM/RGB重复、语境/自然监督及全部资格门仍必须通过。镜像大range传输曾断线，attempt02错误保留；attempt03仅改4MiB分段与有界重试，原ZIP CRC/源SHA不变。

独立版本F/ROOT dataset_revisions/ChrisVanHollen_metadata_order_v1在预处理前封存selection SHA86050b50da540ca1d32a438ce55f81e1d649c0882525a0f0276ce7c84bb62b4c、revision_contract SHAade2b500c2c254800ebba115a33b62996a83f84db7315d8dd92ef37fc4c59541。第四待核验身份由Chris替代Debbie；原两方向失败和全部源保留。当前候选四人为Brown/Patrick/Chris/Cathy，只有全部完整资格通过后才写统一准入并自动24模型和27格；尚未预处理Chris或训练。Cathy live六模块/远端冻结副本不可改，Chris独立helper版本化，最终全部执行ID/输入路径/初始化/ROI/原PCM/汇总/source pins一致后才能开训。

## 预先冻结的描述性分析与平台筛查
在0本轮TFG更新/视频时冻结analysis_contract_v1.json（SHA9ede60017cb5bfe5cbdfc66c47b98115d4e8081e00d64905440c2e64cf75adc2）。四身份为独立实验单位，seed先在身份内平均，六clip先在seed内平均；完整保留两seed与四身份差值、均值/范围/正负方向，不以clip/seed扩大n或做总体显著性主张。主guard20、full敏感性，C/B/Dmin/D0/offset分开解释。

预定训练−6/+6相对0在driver0/scorer0的差，driver−6/+6相对0在train0/scorer0的差，以及固定train0/driver0视频上scorer−6/+6相对0的差。对每训练档与±6，同档驱动/评价总差精确分解为原音频评分下driver差加改变视频上的scorer差；另列固定基线视频scorer项与交互，不给统一百分比。

辅助平台筛查仅用固定0dB driver验证MSE：粗阶段8/9/10均值相对5/6/7、fine15/16/17均值相对12/13/14，绝对相对变化≤2%为描述性平台；保留全部三driver验证曲线和原train loss。它不触发早停、额外训练、test checkpoint选择或改预算，不证明最优收敛/六分钟数据充分。所有结果仍须实际运行后独立核验，当前分析代码和控制通过不当成科学结果。

## 27条目输出模板

以下模板用于每身份/seed和四身份汇总；当前没有分数，空缺不填0。另保留每身份/seed与每clip原数组。

| 条目 | 训练dB | 驱动dB | 评价dB | Sync-C | Dmin | D0 | B | offset |
|---:|---:|---:|---:|---|---|---|---|---|
| 1 | -6 | -6 | -6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 2 | -6 | -6 | +0 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 3 | -6 | -6 | +6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 4 | -6 | +0 | -6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 5 | -6 | +0 | +0 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 6 | -6 | +0 | +6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 7 | -6 | +6 | -6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 8 | -6 | +6 | +0 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 9 | -6 | +6 | +6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 10 | +0 | -6 | -6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 11 | +0 | -6 | +0 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 12 | +0 | -6 | +6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 13 | +0 | +0 | -6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 14 | +0 | +0 | +0 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 15 | +0 | +0 | +6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 16 | +0 | +6 | -6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 17 | +0 | +6 | +0 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 18 | +0 | +6 | +6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 19 | +6 | -6 | -6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 20 | +6 | -6 | +0 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 21 | +6 | -6 | +6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 22 | +6 | +0 | -6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 23 | +6 | +0 | +0 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 24 | +6 | +0 | +6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 25 | +6 | +6 | -6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 26 | +6 | +6 | +0 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| 27 | +6 | +6 | +6 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |

## 数据资格门与存储

四人原料/独立录制与exact split、来源/暴露、音画/脸部/增益、官方预处理资产和跨录制坐标、完整预算共同初始化/更新计数、持久空间备份全部通过后才允许训练。任何一项缺失都不称数据准备完成。用户已授权“核验后再跑”，门通过后无需重复请求一般性训练许可。

机器协议在scripts/experiments/synctalk_multispeaker_full_20261017/protocol.json；持久数据F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017。Windows E约2GB、F约24GB、Ubuntu root约545MB，新增大文件先F，按身份分批处理并及时SHA同步，SHM仅工作缓存。需要清理时仅处理已核验备份的自有缓存，不删除历史唯一副本、不改共享服务；百度盘只有独立读回通过才记备份成功。

## Observations
- [status] IMPLEMENTED：设计和控制器已保存，部分数据/runtime资格通过；全四人准入、训练和生成评分结果尚未完成。
- [decision] 四新身份、英语AVE、每身份两共同seed、官方head+lip+portrait完整流程；训练前数据门为强制条件。
- [decision] 同人不同原录制测试，不将同视频切片、不同文件名或上传ID自动称独立录制。
- [limitation] 四身份是探索性重复，不以片段/帧制造总体显著性；预训练暴露与感知质量分别限定。

## Relations
- implements [[042-synctalk-metadata-ordered-identity-replacement]]
- implements [[041-synctalk-eligible-recording-fallback]]
- implements [[040-synctalk-recording-role-fallback]]
- implements [[039-synctalk-shared-training-flow-geometry]]
- implements [[038-synctalk-training-owned-geometry-and-lazy-loading]]
- follows [[51-synctalk-may-eval-gain]]
- extends [[50-synctalk-may-gain-training-pilot]]
- relates_to [[研究交接总览]]
## 训练运行中的精确字节缓存实施
见[[043-synctalk-byte-exact-epoch-cache]]。首次正式epoch完成后，已独立F全成员持久验证、任务PID退出且无reader的冻结完整cohort_selection可改为相同SHA硬链接；关闭trace/stdout可透明NTFS压缩，须原verify_persisted前后通过。活controller/数值源、运行中冻结副本、数据pins/初始化/实际预算及粗10/fine17完整state保持；每个路径的完整字节仍可读，失败/恢复intent保留。原失败准备stage1缓存仅在逐256MiB及整包独立Ubuntu云读回、F receipt与最后source SHA/no-reader均通过后退休本地重复档，并留locator；PCM/几何/false/manifest保留，旧路径使用前恢复。此项只改变存储，不改变科学条件、门阈值、GradScaler/AMP政策或训练长度。

## 原FP16执行的数值失败边界（2026-10-17T18:13Z）

Brown/seed42/coarse ep5/+6已记录4636尝试、4484成功更新、152 AMP跳步，最后记录global28636，defaultGradScaler降至2.938735877055719e-39。下一步非有限loss触发controlled_training.py:66原资格门，精确NaN/Inf及该步optimizer delta未写入，不能补造；无中途checkpoint，原完整checkpoint仅在epoch完成后保存。已完成0/−6的同epoch各6000尝试/5998成功，但未完成三档块或粗phase验收。F failure archive20成员/全SHA通过，controller_qualification/nonfinite_partial_epoch_failure_attempt01。独立prefix逐4636帧/ray/LR与已有0dB reference相同；低scale不是根因证明，不能称数据增益科学阴性或一般模型失效。

原合同precision=original FP16/default GradScaler no fallback，optimizer_updates禁止coupled skips/scaler tuning/extra-step compensation/precision changes。当前不重启失败epoch、不删除旧失败、不换说话人/样本或按test分数选择。任何新数值稳定性诊断与后续重新训练须单列版本、预算和授权，不能把重跑更新隐入旧attempt。