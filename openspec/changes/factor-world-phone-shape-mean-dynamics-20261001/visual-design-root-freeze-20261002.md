# μ/ρ 与 SELF_GRID：独立视觉验证冻结

状态：ROOT_APPROVED_PRE_INTERPRETATION_ADAPTER_DESIGN。本设计于完整声学/SyncNet 计算已完成、ROOT及实现代理尚未读取新处理效应和几何结果时冻结。它落实既有 eval-design-root-freeze.md 的 independent visual tracking 要求，不声称生成前预注册。执行计划须另绑定实际文件字节、代码和运行环境；本文件批准实现以下固定设计。

## 输入、身份与完整分母

使用已完成 MODEL14 的全新视频，160 recipient contexts、80 evaluation clips、40 speakers、每人两句，五 cell `q00/q10/q01/q11/SELF_GRID`，共800视频。两种评分音频域共享同一视频，只跟踪一次。每方向80句/40人。按 authoritative execution plan 的 ordered context identity、sample/speaker/direction/cell 显式连接；逐文件独立SHA验证800视频。主face身份保持生成计划，不选择其他face。

旧十臂WORLD视觉轨迹只提供已冻结算法/校准和参考输入来源，不作为新800轨迹。新track plan绑定 execution plan、rows/results/protocol/status/runtime flags、physical manifest+QA、face来源和原MFA qualification，记录raw SHA与有则embedded/canonical fingerprint，两类哈希分开。必须实际覆盖全部800行，失败仍保留行和状态。

## 固定全时钟与提取

每context新 `q00` 的完整解码 timestamps 是 recipient-reference clock。q00是自身WORLD谱形/自身时轴基准；四处理保持相同recipient时轴。计划和preflight必须验证五视频的FPS、帧数及全长度与official recipient video-clock一致；FPS25，official frame-count从执行计划机械读取，禁止裁切到共有最短时长。任何不一致写CONTROL_FAILED，不换reference或裁片。

复用冻结video_features提取MediaPipe全帧特征，固定landmarker `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`。新runtime实际记录Python/NumPy/OpenCV/MediaPipe、线程和spawn策略；检查声明/解码帧数一致及严格递增timestamps。CPU提取，不执行TFG/SyncNet forward。

五轨使用旧core单调匹配：容差 `0.5*median(diff(q00 timestamps))+1e-12`，保留q00整个clock与五组index maps。共同有效mask是五轨matched且valid的交集。保存fullclock、matched/valid/common masks与SHA。velocity只用reference相邻且共同有效的帧，validity hole不桥接。无共同frame或无相邻pair保留INCOMPLETE。

## 先封存availability，后读取几何

availability phase只可读取身份、timestamps、valid、帧计数、recipient phone labels/intervals、原MFA qualification与overlay；不可读取mouth_open/width/height/area/coordinates、aggregate geometry或score effects。封存每context时钟/mask、tokens/classes/支持计数、每方向40人的complete-pair phone eligibility mask和全部input hashes，ROOT读回确认后才运行geometry consumer。

保持原CAL `latency=-0.085s`、tone adapter五tone-bar allowlist、旧normalizer/bilabial-open_vowel分类。每类每句至少一个有支持token，每token至少一个共同有效frame；time interval `[start-0.085,end-0.085)`。phone tier来自recipient，禁止供体phone或根据效果选择类别/lag/minimum。原pair_qualified=false即使physical overlay有标签也仍phone unavailable。资格从新五轨metadata机械导出；旧16/17合格speaker读数不预填为新结果。

## 指标与54端点

每context/cell固定三个指标：共同有效frames的mean aperture；共同有效相邻pairs的mean absolute aperture velocity per second；每clip内有支持token等权的open_vowel aperture mean减bilabial aperture mean。精确定义沿旧core/tone adapter。availability冻结后发现非finite/missing geometry不能删eligible人或用零填充。

九个contrast固定：mean=q10−q00；trajectory=q01−q00；joint=q11−q00；grid_cycle=SELF_GRID−q00；mean_minus_trajectory=q10−q01；mean_beyond_grid=q10−SELF_GRID；trajectory_beyond_grid=q01−SELF_GRID；joint_beyond_grid=q11−SELF_GRID；factorial_interaction=q11−q10−q01+q00。九contrast×两方向×三指标=54，全部规划端点与INCOMPLETE都保存。视觉不重复A_RAW/A_RT维度。

## 固定40人bootstrap与共同99%区间

按40人排序roster冻结一次PCG64(20260930)生成20000×40有放回speaker index/count矩阵，所有54共享同一矩阵。每speaker先等权均值其两句，方向分别估计。overall点估计40人等权，任一人缺两句合格geometry则相应endpoint INCOMPLETE，禁止缩成39。

phone点估计在几何前冻结的complete-pair eligible speakers上等权。仍保存完整40人意向分母和mask。每draw使用 `sum(counts[eligible]*values[eligible])/sum(counts[eligible])`，不能用NaN×0，也不能将缺失当零effect。保存全部draw denominator；任何为零则协议FAIL，不删draw/换seed/换roster。已冻结eligible人缺geometry导致相应endpoint INCOMPLETE，禁止缩小eligible mask。

使用shared draws、bootstrap sample SD作为SE、以(bootstrap estimate−point estimate)/SE形成studentized max-|t|，在所有estimable54中计算一个two-sided99% simultaneous临界值和区间。保留计划数/可估数/不完整数、SE零时的确定性状态和精确实现。不得把较小子族称为完整54族。独立视觉族与11776评分族分开，不混合几何单位。输出全部speaker values/masks/index/counts/draw denominators/draw estimates/maxT/point/SE/CI以供第二实现完整重建。

## 复核、归档与解释

新版本extractor/availability/geometry consumer/independent reconstruction分别冻结源码SHA。ROOT审阅stage/preflight与实际source/input字节后执行。完整800 video/NPZ/sidecar身份、array shape/dtype/SHA、input/output inventory、PID/starttick/exit与runtime实际读回；独立重建全54点估计、20000共同bootstrap、denominators、SE/maxT/CI。全部800轨迹及代码/计划/availability/geometry/独立receipt须可恢复归档并实际完整restore核验。

视觉改变可解释嘴部运动响应；phone separation依赖独立标签/固定lag/有限class资格。不得把aperture上升直接命名为嘴型更准确或感知更同步。此设计不新增感知盲评或改变先前11776数值定义。

## 冻结来源

根设计eval-design-root-freeze.md raw `e17d257e43a78af6d7a291641ade6825b61c430ba51b8052383cab54dbea31fb`。接口草案 `.codex/scratch/shape_visual_protocol_draft_20261002_attempt02.md` raw `8d698504283daebabf4f9bca3d741d36f188114e0f2a4a622bb934806574d0ae1` 已ROOT全文读取。

旧复用源码：video_features `2ce7006030e7f93b139cbcc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4`；visual core `3a28f2e7060e8458859455bcb0283f8d1b98558f16cc4e185efd0902d73ea15b`；tone adapter `b725b5feb5c3f30bea754f73bf897d606e04f59940fe8f4fdb13f269306fce98`。原CAL/eligibility/grid inputs须在执行计划逐字节绑定并独立检查，不能将旧track seal称为新证据。
