# A：natural局部时间对比度

## 入口和假设

先完整读取 `../../parallel-replacement-probes-20260909.md`。其资产、媒体、控制、统计、隔离和BM规则均为本设计必需部分。唯一问题：微弱抑制或增强natural mel的局部快变化，是否在原N音轨上改善同步？这不是句义条件，也不是TTS频谱搬运；不同于旧已训练natural-only policy。

## 三臂（参数一次冻结）

对P中的每条N mel M（float32 `[80,308]`）先转float64。只沿时间轴使用5点对称binomial核 `[1,4,6,4,1]/16`；左右各2列采用reflect，不含端点重复（NumPy pad mode=reflect）。不卷积频率轴。记平滑结果S。

```text
R = M - S
R[:,0:2] = 0; R[:,-2:] = 0
b = max(abs(R))
a = min(0.25, 0.5/b) if b > 1e-12 else 0
N      = 原M逐值复制
SMOOTH = float32(clip(M - a*R, -4, 4))
SHARP  = float32(clip(M + a*R, -4, 4))
```

两候选共享同一a；所有308列保留原时间坐标，前后2列bit-exact，不shift、warp、插值、TTS、训练或按分数调整a/核宽。每条preclip最大改动≤0.5 mel单位。不保证clip后每频率均值/谱统计完全不变，须报告其变化，不能声称已纯隔离时间因素。

prepare保存R/a、pre/postclip L2、clip比例、每频率时间均值差、端点和chunk索引。任一记录候选与N完全相同，终态 `INPUT_DEGENERATE`，不补别的样本或加大a。合成常量数组必须恒等，中心脉冲R必须对称且中心不移动；恒等算子必须逐值复现官方N chunks。

## 执行

全16条、8组，参考图均为P/F0静态缓存。公共控制验收完成后才生成SMOOTH/SHARP共32视频/32评分。另有2个F0/N replay视频/评分、2个parity评分；合计34视频/36评分。主N仍为P缓存，两个replay只证明运行链一致。模型、N音轨、U、k0全部冻结。此轮只支持mel接入，不实现vocoder/waveform头。

## 唯一决策

两个预指定主contrast为SMOOTH/N、SHARP/N，均完整报告，不仅报胜者。每方向 `pass` 要求：ΔC/ΔD/ΔA三项99%组bootstrap CI下界全>0，平均ΔC>0.05，至少7/8组三项同时正。

按顺序：工程/validator不一致→BLOCKED；输入退化→INPUT_DEGENERATE；控制失败→CONTROL_FAILED；均不pass→`NO_TEMPORAL_CONTRAST_GAIN_ESTABLISHED`；仅SMOOTH pass→`SMOOTHING_SIGNAL_TO_CONFIRM`；仅SHARP pass→`SHARPENING_SIGNAL_TO_CONFIRM`；两者pass→`BIDIRECTIONAL_SIGNAL_MECHANISM_UNRESOLVED`，不择优或宣称同一单调机制。每个阳性也只建议冻结构造做独立source-group确认，再考虑波形可达性；不是replacement_confirmed。

## 下游交付

包名 `wav2lip_natural_temporal_contrast`；CLI、通用产物按公共契约。新增聚焦测试：沿错轴卷积被拒、reflect端点、常量恒等/脉冲不移位、两方向同幅度且clip后正确、只自由分数正不通过、两个主contrast不漏报、无控制直接candidates不生成。

BM唯一实体 `Wav2Lip natural temporal contrast probe 2026-09-09`。记录两个方向全部数字、clip变化、实际预算、mel-only边界。阶段自审后更新同一实体，不写其他分支。
