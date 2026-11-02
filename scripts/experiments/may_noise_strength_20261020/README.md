# May 已知噪声剂量扫描

只做冻结 wav2vec2 前向，不训练、不渲染、不评分。使用既有 May 整 split 标准化后 train/test 各首 320080 采样，复用报告61封存的白/粉红噪声和3种子。比较 s+n 与 0.5(s+n)，各自减去匹配尺度无添加噪声参照。

19档名义整split SNR：0–100dB每10dB，120–260dB每20dB。FP32全扫描，FP64对120/200/260dB、首种子做精度诊断。记录7层，主端点是末层逐帧单位向量差RMS，使用float64直接相减，避免2−2cos在极弱扰动时消减为零。首GN epsilon补偿和上轮30dB逐值桥接作为控制。

```powershell
& $python tests/experiments/test_may_noise_strength_20261020.py
& $python scripts/experiments/may_noise_strength_20261020/run.py --output F:/Datasets/tts-exp/<fresh-root>
& $python scripts/experiments/may_noise_strength_20261020/transition_fp64.py --output F:/Datasets/tts-exp/<fresh-root>
& $python scripts/experiments/may_noise_strength_20261020/analyze.py --output F:/Datasets/tts-exp/<fresh-root>
```

Python和依赖位置见脚本中的固定路径。producer在任何模型前向前写protocol.json，拒绝覆盖根。analyze独立重建输入SHA、全部层raw样本矩、完整末层矩、旧run桥接、数值控制，输出summary/ratios/precision_diagnostics/independent_qa及PNG/SVG。图中更高SNR表示更弱添加噪声。

“可分辨”采用保守操作规则：两音量的噪声方向扰动都超过各自无添加噪声FP32-vs-FP64差异10倍；它不是严格误差上界。候选明显额外抑制规则是在同一噪声谱形下、两段×3种子、连续3档可分辨SNR均B/A≤0.8。完整比值照常发布；未达到该操作标准不排除细微或局部效应。

首次v1因旧Windows路径分隔符与manifest的POSIX索引不一致，在模型加载和前向之前失败；保留failure.json，用v2新根执行。不得重写历史根。

主扫描538次前向后，观察到test的部分弱剂量抑制、其他剂量增强，而train趋势不同，追加独立冻结的探索性transition_fp64协议：80/90/100dB、首种子、两谱形×两段，28次FP64前向，总566次。它保留原主协议，不是事先指定的确认性实验。analyze同时复核这28次及与主FP64七层参照的逐值桥接。
