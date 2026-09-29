# E2：波形标量增益的replacement探针

先完整读 `../../parallel-replacement-next-20260909.md`。这是已见16条/8组的新固定干预，假设是幅度改变Wav2Lip驱动前端后可能改善与原N的匹配；不是复活“响度已解释TTS收益”。不使用TTS或音量归一化目标。

## 三臂与精确波形

解码完整原N为int16数组p，先以float64计算：
`g_plus=min(10**(3/20), 0.98*32767/max(abs(p.astype(float64))))`。
`g_minus=1/g_plus`。所有记录分别按这个只依赖N峰值的同一规则。
若peak=0或g_plus<=1，整队列INPUT_DEGENERATE，不降音量后重试，也不删除该条。
N保持原PCM字节；GAIN_PLUS/GAIN_MINUS为 `rint(g*p)` 后int16（ties-to-even），断言[-32768,32767]内，不做clip/limiter/RMS归一化。两候选都保存全长16k mono PCM16 wav，增益倒数对称是在量化前，不能声称量化后严格可逆。

从候选PCM解码 `astype(float32)/32768` 的前61440 samples，走父官方audio.melspectrogram得到[80,308]和93个chunk；不直接给mel常数平移。先从原N同路径复算M最大差<=1e-6、chunks一致。不要调用Wav2Lip audio.save_wav（内部会重归一化，抹去处理）。固定两个候选，不扫描分贝或挑选方向。记录实际g/dB、峰值、RMS、量化误差、mel变动/clip与所有hash。

所有TFG输出最终mux未经修改的完整N。增益候选波形只作驱动，不当评分音轨；这样评分器不会因听到放大的音轨而直接改变输入。仍然只检验Wav2Lip/SyncNet这一个系统，不能外推其他模型有同样归一化行为。

## 运行与决策

公共独立控制2 replay+2 parity后生成32候选，共34新视频/36新评分。两个主contrast GAIN_PLUS/N和GAIN_MINUS/N都按公共gain报告。均不通过→NO_WAVEFORM_GAIN_ESTABLISHED；只有一方向通过→对应AMPLIFY_SIGNAL_TO_CONFIRM或ATTENUATE_SIGNAL_TO_CONFIRM；都通过→GAIN_RESPONSE_MECHANISM_UNRESOLVED，不挑高分方向。

完整16条均纳入主端点；任何记录量化后候选与N完全相同则INPUT_DEGENERATE。无需用候选自己的SyncNet评分证明它有效。阳性提供一个波形可实现的单模型候选，下一步仍是独立source-group和第二TFG确认；未达到泛用生成头或内容语义证据。

## 实现验收

包 `wav2lip_waveform_gain`，CLI/阶段/文件/资源按公共契约。纯PCM处理尽量小函数，不要新增通用音频框架。独立validator从原PCM重做rint/增益/长度/全长波形身份，再验mel/视频/评分/统计；不得调用producer波形构造函数。

反例测试：int16最小负值的abs溢出、headroom上限、静音退化、ties-to-even、输出不clip、save_wav归一化被检测、增益候选误mux成评分音轨必须失败、输入不变时相同输出、两个方向不得漏报。科学阴性不自动换参数。

BM唯一实体 `Wav2Lip waveform gain probe 2026-09-09`，报告实际增益范围、两方向C/D/A全部CI、控制证据、预算和waveform可实现但尚未泛化的边界。
