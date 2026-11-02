# May 四类已知干扰的强度与音量配对扫描

冻结 facebook/wav2vec2-base，同报告63的May train/test各首320080采样，先全split原float32标准化，之后才加噪声和乘0.5。不训练、不渲染、不评分。每种噪声三个固定实例；名义整split SNR0–100dB每10dB，加120/140dB；FP64对首实例80/90/100/120dB复核。默认746次前向，具体计数与所有代码/源/输入SHA在前向前protocol冻结。

噪声来自本地其他身份自然录音（Brown、Patrick、Debbie、Cathy），不占用其训练进程：single_speech取Brown/Patrick/Cathy三个录音实例；babble4将四录音片段各自定RMS后等权叠加；modulated_ssn以四录音首60s的平均归一化Welch PSD造随机载波，再乘独立2–8Hz组合包络；clicks是可复现的合成稀疏窗口点击，不冒充真实键盘录音。源自然录音没有做声源分离，不假定完全无底噪。所有裁切、包络及点击事件保留。

```powershell
& $python tests/experiments/test_may_four_noise_20261020.py
& $python scripts/experiments/may_four_noise_20261020/run.py --output F:/Datasets/tts-exp/<fresh-root>
& $python scripts/experiments/may_four_noise_20261020/analyze.py --output F:/Datasets/tts-exp/<fresh-root>
```

同实例/剂量比较s+n与0.5(s+n)，参照分别为s与0.5s。主端点为末层逐帧单位方向变化RMS，直接float64向量相减，不使用2−2cos；辅以相对L2。比值小于1仅代表此编码扰动减小，不代表噪声已清除、语音更好或训练更好。七层原值64帧、完整末层、逐帧矩均保留；本轮8个参照逐层与报告63相同；48个首GN epsilon2.5e-6补偿控制逐层恢复原音量。

FP32两音量扰动都超过相应无添加噪声FP32/64差异10倍才标为可分辨。该规则是诊断参照，不是严格误差上界；FP64覆盖一个实例而非三实例。各SNR的三实例范围是描述范围，不是置信区间。整体一致减敏仅指本批两段×3实例均同向；突出≥20%减小仍需各条件可分辨，不作训练噪声筛选或人群外推。

analyze不导入producer：独立重建24个完整噪声数组，重建全部输入SHA，重算七层raw采样矩与完整末层，检查旧参照桥接、补偿、尺度加法掩码及精度差异。点击噪声额外记录全段和有效事件支持内SNR、峰值与crest factor；固定全段SNR不等于各时刻具有相同干扰强度。

本次v1在370个已提交前向后发生首GN完整输出125MiB主机内存分配失败，原失败与未提交输出不删改。recover_streaming.py在新v2保存旧根完整SHA快照，复用370科学前向，19个独立进程按256时间行流式处理剩余376条件；原科学协议/producer保持，新增34无添加噪声桥接，均对报告63完整七层原值SHA严格相等。另4次桥接为原计划FP64参照。有效科学条件仍746，验收成功前向含额外控制共780，原失败未提交输出另计并保留。流式矩/样本/hash的CPU等值测试通过。

```powershell
& $python scripts/experiments/may_four_noise_20261020/recover_streaming.py prepare --output F:/Datasets/tts-exp/<fresh-recovery-root>
& $python scripts/experiments/may_four_noise_20261020/recover_streaming.py run --output F:/Datasets/tts-exp/<fresh-recovery-root>
```

prepare当前绑定本次失败根，拒绝覆盖；复现新实验应先检查原路径参数和固定SHA，不把v1的失败误作科学阴性。
