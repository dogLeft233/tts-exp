# May 四类已知干扰的训练对照

沿用报告65的第一固定干扰实例，四类×30/90 dB名义整split SNR×1/0.5尺度，另两无添加噪声尺度基线，共18个新模型。May原始监督划分不变；一次完整native标准化后加噪、整体缩放、冻结同一wav2vec2。

18组共享完整随机初始化、seed42、12000成功FP32更新、4150训练帧、逐步实际frame/ray/LR和随机域。每2000步保留全状态；正式训练前每组独立2步roundtrip，不用资格state warmstart。固定最终EMA12，不按test选checkpoint或追加剂量。

主推理均为原May无添加噪声、尺度1；尺度0.5为次对照。固定两历史10秒clip、原PCM16评分、GT共享ROI/时钟、guard20主/full次，报告C/D0和嘴部MSE/L1。核心是噪声×训练尺度的差分之差；不直接比较noisy-low与clean-standard就声称去噪机制。一个训练seed/一个固定噪声实例，不代表总体效应。

本机F根`F:/Datasets/tts-exp/may_four_noise_training_20261020_v1`，媒体单独在`E:/Documents/tts-exp/runs/may_four_noise_training_20261020_v1/renders`。Ubuntu持久根`/home/wjj/tts-audio/may-four-noise-training-20261020-v1`；同机其他任务不停止，所有旧失败与暂停试验不变。

顺序：冻结protocol→features.py→review_features.py→deploy.py→launch.py，同时backup.py只读观察闭合文件并逐成员SHA回读。controller串行资格、独立QA、正式训练、QA、渲染、评分，第一次真实失败停止后续，不自动重跑或改数值配置。完成后独立媒体/距离QA及正式报告另执行，不能把science_complete当最终科学结论。
