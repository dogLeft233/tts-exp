# FlashHead Pro 中文20人九评分器实验

本实验使用192个人目录已有的中文20源20条自然语音与60条云端TTS，保持同人配对和原克隆参考分层。音量按用户要求统一为RMS −24dBFS；规范原件不改，生成和评分绑定同一派生PCM16。该电平与旧RMS0.0376838条件不同，不能拼接旧自然分数。

固定Pro/stream、3.png、512×512、25fps和seed42。重复音频只有在完整WAV SHA、图片、种子/设置均相同且视频SHA相同时才复用；保留80条逻辑媒体与来源、draw身份。九项分开报告，无总分或投票。

远端实验根：`/home/data2/wangjiajun/av-evaluation/jobs/flashhead_chinese20_rms24_20261030_attempt01`。脚本远端副本在评分根的`experiments/flashhead_chinese20_20261030/`。服务器连接先读全局server-192 skill和飞书手册，实时检查资源，长任务用nohup/tmux。代码、环境、cache、输入输出全部位于个人数据区；不修改共享base、驱动或他人进程。

- `prepare.py`：只允许新run，核对原SHA并建立派生输入、冻结协议。
- `generate.py`：复用已加载官方pipeline，每片重新prepare_params重置种子/运动；检查非静态和音轨逐样点一致。
- `controller.py`：等待生成结束和锁释放，单UUID顺序运行九评分器与分析。
- `batch_adapters.py`：复用已验收评分函数体，权重一次加载，绑定新输入/输出和个人缓存。
- `analyze.py --run <结果根>`：独立读取原始产物，来源级配对、bootstrap、signflip敏感性及Holm9。
- `audit.py --run <结果根>`：只读核验原件/媒体SHA、PCM、RMS和原始矩阵/embedding/logits归约。
- `package.py`：审计通过后建立不可覆盖的证据归档。

SyncNet沿用中文原协议PCM16 MKV、min_track25及FP32距离矩阵归约。两个中心5秒分类器只用原生满足125完整帧的条目；不能满足的列不可测，完整来源配对7人。SyncReward沿用官方短片处理：window_segments内部重复尾帧/补齐片段，不给输入媒体补静音；全20来源与原生完整4.75秒的8人子集分开报告。PEAVS全80按文件名字典序一次官方batch64+16，不能逐片代替或从重复项单片分数推导。

三次TTS先在来源内平均，再与该人自然分数比较。分别报告同句参考12人、扩展参考8人、合计20人，实际完整人数按指标列出；无缺失插补。真实音轨平移控制与自然/TTS平均差分开，控制失败不支持感知同步结论。

测试：`python -m pytest tests/test_flashhead_chinese20_20261030.py -q`。
