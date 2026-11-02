# EchoMimicV3-Flash 官方默认配置诊断

在192个人目录执行；每次使用前读取当次飞书服务器手册。所有代码、环境、缓存和新产物都在 `/home/data2/wangjiajun`，单GPU UUID、CPU affinity0–3、4线程、生成器allocator22GiB，GPU生成和评分顺序执行。不得重跑旧attempt01包装器。

独立run：`av-evaluation/jobs/echomimic_v3_official_control_20261030_attempt01`。`run.sh`拒绝已有run；`control.py prepare`先封存计划，随后generation→score→analyze。生成与评分最终收据独占创建。本次源代码不改动旧生成器，只导入旧pipeline loader并将其收据路径绑定到新run；中断后不能盲目重启，应先核验已有产物和部分输出。

本次是6来源诊断：在12同句与8扩展分层中，按自然语音长度排序，各取最短、中位、最长来源；每来源自然和既有C1 TTS各一条，共12条。选择不使用分数。不能替代原中文20来源、三TTS均值的完整实验。

| 条件 | 驱动端音量 | 原生帧数 | seed |
|---|---|---|---:|
| baseline | 原RMS−24 | 完整4k+1 |42|
| lufs_only | 官方整段−23LUFS后截前缀 | 完整4k+1 |42|
| cap_only | 原RMS−24 | min(81,原生4k+1) |42|
| seed_only | 原RMS−24 | 完整4k+1 |43|
| official | 官方整段−23LUFS后截前缀 | min(81,原生4k+1) |43|

官方`run_flash.sh`固定revision `7e89489ca51c0d008fc1963ec6c03fc5bd0b9397`。8步、Flow_Unipc、CFG6/audioCFG3、shift5、TeaCache.1/skip5、BF16等原条件已经一致。sample_size768是上限，作者不会放大512图；所有条件实际512×512。保留已验证CPU offload、BF16 loader修复与保留mask的SDPA；不宣称无修改上游CLI或论文benchmark复现。

先重放一对旧baseline并核对全部解码视频帧哈希。四个干预条件生成48片；基线从冻结视频只读取用，不重新包装回旧目录。每片绑定原图、输入哈希、设置、原生帧数、解码音轨逐点一致、非静态画面及显存峰值。

SyncNet评分固定共同前缀：每来源`min(81,N原生帧数,T原生帧数)`，该来源全部条件同长，音轨为相同RMS24 PCM。官方LUFS只改变条件特征，原入口成片也使用传入原音轨；不把评分器输入增益混入生成器干预。另设`scorer_gain_only`：固定baseline视频，单独把评分音频改为−23LUFS PCM16。保留31列原始距离矩阵，使用既定FP32归约，并以FP64独立核对。

`shift_control.py`在主评分后单独执行真实±400ms循环音频位移控制，检查最优offset反向移动约10帧；Sync-C搜索时延，不能要求可恢复位移必然降低confidence。控制不进入来源配对均值。

`finalize.py`只在所有生成/主评分/时移控制退出后运行，独立复算来源均值、检查父级pins、保存输入/源码/环境、图和接触表，生成不可覆盖证据包。原始逐片与来源配对结果分开；六来源仅描述性bootstrap区间，不做确认性p值，不把Sync-C解释为感知同步。单变量臂定位主效应，联合官方臂仍可能包含交互，不能强行把联合变化完全分摊。

首次评分预检查发现：H264数据包复制截取81帧时，末帧PTS为3.24而非3.20秒；尚未进入SyncNet前向。`score_repair.py`保留失败目录和源码身份，改用按解码帧截取、libx264 CRF0/r25/bf0无损编码，并逐片核对全部前缀解码像素哈希，再沿用原评分器、权重、归约和配对统计。原模型视频、计划和`control.py`保持不变。启动器attempt02的CRLF兼容失败在评分前发生，`post_attempt03.sh`为LF格式的执行入口；两份失败源文件和日志均入归档。不得直接重跑这些已执行入口。

证据包保留原模型视频、基线视频、原始距离矩阵、裁剪音频、像素审计和执行修复来源；只排除可由它们无损重建的评分前缀容器及解码工作缓存，并记录所排除容器的哈希。

本地测试：`python -m pytest tests/test_echomimic_v3_official_control_20261030.py -q`。服务器执行入口需`nohup taskset -c 0-3 bash <新S>/run.sh`；固定run已执行后不再次启动该入口。
