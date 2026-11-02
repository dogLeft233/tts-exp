# EchoMimicV3-Flash 中文20人九评分器

沿用 FlashHead 已冻结中文20源的20自然+60既有TTS及同一3.png。输入直接复制上一轮 RMS −24dBFS WAV 并逐文件核验，不重新合成、调整电平或时间。生成器、评分器及其产物均在192个人目录；使用前读取全局server-192 skill与当次飞书手册。

模型为官方 `echomimicv3-flash-pro` 检查点，8步、512²、25fps、seed42，固定提示词“A person is speaking.”，Flow_Unipc、CFG6、AudioCFG3、shift5、TeaCache0.1/skip5且逐片重置。官方 `infer_flash.py` 的额外 −23LUFS 处理被显式绕过，以保持用户要求的RMS24条件；保留作者特征提取器配置（本次中文checkpoint的`do_normalize=False`，不额外做波形mean/std标准化）。模型CPU offload，torch显存上限22GiB，CPU4线程，单GPU UUID，GPU阶段顺序运行。

帧数依官方规则 `4*floor((floor(samples/640)-1)/4)+1`，不使用演示81帧上限。完整输入音轨保留，最多不足0.16秒尾部没有对应生成画面；不重复尾帧、不补静音、不拉伸/平移。各评分器使用实际共同支持区间，两个固定5秒分类器还要求至少125实际生成帧。官方SyncReward内部短片补齐保留，并另报原生完整4.75秒视频与音频支持敏感性。

实验根：`/home/data2/wangjiajun/av-evaluation/jobs/echomimic_v3_flash_chinese20_rms24_20261030_attempt01`。部署根：`/home/data2/wangjiajun/EchoMimicV3-Flash`。`prepare.py`绑定输入和冻结协议；`generate.py --limit 1`运行真实首片，确认后无limit续跑；`controller.py`复用已验证九评分器的前向与归约，仅绑定新输入及实际支持帧。

统计沿用来源单位：每来源三TTS均值减自然，主要同句12、扩展8、合计20分层；bootstrap10000、signflip100000敏感性、Holm9。无插补和综合总分，评分器变化不能直接解释为感知同步改善。完全相同输入/图片/seed/设置允许明确的成片alias，不能视作独立合成重复。

验证：`python -m pytest tests/test_echomimic_v3_chinese20_20261030.py -q`。完整实验后运行`audit.py`、PEAVS批量敏感性、`package.py`、本地`deliver.py`及`verify_local.py`。先前FlashHead证据不覆盖。

Flash是完整检查点，直接按发布的config构建并strict加载所有键，不先下载随后会被全部替换的基础DiT权重。基础模型仅取T5、CLIP、VAE和tokenizer；大文件对照官方SHA256，小文件对照官方Git blob ID。未安装FlashAttention时采用保留长度掩码的SDPA，避免上游fallback丢弃mask；独立CPU sliced-softmax及GPU BF16/FP32参照检查后再实际生成。

当前Diffusers0.35.2没有上游旧版`load_model_dict_into_meta`入口；T5低内存加载会回退FP32。已在私人包装器中使用CPU加载后显式转BF16，并在offload前断言Transformer、VAE、T5、CLIP四模块均为BF16。最初诊断中止的日志、旧worker、显存记录与修复绑定保留在run的preflight文件中；未采用其未完成产物。
