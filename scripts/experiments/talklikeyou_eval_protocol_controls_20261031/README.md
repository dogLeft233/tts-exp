# 中文 TalkLikeYou 的 SyncNet 门槛和音轨对照

使用中文报告79已完成的80逻辑视频、41唯一生成，固定画面与全部既有TTS draw，不重生成。服务器输入只读，输出落独立job；不恢复HDTF暂停实验。

四格为原生成AAC MP4／规范PCM MKV × min_track25／100。两档门槛实际调用官方完整脸检、追踪和裁剪；只有裁剪AVI的SHA完全相同时，100格才复用对应25格的网络前向。总帧数≤100的片段不可能达到严格大于100检测的条件，记录为不可测，不赋零分、不补帧。

主支持为一个来源的自然及三条逻辑TTS在全部四格都可测的交集；原12个同句参考来源单列。每来源先平均TTS，再减自然，来源等权。音轨25格的全20来源联合支持另报。协议效应为两个门槛差及100格原MP4减PCM，共三项，每分层Holm3；不替代原九评分器Holm9。

`run.py` 冻结全部输入、实际原AAC别名、核心源码及权重SHA，再顺序运行，保存中间AVI、官方提取WAV、人脸轨迹与距离矩阵。波形互相关只诊断延迟，从不对齐评分音频。`analyze.py` 计算来源bootstrap与exact signflip。`audit.py` 从原矩阵和逻辑draw重建均值，并核验画面／检测／裁剪不受音轨选择改变；当前修订额外独立回算bootstrap、signflip与Holm。

`finish.py` 等待既有producer结束后进行冻结后处理和完整归档。本轮原科学producer PID11201完成27条后，在第28条评分前因共享GPU资源门停止。revision02改为等待资源，并复用已核验通过的pipeline阶段；新producer PID60457、finisher PID60458。辅助PID12360的重复启动在fresh-run断言处被拒绝，没有重复科学前向。历史日志、failure.json与resume_gate_preflight.json保留；科学完成以stages、terminal_receipt及独立审计为准。

原后处理随producer资源门退出，未执行分析。revision02后处理另存源码及pins，增加对bootstrap、signflip、Holm和网络实际读入PCM的独立复核；旧版正文保持原样。

本地针对性验证：`uv run --with numpy --with scipy --with pytest python -m pytest -q tests/test_talklikeyou_eval_protocol_controls.py`。
