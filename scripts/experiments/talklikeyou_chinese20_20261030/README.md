# TalkLikeYou 中文20来源九评分器实验

复用报告76的80条RMS−24dBFS规范WAV（20N+60既有TTS）、同一3.png与来源分层；每人三TTS先平均，不把重复draw视为独立来源。无新TTS、增益处理或时间移动。所有新增文件落在192个人目录，旧报告和产物保持只读。

官方源码d1088c49071148853c76d8a8f0c1fb97d895758b，权重revision652b744bbf86510c370ff6f302d046bdd63632e1，prepare重新核验部署源码与权重SHA。沿用推荐habit192、stage0+neutral、motion1.0、CFG1.3、唇部/姿态各一步、audio pose、平滑、overlap5、crop2.3；固定seed42。每个唯一音频在新进程完整加载官方入口，避免跨片缓存/随机状态。相同输入的逻辑draw保留明确alias，所有80媒体仍按PEAVS固定64+16排序批次评分。

TalkLikeYou为唇部FM+Ditto姿态/渲染组合系统。原生25fps；完整16个mel格窗口只生成约输入时长减0.2秒的画面。不移动嘴部时钟、不补尾帧、不延音、不按分数调整offset。完整原音轨逐样点保存在ALAC评分媒体，原AAC为播放副本。两个固定5秒分类器依据实际画面与音频共同完整125帧支持决定资格，不能把5秒音频当作5秒画面。官方短片SyncReward内部补齐照旧保留，完整4.75秒子集另报。

评分及控制沿用报告76/77的既有代码：SyncNet官方FP32/min_track25；CoGenAV Eq1；AVS_u帧cosine；MTD/VocaLiST K5；两分类器P0与整数中心窗；SyncReward BF16；PEAVS固定批次。真实移音控制沿用适配器，SyncNet/PEAVS/SyncReward本队列未重测的边界如实标记。10,000来源bootstrap、100,000signflip、各分层Holm9；同句参考12主要，扩展8与all20补充。评分不是感知真值，也不合成总分。

远端脚本：av-evaluation/experiments/talklikeyou_chinese20_20261030。run：av-evaluation/jobs/talklikeyou_chinese20_rms24_20261030_attempt01。prepare→generate --smoke（首来源N/T）→generate→controller→audit/PEAVS敏感性→package→deliver/verify_local。GPU阶段顺序执行，nohup、单GPU UUID、CPU4、torch显存60%；先读当次飞书1048手册并检查实时资源。失败保留原件，不自动重试科学单元。

旧Wav2Lip中文批次电平与运行时须单列边界；与FlashHead/Echo可核对同一规范输入，原生视频支持仍可能不同。正负均值或显著性状态相同不构成模型效应等价证明。
