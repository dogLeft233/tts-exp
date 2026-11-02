---
title: VocaLiST原版官方Release权重补齐 2026-10-28
type: task
permalink: tts-exp/tasks/voca-li-st-原版官方-release-权重补齐-2026-10-28
status: complete
weight_source: OFFICIAL_GITHUB_RELEASE_FOUND
scoring_authorized: true
deployment: PASS
official_native_scoring: COMPLETE_80
independent_audit: PASS
---

# VocaLiST原版官方Release权重补齐 2026-10-28

用户问完整VocaLiST如何可用；此前已有在Ubuntu部署原版授权。此轮发现并当面更正重要遗漏：作者个人仓库 https://github.com/vskadandale/vocalist/releases/tag/interspeech2022 从2022-09-15即有官方权重备份，title/body明确avoid dependence on Google drive。旧检查Drive空及HF无匹配不等于没有官方权重；先前漏查Releases，不能继续使用原版权重不可得作为现行结论。

## 权重与检查

官方Release有两个独立ZIP而非拆分必须合并：001.zip 1761847911字节，含唱歌Acappella5f/25f及tensorboard；002.zip 882753219字节，只有VocaLiST_Weights/vocalist_5f_lrs2.pth，解压961635703字节。已通过HTTP Range各读取尾部131072字节并用ZipFile列中央目录，收据/home/wjj/av-sync-eval-20261028/weights/vocalist/official_release_zip_index.json，不需下载001包。

002官方地址 https://github.com/vskadandale/vocalist/releases/download/interspeech2022/VocaLiST_Weights-20220915T135751Z-002.zip 。当前下载到weights/vocalist/vocalist_5f_lrs2.zip；初64worker1MiB session30522由仅自有PID59777结束后升并发，当前session77894，workers256/chunk512KiB/reuse-extents，保留69已写分片。最新实际已分配约62.5MiB，非完整ZIP；稀疏.parallel名义长度不代表完成。不要从.partial解压。range_download.py将完成后原子rename并写DOWNLOAD_COMPLETE。

现有原版source /home/wjj/av-sync-eval-20261028/vocalist，origin确是vskadandale/vocalist.git、SHA c265b59bcf7cd8559a265aaf454a9333ec24cb0f。env /home/wjj/.venvs/vocalist-20261028；没有新训练必要。需解压、strict加载models.model.SyncTransformer，预期约80.1M参数，真实媒体GPU前向后才称部署完成。

## 重要前处理

原版test_lrs2.py实际Dataset cv2.imread BGR、cv2.resize96，再/255，取下半48×96；音频sf.read后FloatTensor，官方Mel80/FFT800/hop200/Slaney。不同于MTD torchvision RGB0–255；不能共用MTD视频张量。可复用原512 JPEG帧文件，但必须由原版Dataset重读。后续官方K5..K15指标也应直接原版作者代码，不自定义平均logit。

本次只补齐原版部署，不自动重新评分80条，待先完成真实权重运行验证。Windows1MiB下载测速19.8秒，未更改用户网络或开放代理。HF froggeric/vocalist是无关RVC，不使用。

## Relations

- unblocks [[Ubuntu同步评分器部署与EARS40云TTS 2026-10-28]]
- relates_to [[70-mtd-official-native-cloud-preference]]

## 后续执行已接上（尚未下载完成）

原版模型CPU初始化及CUDA检测通过，参数80106561。下载最新约118.25MiB/841.86MiB，session77894仍运行，未完成、未strict加载。新增scripts/experiments/vocalist_official_smoke_20261028.py与对应real receipt integration test，两机已同步。下载完成后才解压官方002中唯一vocalist_5f_lrs2.pth，按原版Dataset的BGR/255、作者Mel处理原512真实JPEG，strict加载80M模型并GPU前向，写smoke/vocalist_receipt.json。

自动依赖流水线/home/wjj/av-sync-eval-20261028/finish_vocalist_deploy.py 已独立会话后台启动，状态/home/wjj/av-sync-eval-20261028/vocalist_deploy_chain.json，日志vocalist_deploy_chain.log。当前WAITING_DOWNLOAD；脚本等待实际DOWNLOAD_COMPLETE后执行解压、真实GPUsmoke及对应artifact测试，成功才记COMPLETE_DEPLOYMENT_SMOKE_PASS。下载进程若中断可自动断点续传一次；失败记录FAILED。不创建定时任务、不发消息或启动80条评分。状态读取允许下载进度JSON短暂重写，不把race当失败。需以实际receipt判断完成，不可只凭源码环境称已部署。


## 官方全组评分授权与续传修复（2026-10-28）
用户最新明确授权：部署验证通过后，按原版官方方法重评此前80原生媒体/60配对，报告TTS胜率、平均T−N及Sync-C冲突率；此授权取代上文“不自动评分”范围。实际旧下载因SSL EOF退出，旧链FAILED，未部署成功。已保留分片续传，改为官方Release重定向后的release-assets域256并发、512KiB块，独立attempt04下载日志，避免逐片Github重定向握手；不使用第三方权重。修复原版smoke AST环境缺少os变量。
新增scripts/experiments/vocalist_official_cloud_rescore_20261028.py与对应tests，两机同步，冻结run vocalist_official_cloud_rescore_20261028_attempt01，复用正式MTD全512 JPEG帧但由原版BGR/255 Dataset重读、全部新前向。作者Dataset/calc_pdist/eval_model原文AST执行；K5主、K7–15全部报、无时移校正。两个实际CPU端点测试通过。新依赖链vocalist_official_scoring_chain.py等待完整ZIP，真实GPU smoke及artifact通过才自动评分/统计；状态vocalist_official_scoring_chain.json、日志同名.log。尚无原版完整评分结果，不能引用蒸馏版替代。


## 原版真实部署已验证，官方评分运行中
官方002 ZIP完整下载882753219字节、唯一pth解压961635703字节成功（普通ZIP CRC通过，无额外大文件SHA）。实际最终稳定下载配置为32并发持久连接，临时Release URL过期自动刷新；无需再重启任何下载。
smoke/vocalist_receipt.json实际PASS：strict_load=true，80106561参数，作者原版Dataset BGR/255，torch2.5.1+cu121，真实GPU前向logit3.50795126，峰值342389760字节，耗时2.19秒。部署artifact测试实际PASS。vocalist_deploy_chain.json已用实际receipt更正为COMPLETE_DEPLOYMENT_SMOKE_PASS并指向新评分链；旧失败日志保留。官方全80评分已由vocalist_official_scoring_chain.py启动，仍须等summary和独立80矩阵复核完成才报告科学数值。

原版已知时移实测：首3自然样本±6400样本（±400ms）6/6控制通过，相对偏移变化全部准确±10帧，误差0；使用作者函数、新物理PCM音轨，在边界外guard20查看偏移变化，未校正正式原生评分。正式K5–K15保留全部作者窗口和边界掩码。known_shift_controls.json与对应test_original_model_known_pcm_shift_tracking_first3实际PASS。自然媒体20/20前向完成，60云TTS评分仍在运行；不要提前外推胜率。


## 最终完成
原版官方80媒体/60配对已全部完成，run vocalist_official_cloud_rescore_20261028_attempt01，两机小产物与80矩阵均已保存；正式报告[[71-vocalist-official-native-cloud-preference]]、部署[[VocaLiST原版部署]]（docs/deployment/vocalist.md）。K5主：自然.0400089796122、TTS.0306340349935，平均T−N−.009374944619（−.937494462个百分点），TTS18/60=30%、自然42/60，冲突36/60=60%（SyncT/原版N33，反向3），来源bootstrap95%[−1.915168,−0.020771]个百分点，准确区间以summary.json和正式报告为准。K7/9/11/13/15的TTS胜率25/20/15/10/5%，平均差−.937/−1.463/−2.139/−2.902/−3.608pp，冲突65/70/70/70/75%。
独立重算全80矩阵480官方端点、360配对及两支持集计数PASS，真实结果integration测试PASS；6/6物理PCM已知时移控制准确追踪±10帧。原生分数未校正。
重要限制：guard20曲线峰值80/80均−3帧；K5绝对分低，自然/TTS正确窗口72.45%/82.13%集中首尾，guard20均分仅1.2624%/.7972%，12/20及42/60内部为0。scores.json全矩阵含边界掩码global_offset_frames不可当真实零时差，偏移参考independent_audit.json guard20字段。故是官方评分偏好分歧，不证明自然嘴型更好或SyncNet有偏置；strict12 K5平均差区间跨0。
