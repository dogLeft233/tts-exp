---
title: server-access
type: note
permalink: tts-exp/docs/reference/server-access
status: active
project: tts-exp
verified: '2026-10-07'
tags:
- server
- deployment
- tts
---

# server-access

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

## 当前研究服务器 ✅

2026-10-07 用户迁移后的当前入口为 root@connect.westd.seetacloud.com，SSH端口18625。已实际认证连接；SSH ed25519 主机密钥与迁移前一致，指纹 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。旧 root@connect.weste.seetacloud.com:12349 为迁移前历史入口，不再用于新执行。

```text
ssh -p 18625 root@connect.westd.seetacloud.com
```

密码只在运行时内存或环境变量中传入，永不入仓库/BM/日志。当前Codex任务的loopback SSH bridge只是易失会话，重启后需重连；不要将其token当作持久记录。用户同时允许本地CloudDrive B:/experiment-archive、B:/tts-exp-project和公开网络数据用于研究。阿里云Qwen key仅由root在本地进程的DASHSCOPE_API_KEY环境中临时注入；模型qwen3-tts-vc-2026-01-22，100条fresh配对已生成。

## 运行环境 ✅

- GPU：当前只读查询为 RTX 4080 SUPER，32760 MiB总显存、32230 MiB空闲，无计算进程；新设备UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea。torch2.5.1+cu124/CUDA12.4前向合格属于迁移前记录，新设备的实际前向须重新检查。
- Wav2Lip解释器：/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python。
- SyncNet隔离env：/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/。当前只复制python_speech_features的pure-Python包到private deps/python_speech_features_only_attempt05/并引入该parent；不把SyncNet整套site-packages混入Wav2Lip，保持NumPy1.26.4。实际包版本/3源码文件/tree SHA见下方。
- 仓库：/root/autodl-tmp/wav2lip-syncnet/Wav2Lip 和 /root/autodl-tmp/wav2lip-syncnet/syncnet_python。
- 新实验隔离根：/root/autodl-tmp/tts-research-20260930/。未修改原有环境或模型代码。
- 正式输入：portable_source_axis_20260930/inputs/hf_selected；校准run：portable_source_axis_20260930/run/calibration_20260930。

## 核验与限制

官方例子重复生成两次，文件/像素/A/V距离矩阵一致；已知5帧延迟按预期改变offset与固定anchor评分。中文pilot的直接latent解码在primary face抽样帧与原生前向一致，源PCM float identity精确，官方SyncNet A/V特征一致、距离误差3.815e-6。工程证据在本地 runs/recovered_20260930/remote/control_summary.json 与 pilot_four_cell_20260930/pilot_summary.json。

librosa0.10.2的positional mel API由外置shim兼容；版本/默认值已绑定，不能宣称旧机器前端逐字一致。正式科学效果以run seal/summary和独立验证为准。主线协议见 openspec/changes/confirm-tts-source-mechanism-20260930/spec.md。

## Relations

- describes [[TTS 机制研究服务器 weste 12349]]
- supports [[TTS 来源方向与声学机制确认]]
## 私有输出根与容量 ✅

2026-10-01旧/root/autodl-tmp 50G数据盘满，后续run与control使用/root/tts-research-system-20261001/（0700），系统盘缓存搬迁后约18G可用。不要误写/root/autodl-tmp/tts-research-system-20261001/，公共/autodl-pub不作为项目存储。旧inputs和完成run保留原绝对路径只读。

cache-migration/verify_copy.json与cache_migration_complete.json记录下载缓存2.6G的完整路径/字节/SHA/权限核验；仅验证后的下载cache移动，原cache目录成为symlink，原conda环境和模型/科学资产不改。大型波形/JPEG/video任务启动前估算并核space/inodes，controls放在empty scientific output外。Qwen正式输入当前男性100+女性40=140成功，API key不保存。
## 网盘归档与输入保留 ✅

用户2026-10-01授权遇到容量问题时使用百度网盘归档。当前本机 B:/tts-exp-project/archives 可访问；对weste:12349会话只读核验未见fuse/dav/cifs/smb挂载、PATH客户端或常见root/opt/usr-local目录的BaiduPCS/CloudDrive/rclone/bypy/Alist，不据此断言其他实例不存在。可通过本机挂载中转已封存分片，不等服务端客户端部署。

科学归档采用immutable目标目录、逐包字节/SHA验证、成员索引、source_path mapping及恢复步骤。只有归档已核验且独立消费者完成读取后的副本才可释放；保留运行中的输入/模型环境/当次score caches，不把上传请求或本地挂载copy等同云端持久完成。新大临时下载先到${USER_HOME}/.codex/tts-exp-artifacts/20261001/，repo留索引，减少E盘重复数据。

主2560行的分析core301分片全部本地下载并逐包核验：1,587,388,518B，零missing/size/SHA错误；transfer manifest98350b737fcb40434da7fb653b9560869adaf4c9d85cc070f6c55939df47dd73，local verificationa0fce1e7e62d95d6ac15a70677a70c859d428b07e8f51046e5a6ae1a751172ce。位于runs/recovered_20260930/remote/evaluation_20260930_retry3_analysis_20261001/，未把下载包完整称成全成员解压。其不含原生视频/JPEG；native320另有完整C档案。

Aux05纯python_speech_features隔离目录/root/tts-research-system-20261001/deps/python_speech_features_only_attempt05/，实际3纯Python源码（旧13文件数含缓存）；source/copy treeSHA a419b997acfd21da8f4c7324e5be22e32357d085e37ac3376165e676c1e67c17，版本0.6。引入此pure包parent而不混整个SyncNet np2环境；实际Wav2Lip NumPy1.26.4核验。
## 2026-10-07 迁移核验 ✅

持久磁盘保留 `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003` 与训练08源码部署、CPU预检记录；实际完整SHA与迁移前一致。`/dev/shm/cross_tfg_audio_head_20261003` 整个临时命名空间不存在，fit40几何与自然音频编码缓存须在新GPU重新生成，不能沿用旧GPU的同设备资格。当前未发现08绑定或启动记录。持久输出盘空闲7,660,470,272字节。根代理实际只读证据：`.codex/scratch/remote_migration_20261007_root_attempt01/01_read_migrated_current_state.json.response.json`，SHA-256 `3913d11cad83e6d0f560b3d0d9df0f7106dde85d81bd606af507bc6e40527f42`。新密码只在私密运行时，不写入本笔记。

## 局域网 V100 研究机 ✅ 2026-10-09
用户提供并确认 `wjj@192.168.50.2:22`。现有公钥实际登录成功，BatchMode=yes与StrictHostKeyChecking=yes；密码/API永不入文件、argv或日志。主机 `wjj-Default-string`，Tesla V100-SXM2-16GB，UUID `GPU-d12b3309-c299-da83-064d-a3cd619bd560`，driver580.178.04。项目 `/home/wjj/tts-audio/tts-exp`，原dirty源码/autoavsr/mfa环境保留。

✅ W2L隔离runtime `/home/wjj/.venvs/w2l_syncnet_v100_attempt01/bin/python`，Python3.8.20/Torch2.1.2cu121，音频依赖/pipcheck/import/help/CUDA实际PASS，identity/RMS各两次真实生成/原音评分4/4工程PASS。155decoded/157crop/152queries的旧allTrue mask含1补帧query，正式资格按ADR023五帧全部decoded、源有理PTS及实际prewrite验证，四工程行不作候选效果。49输入6161313B已远端全byte/40原自然PCM/快照读回PASS。ASR负责正式40Fit×六方法240行以及GPU ownership。

✅ Muse/Ditto共享runtime `/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01`，Python3.10.20/Torch2.5.1cu121；真实V100 tinyCUDA PASS。35项精简hashlocked依赖已实际安装，35/35artifactSHA/version、pipcheck/import-only PASS，installreport533797B/SHAf46f86fb19b7ad2f2bbc5c403731ce510fe9a02cd687c662fea3c3960550eede。不同GPU/路径/包版本需fresh实际资格，不借旧cu124报告。

✅ MuseV1.5九固定assets3986182419B已实际下载+独立remoteSHA/tree PASS；Ditto Strict09 PyTorch12assets2314719638B已actual下载+worker全SHA读回PASS，Ditto receipt7737B/SHA867dc66c701899ee2163e17030a3254d53f2ae226b4fc23ba4babd9420b42687，ROOT实际receipt/manifest/source/worker核验PASS。位于新run `runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/model_assets/{musetalk_v15,ditto_v04_hubert}`。Ditto官方source新clean commitc3e47eee2e626500017a0556b470d6d4182f85e8；Muse108个源码/运行文件actual create-only stage及全entryreceipt核对PASS（expanded35351419B，无GPU），receipt32301B/SHA c636c73018cfd8d0e18a3a85f81a27d5b5b48fa836079ed78e2091e56e7c7d00。Signal负责Muse fresh source/runtime/render资格，Dev负责Ditto路径/Cython资格，模型资格尚未完成。

最新Muse源码staging后rootfree5427576832B，比5GiBreserve多58867712B，不能SSD重复zip/cache/160WAV/240media；source ZIP上传SHM。SHMfree16769675264B、最新hostRAMavailable30623191040B、cgroup无硬限已实际核验；每个重型stage/每行重测，按实际40音频时长推算、有界working media/归档，canonicalNPZ/ledger/clock/failure原分母持久保留，完整byte读回后才清理已验证temp。旧百度盘Ditto archive是Ampere_Plus TRT无.pth，不兼容这条V100 PyTorch路线。新Dev20和Confirm封闭，完整三模型720行未齐不得选择或更新。