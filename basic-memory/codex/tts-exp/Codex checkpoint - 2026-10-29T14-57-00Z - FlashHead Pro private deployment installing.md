---
title: Codex checkpoint - 2026-10-29T14-57-00Z - FlashHead Pro private deployment
  installing
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-29-t14-57-00-z-flash-head-pro-private-deployment-installing
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-29T14:57:00Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1880e-73d5-7863-a38b-20bcdf7d5ebc
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-29T14-57-00Z - FlashHead Pro private deployment installing

## Summary
用户已授权在个人目录部署 SoulX-FlashHead Pro 并生成测试视频，独立环境已建立，依赖与权重仍在后台安装下载，尚未完成推理验证。

## Story
公共盘接近满载后，用户明确选择 /home/data2/wangjiajun 个人目录，要求遵守服务器 skill 和手册。用户最新更新了 server-192 skill，要求使用前先读取飞书手册；已重新读取新版 skill 并成功打开《1048服务器手册》，确认个人数据区、共用 Anaconda、tmux/nohup 运行与下载前检查公共 checkpoint 的规定。没有改系统、共享 conda base 或其他用户进程。

部署根目录是 /home/data2/wangjiajun/SoulX-FlashHead-Pro。使用共享 /home/user/anaconda3/bin/conda 创建 envs/flashhead Python3.10。官方代码 Git clone 网络挂起，仅停止自己的克隆进程，改用官方 GitHub main ZIP 经 SFTP 上传到 repo。官方安装要求 torch2.7.1+cu128；后台安装中。模型约6GB，HF镜像太慢，改用官方同名 ModelScope 仓库并准备按 SHA256 核验，速度约10MiB/s。公共 /home/data/public/model 有限深度检查未发现 FlashHead/SoulX/wav2vec 权重。

## Working State
持久远端入口为部署根目录下 install.sh、install_dependencies.sh、download_models.py、logs/。机器本地脆弱状态：C:/Users/dogle/.codex/tmp/server192_run.py 是 Paramiko 运行器，从本地全局 BM 的 Windows DPAPI 密文解密登录，固定先前观察到的主机指纹，不输出凭证。它支持脚本 stdin、--upload、--download；脚本输入归一化换行。

2026-10-29 22:55服务器快照：torch pip PID507148（父507147），依赖等待器515714，下载器517685（aria子517730）；重连后先检查进程与日志，不重复启动。Model_Pro safetensors 36%，尚无完整下载 receipt。服务器8张GPU均有别人的进程，推理前重新确认空余，以UUID指定一张A6000并限制CPU线程和显存。系统驱动550.78/CUDA12.1，先测试cu128 wheel能否运行。上游支持原生SDPA回退；可用包装器关闭COMPILE_MODEL/COMPILE_VAE，避免编译与老驱动PTX问题，不需要更改驱动或编译FlashAttention。

依赖脚本使用唯一 opencv-contrib-python4.11、numpy1.26，避免上游同时安装多个OpenCV。xfuser0.4.3与xformers0.0.31用--no-deps；需在完成后检查缺失元数据依赖和模型导入。尚无测试包装器与视频。repo原有大量修改/未跟踪文件，本任务未修改项目源代码；勿清理。

## Verification
SSH重新登录成功，账号wangjiajun、主机user。个人所在盘余量2.7T。手册可读；新版skill未被本轮覆盖。Python3.10环境创建成功；torch安装与权重下载未完成。没有运行GPU推理，不能称部署可用。git status/branch/SHA已读；gh不在PATH，上一checkpoint已有只读GitHub查询确认仓库且无PR，本轮未复查PR。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)，当前本地SHA `2b71639c3b555293375d248350999377168d515d`，未验证GitHub是否可解析。
- [模型官方代码](https://github.com/Soul-AILab/SoulX-FlashHead)
- [服务器手册](https://gcnadw6o1ien.feishu.cn/wiki/Vq5gwxt51i2DVdkDhobcP4iqnkh?fromScene=spaceOverview)
- 官方ModelScope revision 49f773fd58e6403b4168257d59ef1e48fe56c3a1；HF revision 5fbd2e44ba08ab2deb28bd55cff2c15523e51ba5。下载完成将写 model_receipt.json。

## Observations
- [decision] 所有代码、环境、缓存、权重与输出放个人数据目录，遵从用户选择并避免公共盘耗尽。
- [result] 服务器连接、手册读取和独立环境创建已验证；权重与依赖仍在安装中。
- [blocker] 尚待下载完成及验证torch cu128在550驱动上的运行兼容性，尚无生成视频。
- [next_step] 检查三个后台任务日志，补齐依赖并核验权重，使用官方图片和截取2.24秒音频在单GPU执行Pro推理，再用ffprobe确认音视频流。

## Relations
- continues [[Codex checkpoint - 2026-10-29T14-43-00Z - 192 server access verified and FlashHead storage choice]]
