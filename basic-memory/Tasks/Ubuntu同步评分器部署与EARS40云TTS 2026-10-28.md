---
title: Ubuntu同步评分器部署与EARS40云TTS 2026-10-28
type: note
permalink: tts-exp/tasks/ubuntu-同步评分器部署与-ears40-云-tts-2026-10-28
status: complete
tts_status: COMPLETE
deployment_status: MTD_AND_SYNCHFORMER_GPU_PASS_ORIGINAL_VOCALIST_PENDING_WEIGHTS
---

# Ubuntu同步评分器部署与EARS40云TTS 2026-10-28

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

用户授权Ubuntu部署VocaLiST与Synchformer，并提供阿里运行时key调用qwen-audio-3.1-tts-plus；随后明确同意同时部署公开权重的MTDVocaLiST，原版待补权重。密钥不写本笔记、代码或日志。

## 已完成
Ubuntu /home/wjj/tts-audio/tts-exp/runs/ears40_cloud_qwen31_20261028_attempt01 已40/40 COMPLETE，同步完整小run到./runs/同名目录。40位EARS说话人各ref sentences_15_regular、target sentences_16_regular；英语词间空格保留，独立参考非同句。官方voice-enrollment/create_voice+SpeechSynthesizer，enable_preprocess=false、enable_volume_normalization='false'、max_prompt_audio_length30。每人一次成功合成；p063首次明确429 Throttling.RateQuota被拒绝，原拒绝收据改名保留，退避10秒后补成功，39已成功不重抽。provider原始WAV40、natural16 40、tts16 40，合成总438.32秒，范围8.32–14.32秒。未归一化RMS、无时钟编辑、无TFG或正式评分。
新增scripts/experiments/ears_cloud_tts_20261028.py与tests/test_ears_cloud_tts_20261028.py，未提交。远端无pytest；两测试函数直接运行PASS（英语空格与等时长/非RMS重采样）。NodeREPL spawn SSH exit255无远端进程/调用，后改PowerShell变量经SSH stdin成功；不是API失败。运行时key从stdin传入，脚本未保存Authorization或key。

## 模型部署状态与恢复
根/home/wjj/av-sync-eval-20261028。

VocaLiST官方源码git c265b59bcf7cd8559a265aaf454a9333ec24cb0f，env /home/wjj/.venvs/vocalist-20261028。Google Drive匿名目录空、gdown列举[]，可信原版权重仍不可得；用户同意原版待补权重，蒸馏版单独部署。原版不能称带权运行完成。

✅ MTDVocaLiST作者commit5795416f9277094d3ae6215596a436a0a01f79ad，官方release权重完整53278751字节，严格加载13280961参数、真实媒体GPU前向及integration PASS。raw logit3.359825849533081，3.70秒，峰值9269863936字节（包含S3FD batch1）。详见[[MTDVocaLiST部署]]。

✅ Synchformer作者commit b66668a1521d7567cc760e5544b2b5b53179b687，Synchformer-source/固定124推理文件。LRS3官方权重23-12-23T18-33-57.pt完整1131113925字节、135/135分片，session78099结束0。严格加载237460245参数、真实5秒自然音视频GPU前向PASS，21 logits有限，预测offset0秒，耗时7.80秒，峰值1928923136字节。env /home/wjj/.venvs/synchformer-20261028。详见[[Synchformer部署]]。

实际smoke receipts在smoke/{mtd,synchformer}_receipt.json，Windows复制在${USER_HOME}/.codex/tts-exp-artifacts/av-sync-deploy-20261028/。两条artifact integration测试PASS。脚本scripts/experiments/av_sync_deployment_smoke_20261028.py与对应test均在两机当前项目。range_download.py下载按HTTP Range与长度续传，无额外SHA pass。原Synchformer/.git残留不是入口。
## 环境与兼容
每模型独立venv，重型Torch2.5.1/cu121等以.pth只读复用现有环境site-packages，不修改旧环境。Synchformer新增timm.6.12、av12.3、ftfy6.1.1、decord.6、Pillow10.4。默认继承MFA的Pillow12.3触发libtiff/jpeg12 ABI错误，安装独立PyPI Pillow10.4修复；MTD环境从已安装轮子的PIL/pillow.libs/dist-info复制，避免重复网络下载。误赶temp轮子保留窗口导致离线pip路径不存在尝试失败，未改旧环境；真正online pip随后完成。Synchformer已由CPU初始化推进到完整真实checkpoint严格加载与GPU前向PASS（237460245参数）。pip check另有继承的sacrebleu/portalocker版本冲突，未改旧环境，实际推理通过。PyAV12.3不同官方复现实验8/9，需标注并后续媒体decode/parity验证；不能宣称官方bit-exact。
Ubuntu GPU V10016GB，开始630MiB/0%；不关闭他人ComfyUI、不恢复旧暂停训练。部署前磁盘14GiB，现约13GiB可用，留意各自稀疏权重的实际占用。Ubuntu已有127.0.0.1:7890代理读取测试更慢，未改网络配置、未切换下载；Windows7897仅localhost，不扩大LAN权限。

## 本次交付与后续边界

本次已完成40/40英文云TTS、MTDVocaLiST与Synchformer的官方权重部署/真实GPU前向/实际receipt integration。按用户选择，原版VocaLiST保留等待权重，未冒充原版评分。

Synchformer媒体适配器使用pcm_f32le MOV，避免官方reader对Short求mean；实际解码80000采样点与原前5秒最大差0，未归一化/时移。官方特征变换未改。PyAV12.3与原实验老版本不同，后续正式评分先做decode/parity与时移控制。

尚未执行40人TFG生成、新cohort正式评分或同步科学有效性控制；本次不据单例smoke推断TTS优势。旧生产训练与交叉音轨任务继续暂停。原版VocaLiST权重以后补。