---
title: Codex checkpoint - 2026-10-17T02-34-56Z - SyncTalk gain inference verified
  and normalized comparison scoped
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t02-34-56-z-sync-talk-gain-inference-verified-and-normalized-comparison-scoped
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17 02:34:56+00:00
username: laptop-e133gsle\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-17T02-34-56Z - SyncTalk gain inference verified and normalized comparison scoped

## Summary
官方AVE May模型的纯幅度干预已产生实际视频与固定原音轨SyncNet响应；官方HuBERT近似抵消幅度，但尚无匹配生成器权重验证视频响应。

## Story
用户希望检验输入音量是否通过TFG改变生成和同步评分，先用现成May checkpoint跑−6/0/+6dB，统一原始自然音频评分，明确暂不训练；空间不足可想办法解决。随后要求核实原装编码器是否真的不吃增益，追问官方归一化后的Sync-C是否不变，并授权尝试。最新请求是解释SyncTalk官方版本、推荐适合本实验的版本和具体设计，尚未授权恢复训练。

工作按编码前处理→真实编码→已训练条件入口→视频→固定音轨评分逐层推进。报告47验证AVE保留增益、HuBERT整段归一化近似抵消、DeepSpeech仅前端有响应；报告48使用原版AVE May实际生成四组视频；报告49复用同May三档全段driver严格加载官方HuBERT，完整前向两次并抽取同12秒区域。未把微小编码变化当作视频或分数不变的证据。

现阶段推荐沿用固定官方SyncTalk commit的AVE+May作为幅度主实验。显式统一RMS后再进同一AVE/May可作为消除电平差的额外推理干预，但这不是官方AVE已有归一化，也不能称为HuBERT生成结果。HuBERT生成对照需取得匹配的HuBERT-trained checkpoint；否则只有将来单独授权训练才能闭合，不能随机适配1024维到512维。训练电平×测试电平矩阵属于后续方案，当前0训练。

## Working State
持久证据以正式报告47、48、49及对应本机runs为准。官方SyncTalk固定commit为43079fb3f349675f6cb33031bb07262f7566cdd5；May ngp.pth SHA256为79e5c0056702357e176e9895b237bfcab1a517b85f4e468586282954e7379e79。HuBERT模型facebook/hubert-large-ls960-ft revision ece5fabbf034c1073acae96d5401b25be96709d8，权重SHA256为9cf43abec3f0410ad6854afa4d376c69ccb364b48ddddfd25c4c5aa16398eab0。

本机runs/synctalk_amplitude_audit_20261016保存完整May.zip、trial_may.zip、HuBERT和前端原数组；runs/synctalk_may_gain_inference_20261016保存4个正式FFV1/PCM视频、评分特征/矩阵/共享ROI、独立QA和三联预览；runs/synctalk_may_hubert_gain_20261016保存18个HuBERT阶段数组、同12秒数组及独立公式复核。三轮完整归档均远端/本机SHA一致。百度盘写缓存占根盘，上传未成功，绝不称为云备份；没有删除历史实验。

Ubuntu执行机wjj@192.168.50.2，项目/home/wjj/tts-audio/tts-exp。本轮2026-10-17只读复查显示根盘约1.1GB空闲，/dev/shm 16GB全空，之前SHM runtime已经消失；持久原数组仍在本机，任何新渲染需要恢复隔离CUDA12.1/GCC12/deps工作环境。现有root官方checkout、两个May checkpoint可读。SSH使用已验证key和BatchMode，无密码写入。共享torch环境禁止改动。

当前分支为既有codex/cross-tfg-audio-head-20261003，工作树有9条修改和1009条未跟踪条目，绝大部分与本任务无关；本任务代码和测试仍未跟踪，未提交/推送/创建PR。GitHub只读API核对仓库为dogLeft233/tts-exp，当前分支PR查询为空；当前SHA的GitHub解析返回422，因此保留本地SHA，不构造已推送commit链接。不得全量清理、stage或commit。

## Changed Files
- scripts/experiments/synctalk_amplitude_audit_20261016/：真实前端审计、严格HuBERT加载、May条件入口、下载与独立验证。
- scripts/experiments/synctalk_may_gain_inference_20261016/：隔离部署/渲染/固定PCM评分/独立验证/同May HuBERT source-clock分析；官方原checkout不改，工作副本仅4个CUDA backend构建标志调整为C++17并使用隔离GCC12。
- tests/experiments/test_synctalk_amplitude_audit_20261016.py、test_synctalk_may_gain_inference_20261016.py、test_hubert_may_source_clock_20261016.py。
- basic-memory/docs/experiments/47-synctalk-amplitude-frontend.md、48-synctalk-may-gain-inference.md、49-may-hubert-gain.md，以及研究主线地图和实验README索引。

## Verification
报告47：5真实来源、13安全条件、301原数组/299比较独立复核通过；AVE编码相对L2变化8.783%–14.628%，May音频MLP/注意力后11.993%–18.643%，重复误差0。DeepSpeech FLOAT WAV直接cast int16全零属于错误输入，未冒充科学阴性；TF完整图尚未跑。HuBERT最初自动加载遗漏旧weight_norm参数的attempt02作废，canonical attempt03严格映射并加载全部参数，有效位置卷积误差0。

报告48：官方May验证帧5520–5819共300帧/12秒，−6/0/0重复/+6四臂。原始PCM SHA256为68ffa2de3aa08083933670a54ca6471a11d293d90de4c64a702d73143903ada9，三档驱动+6峰值约0.949，无削波。共享0dB人脸轨迹；lossless评分合同；主guard20 Sync-C为8.906/9.262/9.262/9.373，Sync-D为5.927/5.791/5.791/5.752，offset均−3帧。−6/+6的ΔC为−0.356/+0.111。0重复浮点图像、RGB、矩阵逐位相同；1200帧和36580距离独立重算通过。视频有口部像素响应，不能称为感知改善；只有单身份片段，无总体推断。

报告49：官方整段do_normalize=true先于20秒分块；每档两次完整前向共6次，18数组/epsilon公式/相同时钟切片独立复核通过。同12秒HuBERT hidden相对L2为−6dB 0.009302%、+6dB 0.009009%，重复误差0；不是严格零变化。0新视频/0SyncNet。

对应unittest分别1个损坏拒绝、3个评分合同、1个source-clock检查通过；完整验证在Ubuntu运行。FP16/H264预览不作为正式评分；full边界敏感性不替代主guard20。CUDA编译失败和评分cwd失败保留，未覆盖canonical结果。

本轮官方可用性只读核查：README只公开AVE May现成生成器，两份实际pth首层均[256,512]；官方HF仓库只有SyncTalk-Windows.zip。HTTP Range读取其ZIP64目录（33.6MB，未下载5.67GB整包），仅发现同两个May生成器checkpoint，未找到HuBERT生成器。初次4MB尾部不足中央目录的解析失败已纠正；成功库存位于.codex/scratch/synctalk_windows_zip_inventory.json。SyncTalk_2D当前inference_328.py硬走AVE提取，公开树未见May身份生成器checkpoint；SyncTalk++项目页Code仍指向原SyncTalk，作者公开repo列表只有SyncTalk和SyncTalk_2D，尚未确认独立++源码/权重。未据此断言所有外部位置绝无模型。

## References
- [当前项目仓库](https://github.com/dogLeft233/tts-exp)
- [官方SyncTalk](https://github.com/ZiqiaoPeng/SyncTalk)
- [官方SyncTalk_2D](https://github.com/ZiqiaoPeng/SyncTalk_2D)
- [官方SyncTalk++项目](https://ziqiaopeng.github.io/synctalk++/)
- [官方整合包仓库](https://huggingface.co/ZiqiaoPeng/SyncTalk/tree/main)

## Observations
- [result] AVE May对纯幅度干预存在可复核的视频和固定原音轨SyncNet响应；HuBERT表示近似抵消但不严格不变。
- [decision] 保留用户不训练约束，评分使用相同原自然PCM和原时钟；不以涨Sync-C代替视觉/感知改善。
- [decision] 主实验继续固定原版AVE+May；显式波形归一化是单列干预，HuBERT更换需匹配已训练生成器。
- [blocker] 未取得匹配HuBERT的SyncTalk生成器checkpoint，不能测其视频和Sync-C；现有May不可直接替换编码器。
- [blocker] SHM执行环境已清空，未来执行需从持久资源恢复；服务器根盘空间仍紧张。
- [next_step] 先向用户解释官方版本/编码器区别和分阶段设计；若继续当前不训练实验，恢复隔离runtime，冻结AVE显式归一化对照并保留原PCM评分，HuBERT生成层等待匹配checkpoint。

## Relations
- relates_to [[47-synctalk-amplitude-frontend]]
- relates_to [[48-synctalk-may-gain-inference]]
- relates_to [[49-may-hubert-gain]]
- relates_to [[研究主线与证据地图]]