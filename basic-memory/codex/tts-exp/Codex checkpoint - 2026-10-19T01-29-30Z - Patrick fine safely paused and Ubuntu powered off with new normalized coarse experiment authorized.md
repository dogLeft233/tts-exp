---
title: Codex checkpoint - 2026-10-19T01-29-30Z - Patrick fine safely paused and Ubuntu
  powered off with new normalized coarse experiment authorized
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t01-29-30-z-patrick-fine-safely-paused-and-ubuntu-powered-off-with-new-normalized-coarse-experiment-authorized
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-19T01:29:30.757800+00:00'
username: dogle
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

# Codex checkpoint - 2026-10-19T01-29-30Z - Patrick fine safely paused and Ubuntu powered off with new normalized coarse experiment authorized

## Summary
Patrick原AVE三档fine已完整暂停在第14轮并持久保存，Ubuntu随后关闭；用户现另授权加入音量归一化入口的新Patrick三档粗训练及完成后评测，尚未开始该新实验GPU任务。

## Story
原目标是四身份、两seed、三训练gain完整实验；Patrick coarse10 seed42及描述性27格已经交付，后按用户要求暂停Chris、先跑Patrick fine11–17。用户随后明确要求先暂停训练并关Ubuntu。因此停止原Windows调度器，让已在运行的+6 fine14正常rc0完成，由只封存的边界观察器保存完整RAW state、job与trace；没有发GPU信号、重放optimizer或启动fine15。

三档fine14/global84000在F全验后执行一次sudo shutdown -P，2026-10-18T15:15:22Z返回rc0，15:16:20Z计划断电。SSH明确进入shutdown/PAM nologin，直连Ethernet index6随后Disconnected/0bps；Meta代理的TCP/ICMP曾产生假在线，保留失败探针，不重复下关机命令。不声称有物理电源传感器。

最新用户于本轮授权“用加入音量归一化的音频编码器运行Patrick的粗训练，还是三档音量，完成后评测”。这是另立实验的授权，不是恢复旧fine/Chris/Cathy。当前已直接SSH确认Ubuntu新boot id a257ef30-7bdc-406b-aaa1-891b50b0dacf、/dev/shm 16G全空、V100可用、根盘约615MiB余量。下一步沿原AVE路径审计归一化插入位置，冻结新合同并恢复隔离runtime/持久数据，不复用旧未归一化音频特征。

## Working State
权威旧暂停证据为FROOT/trials/patrick_fp16_fine_v3/user_pause_v1/paused.json、fine_ep0014_boundary_qualification.json、poweroff_preflight.json及poweroff_receipt.json；FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017。旧Windows controller177316和capture observer182424已停止，heartbeat synctalk-27 PAUSED。SHM运行副本已因重启消失，F完整原字节保留；任何新运行须恢复并逐SHA核验runtime/data/core。旧fine重启须另受用户明确授权，不能重启旧controller或重放第14轮。

Patrick gain0/−6/+6累计成功83964/83962/83962，各epoch14/global84000。最新F full RAW checkpoint SHA依次d3cc8d4fb4f5cef78e66f0dd027d875a7d8ac0547c8a25ca726ce74f95ca3aa3、380f8aea8f9220b584e5623b787915152a7a621fff22d60b24eefcaf501399f2、cb0ee19ff978120fc1c8ba64b330feaf6cee229b105e1ca70341c962342fd369。fine11–14共12/21 epochs，fine17及新27未完成。Chris仍0/−6 coarse2和+6 coarse1，Cathy未启动。

原fine contract SHA7d6276b92c406a8496de8f230c66db0eed37351fec259ca8af7c8c3b7ed073b9与7冻结源不改。原coarse actual0/−6/+6=59985/59985/59984，CONTROL_FAILED不因fine逐epoch相等而消失。新归一化实验应从共同完整随机初始化独立训练，seed42/−6,0,+6/6000 attempts×coarse10，保留FP16/defaultGradScaler/guard、loss/AdamW/EMA和原数据/采样/评分合同，不从旧coarse或fine warmstart，不补步。此为当前拟沿用设置，归一化精确定义/新冻结合同尚待实现资格。

## Changed Files
暂停/关机实现与证据：scripts/experiments/synctalk_multispeaker_full_20261017/pause_patrick_fine_boundary_v1.py及F OUT/user_pause_v1/；独立闭合epoch复核源在OUT/independent_reviews/，block14使用user_pause_v1中的实际boundary文件（旧controller没写normal blocks/fine_ep0014.json）。Task《暂停 Chris 并精调 Patrick 三档及27格》已回读status paused/当前头部关机证据。scratch脚本与该实验源码为本机未跟踪文件，尚未提交；不批量stage/清理其他实验。

细粒度准备仍保留：fine_score_candidate_revision02的score SHA2502c1e843eedeb1e9bd568936784f79d4361fd2f95637412c5d10aed2977711和helper SHA34cdfac45e178d75c6697175cffb13f2f35b100032055ade6e50fa8cc5fb4950，未真实fine17渲染/部署。粗fine比较候选source SHA4033036f8fd26bdbe2fe6b2e68912f4bbbf6a1a93082322f3cfaf0cf48fe94a8仅CPU资格，无fine实验输入。新归一化粗训练不冒用这些fine资格。

## Verification
暂停时三档fine11–14全部闭合epoch独立archive全成员、当时standalone完整checkpoint SHA、6000 trace逐行/finite loss核验通过；每个epoch18000跨gain frame/pixel/rays/lip/rect/LR配对相同。fine11成功5988/5986/5987，fine12–14均5997；没有GPU重算或全tensor重载。旧已轮转中间cp不要求再次物理存在，不覆盖原review。poweroff_preflight再次fresh核7源、Patrick fine14/coarse10及Chris最新共9个完整cp、Brown原失败archive和原coarse27。失败的缺normal block14、GNOME桌面GPU谓词、PAM探测和代理在线探测均保留，不冒称训练失败。

本轮直接复读paused.json与poweroff_receipt.json均passed，已读Startup Router、地图、实验/任务规则及coding schema，验证同chat前序身份。Git直接读到branch codex/cross-tfg-audio-head-20261003，本地HEAD 2b71639c3b555293375d248350999377168d515d，status --short共1053条（大量其他任务文件）。GitHub只读REST确认canonical仓库，当前branch PR为空，HEAD远端无法解析，不构造commit链接。新归一化实验尚无GPU前向/训练/评测结果。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)
- 本地未解析到GitHub的HEAD：`2b71639c3b555293375d248350999377168d515d`。
- F OUT/user_pause_v1/paused.json、poweroff_preflight.json、poweroff_receipt.json。
- 原Brown失败archive SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72与原coarse27 SHA0c727888b1fbeb09e573ab26bc7129d453aa03b45de345b460d66973ac6fac1a保持。

## Observations
- [result] 原Patrick三档已安全暂停在fine14，完整断点与失败/coarse产物F保留；Ubuntu关机已接受并直连断开，现已新boot开机。
- [decision] 最新授权独立加入音量归一化的Patrick coarse三档及完成后评测；旧fine/Chris/Cathy继续暂停，不继承旧controller调度。
- [decision] 不存凭据，不改变既有训练数值政策，不把SyncNet变化等同感知或训练gain因果；原CONTROL_FAILED保留。
- [blocker] 新归一化入口定义与真实资格、SHM环境恢复和新合同尚未完成，0新实验分数。
- [next_step] 确定并冻结同AVE归一化入口，恢复隔离运行环境和Patrick原数据，通过真实入口与跨gain采样资格后启动独立seed42三档coarse10及27格评测。

## Relations
- continues [[Codex checkpoint - 2026-10-18T12-08-08Z - Chris safely paused and Patrick FP16 fine entry qualified with training running]]
- relates_to [[暂停 Chris 并精调 Patrick 三档及27格]]
- relates_to [[047-synctalk-patrick-fp16-fine-and-safe-pause]]
- relates_to [[研究主线与证据地图]]
