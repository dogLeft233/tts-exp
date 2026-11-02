---
title: Codex checkpoint - 2026-10-18T10-48-35Z - Patrick 27-grid delivered and Chris
  FP16 training started
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t10-48-35-z-patrick-27-grid-delivered-and-chris-fp16-training-started
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 10:48:35.606089+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6-astra
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-18T10-48-35Z - Patrick 27-grid delivered and Chris FP16 training started

## Summary
Patrick单人粗训练27格已独立核验并交付，安全评分边界已释放，Chris全监督恢复后真实开始FP16训练。

## Story
原目标是四身份两seed三训练档的24完整模型及三轴27格。Brown原FP16失败后，用户要求保持FP16、另立更新前保护并查上游溢出，随后先试Patrick/Chris/Cathy，并要求首人完成后先看分数。独立trial保持原数值政策，每人seed42三档各coarse10，fine/seed2026未自动启动。Patrick三档各60000attempts完成，实际成功−6/0/+6为59985/59985/59984，原CONTROL_FAILED保留；未因分数补步或挑checkpoint。

early_score_preview_v2固定coarse10 EMA和原六段test，完成9视频27格；独立数值核验与持久交付后，10:39:55UTC一次写入V3 preview_delivery_release.json，绑定delivery receipt SHA9533b56022e7436774bdcb4d8e80c3d80144d477e03919644b569c1b09ee7f92。未重启controller或中断GPU。交接容量预检差约6MB，鲜核9份F评分归档225成员后，仅通过Linux独占lease退休216个remote评分npy副本，F原数组/归档及remote scores.json保留。随后原controller正常退休Patrick剩余2976非control并全恢复Chris。

实际交接已核验：Chris完整监督16500文件恢复并逐字节验证通过（16428恢复、72复用）；Windows181520状态training，remote job805176/child805186/training805187实际运行seed42/0dB/coarse ep1。一次实时trace前缀global1520、1520成功、0skip、loss有限；仅前缀，不是epoch完成或完整稳定性证明。Patrick剩余2976非control test缓存已由原controller按F全验与独占lease退休，现72controls保留；以后加载必须显式恢复。Cathy仍未启动。

最新用户询问单人初步结论，要求结果最多三句、结论最多三句易懂表达。应说明固定driver/scorer0时训练−6/0/+6 C为8.213/8.068/7.770，九个同驱动同评分条件排序一致；驱动响应取决于模型，单改评分音量也改分。只描述实际模型，不宣称低音量训练因果、最佳音量或感知改善。

## Working State
FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017；OUT=FROOT/trials/fp16_other_identities_trial_v1；ROOT=/dev/shm/synctalk-multispeaker-data-20261017；SSH wjj@192.168.50.2。PID/SHM/dirty工作树是本地易变状态，真实receipt/ps优先。Windows181520/V3继续串行Chris后Cathy，preview176764正常退出不重启。controller_failure.json不存在，failures[]。本轮df SHM240513024B、root773312512B可用，不将容量波动归因未执行清理。

权威结果OUT/early_score_preview_v2/delivery_v1/report.md、scores_27_cells.csv、mean_distance_curves.csv、patrick_27_grid.png、independent_review.json、delivery_receipt.json；all27_preview.json SHA0c727888b1fbeb09e573ab26bc7129d453aa03b45de345b460d66973ac6fac1a；independent_review SHA043bc7d311f5f7193f2f7fe6719f52520db095e0640b6a54f3e46638d735eed4。9原lossless媒体在cloud locators，F重复tar/remote媒体已按全验退休；F保留H264/PNG/head/wav/评分原矩阵和归档，读取lossless须显式恢复。评分216个remote数组也须从F显式恢复。release已写不可重复，delivery_receipt内release_writtenfalse是写release前不可变快照，不应修改。

Chris full_restore archiveSHA1092413d7832964738f13d6012c44de8ce8ca79638854ccf78b7e5b2a4006f00，manifestSHA15462d0c93d2e298ed7162b88a05ff7fb3c28166f3862cbb91c5b1533382afa3。冻结trial contractSHA787d917d259eeca89311016737893348578335e3623f217858797926311893bd，V3源SHAd0ad60214e0c4d36193505c7aeb6b1b507b9838cb1d2001a1eb89a0f98f6b712。保持FP16/defaultGradScaler、原loss/AdamW/EMA/65536rays/maxsteps16/共同初始化；首次skip和terminal完整state捕获不变，lost不重放。Brown原archiveSHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72、旧global false和原版本保留；105372/133984/610686不重启。guard不等于上游eye/sigma增长根因完全修复。

## Changed Files
已交付阶段新增.codex/scratch/final_review_patrick_preview.py、plot_patrick_preview.py、deliver_patrick_preview.py与F delivery_v1；Task/ADR046/研究地图及BM结果报告已更新并读回。本轮仅只读确认Chris启动、更新Task当前头部和写此checkpoint，未改训练源或重新跑GPU。checkpoint辅助脚本.codex/scratch/checkpoint_patrick_delivery_chris_started.py是本地未提交工具。git status当前1044项dirty，包含大量既有工作，未批量stage/clean。

## Verification
已完成最终复核：9评分归档225成员freshSHA/local一致，另一按列NumPy float64实现重算162个245×31矩阵，maxmatrix5.241944709410973e-06、maxendpoint2.782323459626923e-06<3e-5；全部27格六clip等权/offset、27PNG与rawhead、9PCM核验通过。全部1500frame FFV1/PCM逐像素及独立Linux云整档读回是既有执行receipt，最终review未再次恢复全部电影/重读云payload或重跑GPU。图表首次缺matplotlib失败保留，随后仅用已有隔离绘图库作图并目视检查，无共享环境安装。最终交付33文件manifest读回通过。

本轮实际ps证明805176、805186、805187连续运行，execution.json state=running；attempts.jsonl实时global1520成功1520/0skip，不能外推以后。误读trace.jsonl路径返回不存在，随后目录清单定位真实attempts.jsonl，不把路径错误当训练失败。Chris恢复passed证明16500文件，receiver training_admittedfalse仅该恢复receipt范围，不撤销原数据资格。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，本轮GitHub仓库身份已核验；当前分支无open PR。
- HEAD `2b71639c3b555293375d248350999377168d515d`为本地核验，未验证GitHub commit可解析，不构造commit链接。

## Observations
- [result] Patrick完整27格粗阶段描述预览已交付；Chris完整监督恢复后真实开始FP16 ep1，Cathy待运行。
- [decision] 保留原CONTROL_FAILED、Brown失败与所有旧版本，三轴分开解释，不以预览选条件或加预算。
- [blocker] 单人单seed粗训练且实际更新数不等，不能形成训练gain受控因果结论；上游溢出根因及原24完整模型仍未完成。
- [next_step] 跟随V3真实receipts完成Chris/Cathy已授权FP16粗训练及异常封存，正常运行静默，不重放已完成或失联job。

## Relations
- continues [[Codex checkpoint - 2026-10-18T10-08-50Z - Patrick coarse first score table independently verified]]
- tracks [[试跑 Patrick Chris Cathy 的 FP16 粗训练]]
- relates_to [[SyncTalk Patrick coarse10 seed42 27格描述性预览]]
- relates_to [[046-synctalk-first-identity-early-score-preview]]