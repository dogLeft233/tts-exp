---
title: 执行 Exp 31 Week 1 特征诊断 (CKA + 线性探针)
type: task
permalink: tts-exp/tasks/执行-exp-31-week-1-特征诊断-cka-线性探针
status: in-progress
tags:
- task
- experiment
- replacement
---

# 执行 Exp 31 Week 1 特征诊断 (CKA + 线性探针)

- Status: in-progress
- Date: August 31, 2026


## Observations

- [status] in-progress

## 目标

按 `docs/specs/replacement-salvage-protocol.md` Week 1 执行:
1. Step 1.1: 提取 40 records (exp_a train 20 / val 8 / test 12) 的 WavLM L6 特征 (natural/tts 对齐切片)
2. Step 1.2: 逐 record CKA 相似度 → mean < 0.7 才 PROCEED
3. Step 1.3: LogisticRegression 线性探针 → val_acc > 0.7 才 PROCEED
4. Step 1.4: (回顾性) Exp 30 TTS-kv adapter 注意力可视化
5. 输出: runs/diagnostic_week1/{cka_summary,probe_summary,final_decision}.json

## 环境

- worktree: `.claude/worktrees/lrs3-wavlm-resynthesis-50/` (Exp 30 代码)
- python: `/home/wjj/miniconda3/bin/python` (torch 2.13 cuda + sklearn 1.9 + soundfile + textgrid)
- 模型: torch.hub 缓存 WavLM-Large.pt + prematch_g_02500000.pt (已有)
- 数据: `tmp/lrs3_policy_a1_200_20260828/policy_cohort/` (280 records, 96-frame 对齐, span=61440 samples)

## 决策门

- mean_cka < 0.7 AND val_acc > 0.7 → PROCEED_TO_WEEK2
- mean_cka ≥ 0.9 → STOP_HIGH_CKA (特征冗余, pivot 到 duration control)
- val_acc < 0.6 → STOP_LOW_SEPARABILITY

## 进度
## 进度 (2026-08-31 13:50 更新)

### Week 1 ✅ 完成 → PROCEED_TO_WEEK2
- CKA: mean=0.3828 (std 0.050), train/val/test 三组一致 (0.379/0.395/0.381)
- 线性探针: train_acc=1.0, val_acc=0.9411, val_auc=0.9871
- 注意力可视化 (Exp 30 TTS-kv step20): mean_row_entropy=0.9978 ≈ 均匀, 无对角线/块状结构
- 结论: TTS 特征与 Natural 高度可分 → 帧级 cross-attention 失败源于对齐/利用方式而非特征冗余
- 产物: `tmp/runs/diagnostic_week1/{features,cka_summary.json,probe_summary.json,final_decision.json}`

### Week 2 准备 ✅ (代码全部就绪, 10 个单元测试通过)
- **Step 2.1**: MFA3 phone alignment 完成, 40/40 records 通过 natural-span 前缀检查
  - 修正: natural 训练切片=完整音频前 3.84s → phone bounds 裁剪到 span; MFA3 空 label 保留为 spn; time_to_frame 单调 clamp; 边界 collapse 跳过
  - 产物: `tmp/runs/diagnostic_week1/week2_mfa/{corpus,alignments,phone_alignments,mfa_summary.json}`
- **Step 2.2+2.3**: `phone_replacement_model.py` — PhonePooling (mean) + PhoneCrossAttentionAdapter (零初始化恒等验证通过)
- **Step 2.4**: `week2_train_phone.py` — 三臂训练 (own_tts/own_natural/shuffled), 复用 Exp 30 loss/optimizer/checkpoint 计划
- **Step 2.5**: `week2_validate_phone.py` — 复用配对门 + val 选择
- **Step 2.6**: `week2_test_phone.py` — 官方渲染 + 严格配对门 + SALVAGED/ABANDONED 判定
- 测试: `tests/experiments/lrs3_syncnet_finetune/test_phone_replacement_model.py` (10 passed)

### ⚠️ GPU 硬件掉线 (2026-08-31 12:41)
- `Xid 154, GPU has fallen off the bus` (V100 SXM2 16GB) → 需要 root `nvidia-smi --gpu-reset` 或重启; 本机无免密 sudo, 远程服务器全 refused
- **决策: 降级 CPU 训练** — 32 核, 三臂并行后台运行中 (nohup + /tmp/week2_train_*_phone.log)
- SHUF 臂修正: exp_a_split.json 无 shuffle_map (exp_b 才有且只覆盖 17/20) → 生成 `data/splits/exp_a_shuffle_map.json` (seed=0, 40 条确定性映射, 38 个 partner 全部在 crop/policy manifest)

### 训练状态 (2026-08-31 13:50)
- TTS-kv-phone: ✅ step 0 已保存, 训练中
- NAT-kv-phone: ✅ step 0 已保存, 训练中
- SHUF-kv-phone: 🔄 prepare 阶段 (需加载 40 条记录, 较慢)
- 预计 16h/臂 (CPU), 三臂并行; checkpoint 计划 0/40/80/120/160/200

### 恢复后命令 (GPU 恢复可切换, 但 CPU 训练已在跑, 不打断)
```bash
cd /home/wjj/tts-audio/tts-exp/.claude/worktrees/lrs3-wavlm-resynthesis-50
# 三臂训练 (CPU, 已在后台运行)
# 2. val 选择 + 测试
python .../week2_validate_phone.py --checkpoint-dir tmp/runs/week2_train_tts_phone --kv-mode own_tts ...
python .../week2_test_phone.py --checkpoint-tts ... --checkpoint-nat ... --checkpoint-shuf ... --shuffle-map data/splits/exp_a_shuffle_map.json
```

## 进度 (2026-08-31 13:30)

### Week 1 ✅ 完成 → PROCEED_TO_WEEK2
- CKA: mean=0.3828 (std 0.050), train/val/test 三组一致 (0.379/0.395/0.381)
- 线性探针: train_acc=1.0, val_acc=0.9411, val_auc=0.9871
- 注意力可视化 (Exp 30 TTS-kv step20): mean_row_entropy=0.9978 ≈ 均匀, 无对角线/块状结构
- 结论: TTS 特征与 Natural 高度可分 → 帧级 cross-attention 失败源于对齐/利用方式而非特征冗余
- 产物: `tmp/runs/diagnostic_week1/{features,cka_summary.json,probe_summary.json,final_decision.json}`

### Week 2 准备 ✅ (代码全部就绪)
- **Step 2.1**: MFA3 phone alignment 完成, 40/40 records 通过 natural-span 前缀检查
  - 关键修正: natural 训练切片=完整音频前 3.84s → phone bounds 裁剪到 span; MFA3 空 label 保留为 spn; time_to_frame 单调 clamp
  - 产物: `tmp/runs/diagnostic_week1/week2_mfa/{corpus,alignments,phone_alignments,mfa_summary.json}`
- **Step 2.2+2.3**: `phone_replacement_model.py` — PhonePooling (mean) + PhoneCrossAttentionAdapter (零初始化恒等验证通过)
- **Step 2.4**: `week2_train_phone.py` — 三臂训练 (own_tts/own_natural/shuffled), 复用 Exp 30 loss/optimizer/checkpoint 计划
- **Step 2.5**: `week2_validate_phone.py` — 复用配对门 + val 选择
- **Step 2.6**: `week2_test_phone.py` — 官方渲染 + 严格配对门 + REPLACEMENT_SALVAGED/ABANDONED 判定
- 依赖补齐: textgrid, soundfile, python_speech_features, librosa

### ⚠️ 阻塞: GPU 硬件掉线
- 2026-08-31 12:41:50: `Xid 154, GPU has fallen off the bus` (V100 SXM2 16GB, 0000:07:00.0)
- nvidia-smi 全部 Unknown Error; PCIe link 已被 AER 重置但 GPU 未恢复
- 需要 root: `sudo nvidia-smi --gpu-reset` 或重启机器; 本机无免密 sudo, 远程服务器全部 TCP refused
- 训练已估算: CPU 约 6.5h/臂 × 3 臂 = 19.5h (备用方案, 未启动)

### 恢复后命令 (GPU 恢复后立即执行)
```bash
cd /home/wjj/tts-audio/tts-exp/.claude/worktrees/lrs3-wavlm-resynthesis-50
# 1. smoke 验证 (2 步)
/home/wjj/miniconda3/bin/python scripts/experiments/lrs3_syncnet_finetune/week2_train_phone.py \
  --split-file data/splits/exp_a_split.json --phone-dir tmp/runs/diagnostic_week1/week2_mfa/phone_alignments \
  --cohort-dir tmp/lrs3_policy_a1_200_20260828/policy_cohort --resource-root /home/wjj/tts-audio/tts-exp \
  --kv-mode own_tts --output-dir tmp/runs/week2_train_tts_phone_smoke --steps 2 --save-steps 0,1,2
# 2. 三臂训练 (200 步, 可并行)
python .../week2_train_phone.py --kv-mode own_tts  --output-dir tmp/runs/week2_train_tts_phone
python .../week2_train_phone.py --kv-mode own_natural --output-dir tmp/runs/week2_train_nat_phone
python .../week2_train_phone.py --kv-mode shuffled --output-dir tmp/runs/week2_train_shuf_phone
# 3. val 选择 + 测试
python .../week2_validate_phone.py --checkpoint-dir tmp/runs/week2_train_tts_phone --kv-mode own_tts ...
python .../week2_test_phone.py --checkpoint-tts ... --checkpoint-nat ... --checkpoint-shuf ...
```

## Relations

- project [[31 - Replacement Salvage: Phone-Aligned Cross-Attention]]
