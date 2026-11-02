---
title: LRS3 visual dynamic specificity 2026-09-10
type: experiment
permalink: tts-exp/experiments/lrs3-visual-dynamic-specificity-2026-09-10
status: concluded
date: '2026-09-10'
tags:
- openspec
- concluded
- parallel
---

# LRS3 visual dynamic specificity 2026-09-10

本次根据最新实验历史，按用户要求设计适合GPT Luna独立实施的轻量OpenSpec；CPU分析、独立验证和root阶段审阅已完成。

父115/133 observed，缺失界限跨0。本回顾性指标采用MSE分解和固定时间反转；21组observed条件估计与23组原分母bounds分开。实测 `b_dynamic mean=-0.002322`，99% CI `[-0.026600, 0.022207]`，结论为 `NO_OBSERVED_DYNAMIC_ADVANTAGE_ESTABLISHED`。

## Observations

- [status] concluded
- [question] 共同支持上剔除静态嘴形后，候选是否具有更接近真实视频且对时间顺序敏感的动态轨迹？
- [openspec] openspec/changes/probe-lrs3-visual-dynamic-specificity/
- [report] openspec/parallel-next-experiments-20260910.md
- [inputs] openspec/next-experiments-20260910-inputs.json
- [budget] CPU-only，0模型/0视频/0评分；不训练、不生成TTS、不下载模型。
- [boundary] 已见数据探索/纠错续验；replacement_confirmed=false、waveform_head_authorized=false、generalization_established=false；不修改父run与旧阴性。
- [parallel] 三路独立准备/分析；单GPU使用公共flock，先phone-core后reference；仅更新本实体。

- [implementation] 已实现 fixed parent/lock/review/tree/H bindings、399 feature byte hashes、H support exact-content 检查、static/dynamic/total MSE、reversal controls 与 observed/missingness bounds；现有 run 需显式 `--resume` 才可复用。
- [validation] 11 tests、compileall、Ruff、OpenSpec strict 均通过；`runs/lrs3_visual_teacher_missingness_20260910_v5` validator PASS（兼容旧目录别名）；validator 绑定并重读 H/records.json，timing natural/candidate 均为 true。
- [result] 115/133 records、21/23 groups observed；0模型/0视频/0评分。本轮只支持“未建立 observed dynamic advantage”，不支持 replacement 或 generalization 结论。
- [review] root 阶段审阅补齐 feature 内容/字节绑定和 active=0 BLOCKED 处理；未发现阻断问题。
- [next] 若要收窄缺失性不确定性，需另行设计新 cohort；当前实验已收敛。

## Relations

- follows [[LRS3 visual teacher missingness audit 2026-09-09]]

## Changelog
| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 根据最新结果及源码偏差创建独立OpenSpec计划；尚未执行 | September 10, 2026 | user |
| C CPU分析、独立验证与root审阅完成：9 tests、实际run validator PASS，结论已记录 | September 10, 2026 | agent/root |
| 三项实验统一终审通过：self-hash、focused tests、compileall、OpenSpec strict 和各 run validator 均 PASS | September 10, 2026 | agent/root |
| 终审补强：validator 绑定 H/records.json 防止重签重分组；聚焦 pytest 11、Ruff、compileall、OpenSpec strict 和 validator 均 PASS | September 10, 2026 | agent/root |