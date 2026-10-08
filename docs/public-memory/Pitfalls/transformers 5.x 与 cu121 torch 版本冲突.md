---
title: transformers 5.x 与 cu121 torch 版本冲突
type: pitfall
permalink: tts-exp/pitfalls/transformers-5.x-与-cu121-torch-版本冲突
tags:
- pitfall
- transformers
- torch
- cuda
- compatibility
---

# transformers 5.x 强制 torch≥2.6,cu121 装不上

## 现象

本机 torch 2.5.1+cu121(CUDA 12.1,适配 V100 sm_70),但 transformers 5.x 要求 torch≥2.6,升级 torch 则失去 cu121 支持 → 依赖冲突。

## 原因

- transformers 5.x 最低 torch 2.6;而 cu121 最高只到 torch 2.5.1
- V100 (sm_70) 又不能用 CUDA 13.x(必须 cu124 及以下)

## 修复

固定用 transformers 4.x(如 4.51.3)配 torch 2.5.1+cu121;或全套升到 cu124(torch 2.5.1+cu124 / 2.6.0+cu124 已验证可跑)。

## Observations
- [area] env
- [symptom] transformers 5.x 安装/导入要求 torch≥2.6,与 cu121 冲突
- [cause] 版本矩阵:V100(sm_70) → 必须 CUDA≤12.x;cu121 → torch≤2.5.1;transformers 5.x → torch≥2.6
- [fix] 用 transformers 4.x;或 torch 2.5.1/2.6.0 + cu124

## Relations
- relates_to [[tts-exp]]
