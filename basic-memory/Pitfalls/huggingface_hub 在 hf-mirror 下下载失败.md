---
title: huggingface_hub 在 hf-mirror 下下载失败
type: pitfall
permalink: tts-exp/pitfalls/huggingface-hub-在-hf-mirror-下下载失败
tags:
- pitfall
- huggingface
- hf-mirror
- download
---

# huggingface_hub 在 hf-mirror 下 HEAD 请求失败

## 现象

设 `HF_ENDPOINT=https://hf-mirror.com` 后,`snapshot_download`/`hf_hub_download` 抛 `LocalEntryNotFoundError` / `FileMetadataError`。

## 原因

mirror 的 API 元数据端点(`/api/models/...`)308 重定向回 huggingface.co,HEAD 请求被拒;huggingface_hub 先 HEAD 后 GET 的流程在镜像下不可靠。

## 修复

改用 curl 直连 resolve 端点:

```bash
base="https://hf-mirror.com/<org>/<repo>/resolve/main"
curl -sS -L -o <file> "$base/<file>"
```

- `-L` 必须(LFS 文件重定向到 CDN)
- 注意:mirror 上不存在的文件返回 15 字节 "Not Found" 伪文件 — 批量下载后检查文件大小并对照源端清单
- 下载后 md5 校验

## Observations
- [area] env
- [symptom] HF_ENDPOINT 镜像下 huggingface_hub 客户端 LocalEntryNotFoundError/FileMetadataError
- [cause] mirror API 元数据端点 308 重定向回 huggingface.co,HEAD 被拒
- [fix] curl 直连 resolve 端点,检查伪文件,md5 校验

## Relations
- relates_to [[tts-exp]]
