---
title: 旧SyncTalk May与HuBERT资源的百度云恢复位置（2026-10-19）
type: note
permalink: tts-exp/codex/remember/旧-sync-talk-may-与-hu-bert-资源的百度云恢复位置-2026-10-19
project: tts-exp
repository: dogLeft233/tts-exp
tags:
- codex
- manual-capture
---

# 旧SyncTalk May与HuBERT资源的百度云恢复位置（2026-10-19）

用户要求自行解决E盘满盘，明确允许旧数据移入百度网盘后删除。已将两份旧原始资源逐片上传，并在Ubuntu独立逐片和整项读回SHA通过后，以Windows独占文件handle再次核对原文件hash/大小，再定点删除E盘副本。共归档2487161749字节；原实验数组、结果、模型断点和失败现场未删除，训练仍因数值terminal停止。

- May-original：原位置 `./runs/synctalk_amplitude_audit_20261016/resources/May.zip`；1225104190字节，SHA256 `008fe3fb46430a92beff334e26f37eab229c717ff5b577dfbba90f46c3848d4b`。云目录 `B:\tts-exp-research-models\synctalk-multispeaker-full-20261017\E-storage-May-original-20261019\transport_attempt01`；完整恢复清单 `F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/storage_reclamation_20261019/May-original_cloud_restore_locator.json`。
- HuBERT-large-original：原位置 `./runs/synctalk_amplitude_audit_20261016/resources/hubert-large/pytorch_model.bin`；1262057559字节，SHA256 `9cf43abec3f0410ad6854afa4d376c69ccb364b48ddddfd25c4c5aa16398eab0`。云目录 `B:\tts-exp-research-models\synctalk-multispeaker-full-20261017\E-storage-HuBERT-large-original-20261019\transport_attempt01`；完整恢复清单 `F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/storage_reclamation_20261019/HuBERT-large-original_cloud_restore_locator.json`。

恢复时按对应manifest的parts顺序拼接bundle.partNNN到原文件名，逐片与整项SHA均一致后才重新使用；不要将分片直接当模型或ZIP。原路径旁也保留`.cloud_restore_locator.json`。所有源、清单、云端读回证明和本地退休凭据在 `F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/storage_reclamation_20261019/`，完整transport证明在同FROOT的`cloud_delivery/E-storage-*/transport_attempt01/`。本次只解决存储和发布失败记录，不恢复训练或启动新评测。

## Observations
- [result] 两项旧资源的云端完整字节和恢复位置已验证，E盘本地副本已按用户授权退休。
- [constraint] 今后使用这两份资源必须显式恢复并验证SHA，不能假定原E路径仍有payload。
