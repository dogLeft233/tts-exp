# 服务器访问

## 已配置服务器

| 用途 | 端口 | GPU | 状态 |
|------|------|-----|:---:|
| TFG 推理 (16461) | `connect.weste.seetacloud.com:16461` | RTX 4080 SUPER 32GB | ⚠️ TCP refused 2026-07-30 |
| TFG 推理 (43417) | `connect.weste.seetacloud.com:43417` | RTX 4080 SUPER 32GB | ⚠️ TCP refused 2026-07-30 |
| SyncNet 评估 (26502) | `connect.westc.seetacloud.com:26502` | RTX 4080 SUPER 32GB | ⚠️ TCP refused 2026-07-30 |
| 实验数据 (22035) | `connect.westb.seetacloud.com:22035` | — | ⚠️ TCP refused 2026-07-30 |

> 密码不存储在本文件中，使用 `sshpass` 传递。

> 2026-07-30 仅做 TCP 可达性探测；上述端口均返回 connection refused，需重新启动或确认容器状态后再部署。

## 已下线服务器

| 地址 | 原因 |
|------|------|
| `connect.weste.seetacloud.com:43794` | SoulX-FlashHead Pro；已完成批量推理后关闭 |
| `connect.westd.seetacloud.com:48111` | AVTR-1；已完成批量推理后关闭 |
| `connect.westc.seetacloud.com:26795` | AvatarForcing；已完成批量推理后关闭 |
| `connect.westc.seetacloud.com:41601` | LeapTalk；已完成 26 条批量推理后关闭 |
| `connect.westd.seetacloud.com:14453` | 旧 TFG 推理 (LatentSync/JoyVASA/V-Express) |
| `connect.westd.seetacloud.com:20398` | 旧评估服务器 |
| `connect.westd.seetacloud.com:24599` | 旧 Ditto 实验 |
| `10.1.114.114:1986` | 旧 1080Ti 集群 (Wav2Lip/MuseTalk) |

## 连接方式

```bash
SSHPASS='<password>' sshpass -e ssh -p <port> \
  -o StrictHostKeyChecking=accept-new \
  root@connect.<region>.seetacloud.com
```

## 关键路径 (26502 评估服务器)

```
/root/autodl-tmp/repos/tts-exp/       # 主仓库
/root/autodl-tmp/envs/syncnet/        # SyncNet conda 环境
/root/autodl-tmp/envs/ditto/          # Ditto conda 环境
/root/autodl-tmp/checkpoints/         # 模型权重
```

## 网络注意事项

- 中国大陆服务器：HuggingFace 被墙 → 设置 `HF_ENDPOINT=https://hf-mirror.com` 或 `source /etc/network_turbo`
- ModelScope 作为 HF 替代品下载权重
