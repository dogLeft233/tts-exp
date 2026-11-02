---
title: LAN V100 夜间降噪功耗策略
type: decision
status: accepted
date: October 11, 2026
permalink: tts-exp/docs/adr/lan-v100-night-power-policy
---

# LAN V100 夜间降噪功耗策略

✅ 已部署到 `wjj@192.168.50.2`，服务 `gpu-night-quiet.service` 为 enabled / active。用户要求北京时间 23:00–次日 08:00 降噪，同时尽量保持性能与 GPU 安全。优先采用可逆的功耗上限，不改变风扇、时钟、电压、ECC、驱动或正在运行的研究任务。

## 策略与边界

固定目标设备为 Tesla V100-SXM2-16GB，UUID `GPU-d12b3309-c299-da83-064d-a3cd619bd560`。实际允许功耗为 100–300W，部署前上限 300W。夜间先采用 200W；白天恢复原 300W。200W 是初始折中值，尚未测量实际研究任务的性能损失或声压变化，不能称为已优化的最优值。[NVIDIA 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)说明功耗上限通过功率管理调整性能状态，因此 300W→200W 不代表性能下降三分之一。

常驻进程显式使用 `Asia/Shanghai`，每 15 秒检查一次，正常切换延迟不超过约 15 秒。系统重启后自动重新判断当前时段。热保护全天优先于时段：

| 条件（满足任一） | 功耗上限 |
|---|---:|
| 核心 ≥75°C 或 HBM ≥78°C | 150W |
| 核心 ≥80°C 或 HBM ≥82°C | 100W |
| 核心 ≤68°C 且 HBM ≤70°C，连续 4 次正常采样 | 恢复当前时段上限 |

遥测失败时尝试 100W，并记录错误；掉线时设置也可能失败。没有自动重启、GPU reset、驱动重载或杀任务。软件保护不保证消除硬件、供电、PCIe 或驱动导致的掉线，也不能替代原有硬件散热保护。

## 已执行验证

2026-10-11 22:51–22:53（东八区）实际安装并启用；5 项单元测试通过，覆盖跨午夜、UTC 换算、核心/HBM 保护、冷却滞回、设备身份/无效遥测/功耗边界及设置读回失败。服务器实际执行四个时段边界的只读演算，22:59:59→300W、23:00→200W、07:59:59→200W、08:00→300W。随后实际将设备设为 200W并读回成功，服务按当时白天策略恢复 300W并读回成功。未修改服务器时间，也未等待真实 23:00 边界；未运行重负载性能基准。

探测时核心 47°C / HBM 46°C、约 24.58W，无 CUDA 计算任务；ECC 启用、计数为零。设备报告最高运行温度核心 83°C / HBM 85°C、核心降速 87°C、关机 90°C；自定义保护阈值均提前于对应运行温度上限。主板为 INTEL X99-XD3，V100 风扇读数 N/A；没有可用的 V100 风扇控制或 IPMI 设备。唯一暴露的 PWM 属于 AMD 显示卡，未改动。若 V100 的外接风扇固定转速，限功耗无法保证降低空闲噪声；进一步降噪需要确认散热器与接线方式。

当前启动及此前五次启动日志中仅发现 2026-09-29 的 Xid 43，没有找到 Xid 79。按 [NVIDIA Xid 文档](https://docs.nvidia.com/deploy/xid-errors/archive/index.html)，Xid 43 是应用相关通道故障；这条记录不能确认用户描述的历史掉线根因。

## 维护

源码：`scripts/ops/gpu_night_quiet.py`，配置和服务在同目录；测试：`tests/ops/test_gpu_night_quiet.py`。

服务器正式文件为 `/usr/local/sbin/gpu-night-quiet`、`/etc/gpu-night-quiet.json`、`/etc/systemd/system/gpu-night-quiet.service`，均由 root 拥有。安装包保留在 `/home/wjj/gpu-night-quiet-20261011/`。修改配置后重启服务才会读取新值。

```bash
systemctl status gpu-night-quiet.service --no-pager
journalctl -u gpu-night-quiet.service -n 30 --no-pager
nvidia-smi --query-gpu=uuid,power.limit,power.draw,temperature.gpu,temperature.memory --format=csv
sudoedit /etc/gpu-night-quiet.json
sudo systemctl restart gpu-night-quiet.service
```

停止并取消开机自启：`sudo systemctl disable --now gpu-night-quiet.service`。停止服务会保留当前功耗上限；若温度正常且需要恢复部署前设置，再执行 `sudo nvidia-smi -i GPU-d12b3309-c299-da83-064d-a3cd619bd560 -pl 300`。

## Observations

- [date] October 11, 2026
- [status] accepted
- [decision] 采用服务器本地 systemd 常驻服务，东八区夜间 200W、白天原 300W，全天核心/HBM 热保护，不修改风扇控制，不自动重启。
- [rationale] 系统未提供 V100 风扇控制；驱动认可范围内的功耗上限可逆，并能降低高负载热量。原有低负载约 25W，空闲噪声收益未证实。
- [alternatives] 固定低转速会依赖未知散热硬件与接线；锁时钟或电压增加变更范围；自动重启会中断研究任务。此次均不采用。
- [verification] 实际 200W 设置/读回与常驻服务自动恢复 300W通过；服务 enabled / active；5 项测试通过；性能、声压及真实夜间切换尚未实测。

## Relations

- relates_to [[server-access]]
