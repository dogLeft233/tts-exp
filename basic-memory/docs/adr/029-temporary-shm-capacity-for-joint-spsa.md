---
title: 029-temporary-shm-capacity-for-joint-spsa
type: decision
permalink: tts-exp/docs/adr/029-temporary-shm-capacity-for-joint-spsa
status: open
project: tts-exp
decided: '2026-10-09T20:41:54Z'
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
---

# 029-temporary-shm-capacity-for-joint-spsa

## Choice and context

在已授权的V100实验机wjj@192.168.50.2:22上，将既有/dev/shm tmpfs容量上限临时remount为24GiB，保持挂载目标、类型、原文件/device/inode/权限与原rw,nosuid,nodev,inode64选项，除size外无变化；不改fstab。继续使用冻结的5GiB输出reserve、模型运行时/科学参数/原自然PCM和时钟、完整分母及live资源gate，完成20joint SPSA与四中心；有界归档回收继续并行。

原/dev/shm总16769675264B，free约10412613632B；当前科学point04实际走各模型子集/perrow资源gate，W2L one-row×2+5GiB为5389588260B，全240行路径10379702720B阈值不适用于每个proposal。初始720三模型snapshot媒体实测外推未来20pairs960proposal+480center约10631292834B，超出原SHM全留存能力。资源helper按实际statvfs及caller阈值检查，无冻结总容量pin。机器实测MemTotal32753276kB、操作前MemAvailable25047629824B，存在约7GiB可用swap；原空间限制是tmpfs上限，RAM并未耗尽。

## Rationale and alternatives

调整容量上限不预先分配24GiB内存，能够立即容纳当前科学路线的媒体外推场景与5GiB输出余量，同时继续归档减少实际驻留。原磁盘仅约5.35GBfree，不能承载未来约10.63GB媒体；服务器CloudFS百度盘虽然已挂载、可用约3.7TB，原生模型输出/随机JPEG评估仍需本地高速暂存。只等待逐批源清理会继续延迟实际联合优化。放宽输出reserve、缩小planned分母或先删除未恢复验证的源证据不采用。

该选择是机器本地、可逆的执行配置。未来媒体大小是基于现有行树的估计，没有硬上界；native CPU峰值RSS未测，不把空间估计当RAM峰值证明。ROOT继续观察实际host/cgroup/SHM/GPU资源，冻结每行/子集gate仍是实际执行边界；若发生资源控制失败，保存完整失败账和耐久journal，归档释放空间并用新attempt继续，不能写成科学阴性。

## Actual application and consequences

ROOT在隐藏echo-off SSH PTY内临时sudo remount,size=24G，actualexit0。操作前后独立只读证据在.codex/scratch/root_shm_capacity_expansion_attempt01/。after QC self1d1ff24a058e3732c67d5a554fec3337e5272708af753df8b42382152d6385d2：total25769803776B、free19412742144B、MemAvailable25032273920B，未来媒体外推+5GiB余量后另余3412740190B；当前index/handoff/coordinator源全bytes/SHA/device/inode/UID/mode不变。只确认本次配置实际生效与当下资源，未宣称20pairs已完成。

耐久session已实际创建且ROOT frozen readonly resume全2122源/event通过：fit40_spsa_00e52e562725_e6e2a096d641，step060，iteration0/pendingfalse。精确coordinator05的唯一detached20启动已放行、PID/actualrun结果仍待。控制manifest/ask/tell与日志位于持久job_control，大媒体仍属SHM脆弱状态，完整可恢复归档是交付条件。已有archive12 F+B及从B到F全恢复资格保留，root只读scanner/UID1000 exactmediaunlink规则不受remount影响。

## Observations

- [decision] 临时扩大同一/dev/shm tmpfs至24GiB，以保留5GiB live输出余量推进完整20pair/四中心，并行有界归档。
- [rationale] 实测host可用RAM约25GB且tmpfs上限构成容量瓶颈；固定资源合同不pin原总大小。
- [alternative] 等待逐chunk清理、把未来媒体放root磁盘、放宽reserve/缩分母均不采用。
- [consequence] 不改fstab/源文件/科学端点；SHM仍易失，完整归档恢复必须完成；未来媒体和nativeRSS仍需实际监测。
- [verification] actualremount0，原mount选项/三sentinel的文件内容和inode保持，after资源场景与reserve检查通过。

## Relations

- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[017-measured-pilot-scenario-and-live-capacity-gates]]
- relates_to [[027-durable-spsa-journal-and-final-center-selection]]
- relates_to [[028-owner-cleanup-with-read-only-privileged-proc-scan]]