---
title: 028-owner-cleanup-with-read-only-privileged-proc-scan
type: decision
permalink: tts-exp/docs/adr/028-owner-cleanup-with-read-only-privileged-proc-scan
status: open
project: tts-exp
decided: '2026-10-09T16:36:54Z'
tags:
- engineering
- storage
---

# 028-owner-cleanup-with-read-only-privileged-proc-scan

## Decision

媒体归档与精确回收由实验文件owner wjj/UID1000执行；root权限仅交给固定源码、固定七个已完成leaf与历史PID/ticks范围的只读/proc扫描器。ROOT审核预检与逐文件清单，再建立绑定policy、inventory、实际preflight及精确candidate digest的授权回执。每次全预检和每leaf删除前重新检查实际占用，不能使用之前PASS扫描作为缓存授权。

## Context and rationale

用户授权持续运行实验并用百度网盘解决空间。七个完成leaf已流式归档到F与B，ROOT核验whole tar、全部11473members和源身份，候选9936媒体245861096B；1537keepers/88777064B包含84科学NPZ、原音、frame maps、源码、控制与日志，原路径保持并在回收后逐个全SHA复核。B挂载可读不等于云同步完成。

普通用户全同UID进程占用扫描在四个仍存活的进程cwd_readlink遇到EACCES。忽略它们会失去无writer门限。实际固定privileged scanner02以EUID0读取、target owner UID1000明确区分，能完整检查cmdline/cwd/fds和PID身份；原scan的四个EACCES因此得到实际权限解决。root无需写实验输出或删除文件。

凭据经用户已授权的native SSH控制PTY、sudo-v隐藏stdin传入，绝不写文件、argv、日志或BM。该PTY中normalwjj worker再用固定argv sudo-n /usr/bin/python3 -B exacthelper；不得假定另一次SSH已有ticket，不记录凭据本身。扫描器sourcepath/bytes/SHA、schema、owner、七leaf/PIDtuple、canonical raw/self、actualchildrc0与空stderr严格绑定；任何错误、引用、race或超时闭合失败。

## Alternatives and consequences

以root运行整个worker扩大了权限与文件owner变化，故不采用。跳过live unreadable进程或按state Z猜测可忽略同样不采用。预先扫描一次再长期复用会漏掉新writer；固定ROOT授权只定义候选范围，当前占用仍须每次重新实测。仅复制到B且未核验member也不能替代可恢复归档证据。

保留逐文件dirfd/no-follow/bytes/SHA/inode身份检查与不可变hash-chain journal：先ATTEMPT，unlink成功后记录FILE_UNLINKED并fsync；错误准确报告attempted/deleted/partial。任何失败先取真实journal和所有回执，再决定fresh修复/剩余范围，不自动重跑或递归删除。后续通用archive执行采用同一权限拆分，但其新leaf范围和固定scanner必须另行真实绑定，不能直接沿用七leaf。

## Evidence and affected work

- scanner02 source .codex/scratch/fit40_privileged_proc_scan_attempt02/privileged_proc_writer_scan_attempt02.py，14746B/rawdcfc04d37f9f677b7d28885ff6328cf04db6426a4a7cf4daaaf5acfcbd60fce5；ROOT5tests及一次实际authenticated只读scan PASS，读回在同目录readback_attempt01。
- worker11/policy18 .codex/scratch/fit40_queue_archive_stream_attempt11/，worker80563B/raw1729e627ae02cf7870978a83046b9b2cf33aa2da77d2785545cc4330dfd047b8，ROOT27tests PASS/zero skips/rc0。
- 实际verify-only receipt .codex/scratch/fit40_queue_archive_stream_attempt11/actual_verify_readback_attempt01/verify_only_receipt_attempt01.json，29087B/rawec8db826aed29b94e30ef3aa0fabb3c785964f9e9d8ae8b1f1ca5ab58d6f0d84/self28ea72caec59d9faa61c8b9d5e0db29b2ee842e4caf80930885e69ff461a1b85，actualchildrc0/9936candidate/七leaf零引用；尚不代表已经删除。
- ROOT精确授权 .codex/scratch/root_cleanup11_exact_authorization_attempt01/authorization_attempt01.json，1575B/raw03e669183f56af5cf4c2a5764a7c25d576c2f2369d0450bf895b6e8d45216ed7，candidate digest e1196065210c293355c35d480e8614045c1642b497252c1f6f708a3779f67225。一次实际unlink-approved返回childrc2，真实journal/删除数诊断正进行中，不假定零删或成功。
- [Repository](https://github.com/dogLeft233/tts-exp)，branch codex/cross-tfg-audio-head-20261003，HEAD 5688784dd5c6c1272074f0c4b6c5ff79f117c907 local/unpushed。

## Observations

- [decision] owner wjj执行实验回收，仅固定只读proc scanner特权；精确ROOT授权与每leaf新占用门限分别执行。
- [rationale] 四个live同UID cwd EACCES是实际权限问题，特权读取保持完整writer排除，同时避免扩大删除权限。
- [alternative] 全worker作为root、跳过unreadable/state猜测和复用旧scan均不采用。
- [consequence] 密码只经隐藏PTY stdin；固定范围/hash/rc严格验证，逐文件journal、准确partial计数与1537keeper postSHA强制保留。
- [verification] ROOT scanner02 5/worker11 27tests通过，实际verify-onlyrc0；一次unlink-approvedrc2须按实际回执诊断。

## Relations

- relates_to [[研究服务器缓存清理与可恢复归档]]
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]