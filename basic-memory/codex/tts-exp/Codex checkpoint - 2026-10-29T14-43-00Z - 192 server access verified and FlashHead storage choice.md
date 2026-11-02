---
title: Codex checkpoint - 2026-10-29T14-43-00Z - 192 server access verified and FlashHead
  storage choice
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-29-t14-43-00-z-192-server-access-verified-and-flash-head-storage-choice
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-29T14:43:00Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1880e-73d5-7863-a38b-20bcdf7d5ebc
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-29T14-43-00Z - 192 server access verified and FlashHead storage choice

## Summary
192 服务器的个人账号已验证可登录，SoulX-FlashHead Pro 尚未查到；最新容量检查支持优先部署在个人数据盘。

## Story
用户首先要求读取飞书《1048服务器手册》并保存服务器使用 skill，随后提供个人账号并明确要求存入本地 BM 全局记忆、登录服务器查找 SoulX-FlashHead Pro。已创建全局 server-192 skill，并将登录凭证用 Windows DPAPI 加密存入本机 main 笔记；未把明文密码放入 skill 或笔记。该密文只适用于同机同 Windows 用户，不复制到本 checkpoint。

SSH 成功登录 wangjiajun@10.108.16.192:22，实际家目录为 /home/data2/wangjiajun。用户最新问题是公共目录容量，以及后续模型部署应放在哪里。只读检查显示 /home/data/public 所在 /dev/sdb 为 7.3T，剩余 119G，使用率 99%；个人目录所在 /dev/sdc 为 7.3T，剩余 2.7T，使用率 61%。两处 test -w 均成功。建议代码、独立环境、模型缓存、权重及输出都放在 /home/data2/wangjiajun 下，避免即将耗尽的公共盘。个人盘剩余空间为全盘共享余量，不是个人配额。

## Working State
持久入口是 C:/Users/dogle/.codex/skills/server-192/SKILL.md；本地全局凭证笔记是 C:/Users/dogle/basic-memory/Infra/192 服务器连接信息.md，permalink main/infra/192-服务器连接信息。通过本机 PowerShell 解密并以进程环境变量传给 Paramiko，不打印明文。保存时使用本地 basic-memory CLI，并禁用远程语义嵌入。MCP 路由与本地 main 索引不一致，MCP 中未必能读取本机凭证笔记。

远端检查均为只读，没有部署、安装、上传或启动 GPU 任务。机器检测到 A6000 六张与 RTX3090 两张；运行任务前重新查看占用及 UUID。默认非交互 PATH 未发现 conda。仓库存在大量原有修改与未跟踪文件，本任务未改动项目代码；不清理这些文件。

## Changed Files
- 全局 skill：C:/Users/dogle/.codex/skills/server-192/SKILL.md。
- 本地全局记忆：C:/Users/dogle/basic-memory/Infra/192 服务器连接信息.md。

## Verification
skill 经 uv 临时提供 pyyaml 后 quick_validate.py 检查通过。DPAPI 密文经过同用户解密回读验证；SSH 账号与家目录经服务器实际命令确认。FlashHead/SoulX 名称搜索在有限可访问目录未命中，相关运行进程也未命中；部分目录权限不足或扫描超时，不能断言整台服务器没有该模型。本次公共目录 du -sh 35 秒超时，故尚未取得公共目录自身的总占用。df、ls -ld、test -w 已成功返回；未获得个人 quota 数值。SSH 会话已关闭。

git status、branch、SHA 已读取。GitHub repository metadata 确认 dogLeft233/tts-exp；按当前 head 查询 PR 无结果。gh CLI 不在 PATH。未运行项目测试，因为没有项目代码改动。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)
- 当前本地 SHA：2b71639c3b555293375d248350999377168d515d，未验证 GitHub 是否有该 SHA。
- [飞书服务器手册](https://gcnadw6o1ien.feishu.cn/wiki/Vq5gwxt51i2DVdkDhobcP4iqnkh?fromScene=spaceOverview)

## Observations
- [result] SSH 登录与个人数据目录可写已验证；公共盘余量 119G，个人目录所在盘余量 2.7T。
- [decision] 建议后续 SoulX-FlashHead Pro 部署使用 /home/data2/wangjiajun 下的独立目录；当前只是位置建议，尚未开始部署。
- [blocker] 公共目录自身占用统计超时，现有模型搜索受权限与超时限制，不能确认全机不存在模型。
- [next_step] 部署继续时先读取官方 SoulX-FlashHead Pro 安装说明，核验共享 conda 与当前 GPU 占用，再在个人数据目录建立独立部署。