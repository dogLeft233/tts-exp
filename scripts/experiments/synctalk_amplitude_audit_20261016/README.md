# SyncTalk amplitude audit

正式中文报告：`basic-memory/docs/experiments/47-synctalk-amplitude-frontend.md`。
持久本机交付：`runs/synctalk_amplitude_audit_20261016/`。
Ubuntu任务根：`/home/wjj/tts-audio/synctalk-amplitude-audit-20261016`。

此目录不启动训练。官方代码固定commit
`43079fb3f349675f6cb33031bb07262f7566cdd5`，HuBERT固定revision
`ece5fabbf034c1073acae96d5401b25be96709d8`。

- `audit.py`：逐位DOUBLE增益与官方AVE真实权重前向；HuBERT仅processor；DS仅MFCC前端。
- `audit_hubert.py`：严格加载HuBERT大模型，明确处理旧weight_norm参数名，运行官方helper和逐阶段hook。
- `audit_conditioning.py`：严格加载May checkpoint的音频MLP与注意力，运行官方encode_audio；没有图像渲染。
- `validate.py`：本机独立读回有效原始数组与差异，不依赖torch。
- `download_drive.py`：下载公开官方May/trial_may资源，拒绝HTML，写SHA收据。
- `prepare_may.py`：从完整官方压缩包抽取最小子集，保留原split JSON，写成员清单。
- `setup.sh`：新建独立venv，以.pth只读复用现有包；不改已有环境。
- `cleanup_transport.py`：仅对指定的本任务重复传输包核验SHA，复制到SHM核验后释放根盘；不能用于历史目录清理。
- `inspect_*.py`、`probe.sh`：只读资产/环境核查。

所有科学运行都要求新输出目录，canonical目录不覆盖：
`frontend_attempt01`、`hubert_attempt03`、`conditioning_attempt02`。
HuBERT attempt02有漏载权重，已明确INVALID，不进入结论。

`/dev/shm`是易失运行盘。HuBERT完整持久权重在本机resources/hubert-large；
如Ubuntu重启，应从持久副本复制到独立SHM目录并核验SHA后运行。
百度盘挂载写缓存使用根盘，大文件上传实际失败，不能当作备份成功。

本机验证（需要numpy）：

```powershell
python scripts/experiments/synctalk_amplitude_audit_20261016/validate.py --bundle runs/synctalk_amplitude_audit_20261016 --output runs/synctalk_amplitude_audit_20261016/independent_validation_reproduction.json
python -m unittest discover -s tests/experiments -p test_synctalk_amplitude_audit_20261016.py -v
```

本会话使用Codex bundled Python完成此验证；全系统MiniConda没有numpy/pytest，未修改它。
完整执行命令、环境差异、未运行项目与后续训练准备清单见正式报告。
