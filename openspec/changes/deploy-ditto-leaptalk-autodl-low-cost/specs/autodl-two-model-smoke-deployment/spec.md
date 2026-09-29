## ADDED Requirements

### Requirement: Bound paid operations before execution

部署 agent SHALL 在真实资源操作前确认实例身份、SSH 入口、数据盘目录、同机任务、时价和总费用上限，以及完成/失败时的关机授权。密码与 Token SHALL 只经环境或凭据管理器取得，不写入配置、日志或 spec。默认单实例单 GPU，按 design 的兼容候选与实时价格选卡；本 spec 本身不构成开卡授权。

#### Scenario: Instance or budget is unknown

- **WHEN** 尚无明确目标实例或费用上限
- **THEN** 可以准备本地脚本与无副作用检查，但租机/付费启动前 SHALL 向用户确认缺失信息，不猜用历史服务器

### Requirement: Prepare in no GPU mode before enabling GPU

Agent SHALL 先用无卡模式准备独立环境、固定源码与权重版本、下载完整性清单、输入及命令。CPU 可完成的安装与下载 SHALL 完成后再开卡；只有已说明原因的 GPU/重型构建允许后移。无卡资源不足 SHALL 记录为延后项，不能伪报安装完成。模式切换 SHALL 先保存并结束本任务，再关机、以目标模式开机；保留数据，不释放实例。

#### Scenario: CUDA is unavailable during preparation

- **WHEN** 控制台确认无卡模式，安装或探测发现没有 GPU
- **THEN** 将 CUDA 验证留到 GPU 阶段，不执行推理、不替换为 CPU-only 依赖；输出准备完成项和仍待 GPU 验证项

### Requirement: Keep both inference backends isolated and minimal

Agent SHALL 使用 design 规定的 Ditto PyTorch 与 LeapTalk Lite 路线，两套独立环境，在同一 GPU 上串行运行。SHALL 记录源码 commit、权重 revision/校验信息、实际依赖、命令、后端/精度/分辨率/帧率及 seed（上游支持时）。SHALL 核实 LeapTalk LoRA、audio projection、TAE 实际加载。不安装训练数据，不启动 Web 服务，不修改旧实验环境。

#### Scenario: Upstream entry point differs from documentation

- **WHEN** 选定 commit 的启动参数或依赖与网页示例不同
- **THEN** 检查该 commit 源码/帮助，保存最小适配 diff 与真实命令；不得编造 CLI 选项或静默切换成其他模型

### Requirement: Accept only actual smoke generation

两模型 SHALL 使用同一张可用正脸图、同一段固定 3–5 秒有效语音，各成功生成一条视频即可；不调用新 TTS。SHALL 保存输入来源、截取/重采样参数与 hash。每模型验收 SHALL 同时满足：

1. GPU 阶段 `torch.cuda.is_available()` 为 true，并记录实际型号、总显存、驱动；核实模型计算确实用 GPU。
2. 推理进程真实退出码为 0，MP4 非空、具有音视频流；全片用 ffmpeg 解码无错误，`ffprobe` 记录尺寸、帧率、帧数和音视频时长。
3. 画面覆盖输入语音时长，允许尾差不超过一个已记录的官方 chunk 时长；非 chunk 模型允许 0.2 秒。最终音轨时长与输入差 ≤0.2 秒；不允许仅产出首帧/首 chunk 就通过。
4. 实际查看头/中/尾帧及短视频，确认人脸可见、视频有运动、不是黑屏或静态图片，并试听音轨；若无法进行播放/试听，明确 `REVIEW_PENDING`，不能宣称完整验收通过。不设同步质量分数门槛。

若官方输出无声，SHALL 保留原视频并用原输入音频生成带声副本；记录 mux 命令、保留原视频时长，不通过截短输入掩盖未完成推理。SHALL 记录进程退出码、耗时、GPU 显存使用采样最大值及采样间隔（如 1 秒；不是精确分配峰值）。

#### Scenario: MP4 exists but inference failed

- **WHEN** 返回非零退出码、视频不可解码、只有静帧或未覆盖输入语音
- **THEN** 该模型 SHALL 为 FAILED，不因文件存在而判 PASS；另一模型的结果独立保留

### Requirement: Stop within the approved cost window and preserve evidence

Agent SHALL 使用有界任务超时并处理子进程，遵守 design 的 GPU 窗口和重试限制。退出码不能被 `tee` 或关机命令覆盖。完成/失败/超时后 SHALL 先保存结果，再仅在已核实的目标 AutoDL 实例内执行已授权关机；不得在本地设置 shutdown。SHALL 验证控制台/API 最终 stopped 状态，不能把 SSH 断连当成关机证明。

产物 SHALL 保存于远端 `runs/<run_id>/`，至少含两模型视频、执行日志、输入/版本清单、`result.json`、简短 `result.md`、可重跑命令和报价/用时记录。`result.json` SHALL 分开记录两模型 `PASS/FAILED/BLOCKED/REVIEW_PENDING`、总体部署状态、关机 `CONFIRMED/UNCONFIRMED`；仅两模型均 PASS 时总体 PASS。日志和媒体 SHALL 回传到本仓库的新 `runs/autodl_ditto_leaptalk_<timestamp>/`，不覆盖历史 results。回传失败时保留远端数据并报告路径，不为无限重传持续开 GPU。

#### Scenario: Lite still runs out of memory

- **WHEN** 官方低显存路径在限定重试后仍 OOM
- **THEN** 保存显存与错误证据、按授权关机，报告该模型 FAILED 和待确认的最低报价升级选项，不擅自增加付费资源

#### Scenario: Shutdown cannot be confirmed

- **WHEN** 关机请求后无法查询实例最终状态
- **THEN** 保存 `shutdown_status=UNCONFIRMED` 并明确告知用户可能仍计费，不声称费用已经停止
