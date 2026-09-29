## Context

🔄 待部署，尚无本轮实测显存或成功产物。用户希望“能跑就行、卡越便宜越好”，明确先无卡安装再开卡；本地 16GB 显存不是租卡显存上限。

官方网页核对日期：2026-09-06；模型 README 已读取，远端依赖安装与控制台价格未验证。实施时先固定源码 commit、权重 revision，再读取该版本的安装文件和入口参数；网页抽取缺失的命令不得凭空补造。

## Goals / Non-Goals

交付范围见 proposal。只封装官方离线入口，输入使用同一张正脸图片和同一段 3–5 秒语音；不追求实时 FPS，不将部署成功解释为泛化或 replacement 成功。

## Decisions

### 1. 一台实例、两个环境、两次串行推理

建议任务根 `/root/autodl-tmp/ditto-leaptalk-deploy`，实际先验证挂载与剩余空间。下设 `repos/`、`envs/`、`models/`、`inputs/`、`runs/<run_id>/`；包缓存也放数据盘。不覆盖旧 `ditto` 环境、仓库、权重或输出。

两个脚本各接收 `--image`、`--audio`、`--output`；环境及模型根使用明确配置或任务专用环境变量。调用绝对 Python 路径，不依赖交互式 conda activate。实施前检查上游 shell 是否硬编码覆盖环境变量；必要时生成小配置补丁并保存 diff。不把 Web 的 `.env` 参数误当成离线 CLI 参数。

### 2. 无卡先完成所有不需要 GPU 的准备

串行创建环境、安装预编译包、下载权重及依赖辅助模型、准备输入；记录下载来源/revision、文件大小、SHA-256。核对上游大小/校验值（如提供），排除 LFS 指针、HTML 错误页和部分下载。配置与代码引用的文件必须齐全；不在 2GB 内存里加载整套模型来验证。

优先兼容 wheel；如 CUDA 扩展没有兼容 wheel，记录扩展名、版本和原因，留到 GPU 阶段有限时编译。无卡阶段只下载这类源码/构建依赖。不因 `torch.cuda.is_available()==False` 判定无卡安装失败，也不为绕过检查改装 CPU-only torch。

AutoDL 无卡模式文档当前标注 0.5 CPU / 2GB RAM / 0.1 元每小时，会释放 GPU；不是免费，也不保证之后有卡。下载加速仅在目标 AutoDL 的下载子 shell 内 `source /etc/network_turbo`，不修改全局代理配置。[AutoDL 省钱说明](https://www.autodl.com/docs/save_money/)；[下载加速](https://www.autodl.com/docs/network_turbo/)。

### 3. 模型路径

| 模型 | 首选部署路径 | 必须准备的内容 |
| --- | --- | --- |
| Ditto | Python 3.10，官方 PyTorch 配置 | `digital-avatar/ditto-talkinghead` 内的 `ditto_pytorch/` 与 `ditto_cfg/v0.4_hubert_cfg_pytorch.pkl`，含 ONNX 辅助模型 |
| LeapTalk | Python 3.12；torch/torchaudio 2.7.1、torchvision 0.22.1；官方 Lite，1 step，关闭可选 compile | 官方指定的 SoulX-FlashHead-1_3B、wav2vec2-base-960h、`z-rx/leaptalk` 的 LoRA、配套 audio projection 和 Lite TAE |

以上来自 [Ditto 官方 README](https://github.com/antgroup/ditto-talkinghead) 和 [LeapTalk 官方 README](https://github.com/zhangrongxiang/LeapTalk)。选定 revision 的依赖要求优先；记录实际 CUDA wheel、驱动与包版本。LeapTalk 基座与 wav2vec 的完整下载 repo ID 从该版本 README 核对，不凭目录名推断。

Ditto 沿官方 `inference.py` 指定 `--data_root`、`--cfg_pkl`、`--audio_path`、`--source_path`、`--output_path`。PyTorch 路径不保证完全没有 ONNX/CUDA 或导入级 TensorRT 依赖：按该版本实际导入安装，不建设 TRT engine 链路。历史 BM `tts-exp/docs/reference/ditto` 仅作兼容性线索，旧机器路径、Flask 和 cuDNN8/TRT 配置不直接套用。

LeapTalk 沿官方 `inf.sh` / `inference.py`，核实其 `LITE=1`、单步和 compile-off 的真实设置及生效日志；必须加载 LeapTalk checkpoint，而非只运行 SoulX 基座。保留该版本默认支持的分辨率、帧率和 chunk 约束；短音频降低 smoke 总耗时，但不保证降低常驻显存。

### 4. 卡型按报价选，不写死“最低显存”

先比较当前可用的 Ampere 或更新架构、至少 16GB 显存、至少 32GB 主机内存的单卡实例。RTX 3090 24GB 可作报价基准；同时比较更便宜的兼容 16GB/24GB 卡。此为试跑候选条件，不是两模型最低配置承诺。依赖若要求更高能力则剔除不兼容候选；不为极低时价引入旧架构适配工作。

在候选中选最低报价且依赖可用的机器；报价接近时优先 24GB 减少显存试错。记录候选、单价、显存/内存、镜像及选择理由。无实时报价时不能声称某卡最便宜。先确认时价上限、总费用上限及目标实例，再租用；不在本 spec 中虚构价格或预算授权。

首轮 GPU 总窗口建议最多 60 分钟（包括编译、诊断与收尾），每模型单次推理最多 10 分钟，每模型最多一次有明确修改的重试；实际取预算允许时间和 60 分钟的较小者，预留至少 5 分钟收尾。OOM 后释放本任务进程，仅尝试上游已支持的低显存选项；禁止无限重试、自动加卡/升价。仍失败就保留证据、关机，提出下一档最低报价供确认。

## Risks / Trade-offs

- 无卡内存不足 → 停止重复安装，保留完成项，将确有必要的重操作列为 GPU 待办；权重下载和轻量配置仍应在无卡阶段完成。
- 重新开机缺卡 → 保持关机，报告库存；克隆/迁移需确认目标、费用和数据，不删除原实例。
- Lite 或 PyTorch 与历史后端不同 → 独立标记 `leaptalk_lite` / `ditto_pytorch`，不合并历史指标。
- GPU 命令退出不等于停止计费 → 保存产物后请求关机，并从控制台或已核实 API 验证实例 stopped；无法确认就显式报告。

## Migration Plan

无历史数据迁移。执行前核实实例和授权；无卡准备完成后关机再正常开机。GPU 任务完成、失败或达到时限都保存日志并按事先确认的收尾授权关机，不能影响同机其他任务。回退仅停止本轮进程、保留任务目录，不卸载旧环境，不释放/删除实例。
