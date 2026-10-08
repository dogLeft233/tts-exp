# Step-120 自然音频增强头候选

[离线包](v3_step120_fit_candidate_offline_bundle_attempt02.zip)包含七个原始源码文件、冻结配置、checkpoint、训练记录和 selection record，以及中文使用说明和逐文件 SHA-256。解压后从包根目录调用 `python -m scripts.audio_enhancement.enhance_v3_attempt02`；完整命令见包内 `README.md`。

配置要求 CPU Python 3.10.12、NumPy 1.26.4、PyTorch 2.5.1+cu124，PyTorch intra/inter-op 各一个线程。包内不包含运行库、数据或 TFG 权重。增强波形只用于驱动视频；原始自然音频须单独保留，生成后用 `mux_original_pcm` 将完整原始 PCM WAV 无损复用到 `.mkv` 或 `.mov`。该接口同时核对音频、视频码流与帧时钟。Stage-A 接收原始音频，由其内部应用增强一次；不要先增强再传入。

这是 Fit40/20 位说话人、固定 Adam 第 120 步的 **NOT_SELECTED 候选**。未用 Dev 或 Confirm 选择参数；跨模型资格仍待完整三模型 Dev 验证。本包的工程验证和 Fit 代理结果不构成跨模型收益的验收。

## 隔离包 CPU CLI 复现

隔离 ZIP（13 个成员、44,326 B，SHA-256 `32145ea20d18602d2fda2b66ca6fcc7256c215ad9fd70ca23da5a91534a33700`）已完成一次只用 CPU 的 Fit-only CLI 复现。`BAC009S0052W0121` 输入为单声道 16 kHz、62,639 个样本，原 WAV SHA-256 `2d45f2b65dec29f77bcd56f05f883b16207a99e7848df3a06a203819a915b261`，规范化 float32 PCM SHA-256 `f4e611e9f816665ad7c4dce4980f461c4d9af4fe6fe9678c8f756bd03fa6b36c`。固定环境下命令用时 2.389 秒；Python 3.10.12、NumPy 1.26.4、PyTorch 2.5.1+cu124，CPU，torch intra/inter-op 均为 1，CUDA 未初始化。

独立下载读回确认输出是 RIFF/WAVE IEEE float32 little-endian，单声道 16 kHz、62,639 个样本：WAV 250,600 B，SHA-256 `9cf46db4c5caf7f058bda52b290ab5a79405fddf97ad541bfd9dc72624b6bfa0`；PCM SHA-256 `ef72ce37b21878436a7f38888bff04a71446d71a4d2e7ebf85bb84fec6cd2f6c`。12,233 B 的 CLI metadata SHA-256 为 `2aab5bfecf4a6d5ce6e916cc0fe1b3e453b216afc238e508904704a51999db76`。输出 WAV 与此前实际 Fit CLI smoke 的原始字节 SHA 完全相同；metadata 因隔离输出路径变化而原始 SHA 不同，CLI 回执记录其稳定字段投影相同。该单样本结果只证明此隔离包路径可复现，不是 TFG 得分或跨模型效果结论。

从解压后的包根目录，以原始 WAV 驱动候选头一次：

```bash
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
python -B -m scripts.audio_enhancement.enhance_v3_attempt02 \
  --input /path/to/original_natural.wav \
  --output /path/to/enhanced_generator_drive.wav \
  --config runs/cross_tfg_audio_head_20261003/v3_step120_candidate_materialization_fit120_attempt11_attempt01/v3_method_candidate_config.json \
  --metadata /path/to/enhanced_generator_drive.metadata.json
```

将增强 WAV 仅用于 TFG 生成；最终交付音频仍取完整原始 WAV：

```python
from pathlib import Path
from scripts.audio_enhancement.overdub import mux_original_pcm

info = mux_original_pcm(
    Path("generated_video.mp4"),
    Path("original_natural.wav"),
    Path("final_with_original_audio.mkv"),  # 也可用 .mov
)
print(info)
```

CLI 回执：[实际隔离运行](v3_step120_actual_isolated_cli_response_attempt02.json)（5,461 B，SHA-256 `27a295f78011f27f77a739ad04b2fadc6f1a4e2dbdde94de89c9da3b83fcd3c3`）；[独立 RIFF/PCM/metadata 读回](v3_step120_actual_cli_output_independent_qc_attempt01.json)（3,672 B，SHA-256 `49d166ba837070947c881f8bfbb790c05c5133555908e72c94818ef8f73b7bec`）。[ROOT 复核](v3_step120_actual_cli_output_root_qc_attempt01.json)另行读取 WAV 的实际 PCM、计算峰值/RMS，并核对全部冻结构造参数、权重与来源。

## 工程证据

- `v3_step120_candidate_bundle_manifest_attempt02.json` 记录 11 个 payload 原始字节来源与 SHA。
- `v3_step120_candidate_bundle_zip_readback_receipt_attempt02.json` 是逐个读取 13 个 ZIP 成员的实际校验记录。
- `v3_step120_candidate_bundle_root_qc_attempt01.json` 独立校验 ZIP、原文件、配置、checkpoint 参数和 provenance 路径闭包。隔离包 CPU CLI 回执和下载后 RIFF/PCM/metadata 独立读回均已通过，见上节。
- `dev360_cpu_ready_summary_attempt09.json` 和两份 CPU QC 记录修复后的 binding09/360 计划及实际预检。
- `dev360_actual_controls18_launch07_receipt_attempt10.json` 与启动 QC 记录真实 detached controls360 启动。启动 QC 覆盖首五行工程 metadata，不代表完整媒体或效应验收。
- `dev360_failed_controls17_compact_snapshot_receipt_attempt02.json` 与两份 compact QC 记录旧失败批次紧凑快照的 1,968 个实际目标文件读回：1,966 个对象 / 80,023,309 B，保留完整 360 行的 120 PASS、240 FAILED。完整媒体另行归档。
- [实际候选计划准备](candidate180_actual_metadata_rebind_summary_attempt01.json)及[独立绑定 QC](candidate180_actual_metadata_independent_qc_attempt01.json)记录当前服务器的180条有序候选、360个不同对照引用、三前端与60条Wav2Lip帧映射。仅元数据准备，尚未候选GPU生成或运行资格验收。
- [实际 ASR 静态部署](candidate180_actual_asr_static_stage_summary_attempt01.json)、[独立读回](candidate180_actual_asr_static_stage_independent_qc_attempt01.json)及[ROOT 联合复核](candidate180_actual_metadata_and_asr_static_root_qc_attempt01.json)记录32个逐文件SHA核对的静态成员；尚未API内容检查。实际generator输入PCM须在候选完成后单独检查。
- [对照161行实际快照 QC](dev360_actual_observer_poll04_root_qc_attempt01.json)核对原始回包、PID、自哈希及六单元完整计数；[旧媒体归档启动](dev360_failed_controls17_media_archive_attempt03_launch_receipt.json)与[容量失败回执](dev360_failed_controls17_media_archive_attempt03_failure_receipt.json)保留真实失败。该归档未产生tar或成功回执，后续采用一份持久tar和独立SHM传输分块修复容量。

所有这些 JSON 和 ZIP 都保留实际原始字节；大型 run 数据仍由运行目录与持久快照提供。旧失败批次及其完整失败分母保留。
