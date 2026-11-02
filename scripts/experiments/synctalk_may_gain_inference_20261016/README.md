# May checkpoint gain inference

正式报告：`basic-memory/docs/experiments/48-synctalk-may-gain-inference.md`。
持久本机资源/结果：`runs/synctalk_may_gain_inference_20261016/`。
Ubuntu运行根：`/dev/shm/synctalk-may-gain-inference-20261016/`（易失）。

固定May验证前300帧，−6/0/0重复/+6dB。完整驱动WAV增益后从官方AVE重算，
四个正式视频都带同一段原始PCM。无训练、无后选片段或偏移。

canonical输出为`render_attempt01`和`score_attempt02`，完整QA在
`independent_validation.json`。不要覆盖这些目录。

## 重建环境

本次工具链、wheel、GCC deb和May子集已持久保存；SHM丢失后，先创建同名根目录，
从本机resources复制CUDA tar到远端resources/、wheel到resources/wheels/、
GCC包到resources/gcc12/，再复制本目录全部脚本、protocol.json和score_protocol.json。
运行如下步骤，均不修改系统或已有环境：

```bash
R=/dev/shm/synctalk-may-gain-inference-20261016
A=/home/wjj/tts-audio/synctalk-amplitude-audit-20261016
bash "$R/setup_render.sh"
$A/env/bin/python "$R/reuse_cuda_headers.py"
bash "$R/build_gcc12.sh"
tar -xzf "$R/may_inference_subset.tar.gz" -C "$R"
bash "$R/run_render.sh" "$R/render_reproduction01"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/wjj/.venvs/syncnet/bin/python "$R/score.py" --render "$R/render_reproduction01" --output "$R/score_reproduction01" --protocol "$R/score_protocol.json"
```

依赖前提：之前审计的官方checkout及独立env仍在持久根目录；现有CUDA库/头文件来自
只读共享torch环境。若这些资产也丢失，应按报告47恢复并验证，不能静默改版本。
`setup_render.sh`只解包和准备源码；`build_gcc12.sh`使用隔离GCC12，避免系统GCC13失败。
旧`build_render.sh`是初期GCC13探测记录，不是最终重建入口。

## 检查和播放

- `validate.py`从实际driver、原May PCM、正式mux、RGB哈希和音/视特征重建控制与评分；需numpy/cv2/soundfile/ffmpeg。
- `score.py`保存共享S3FD裁剪、FP32网络特征、31lag距离、full与guard20曲线。
- `previews.py`产生H264/AAC三屏观看版，明确不参与评分。
- `environment_snapshot.py`记录版本、原始资源/编译so哈希和官方工作副本diff。
- `prepare_inference_assets.py`在任何新分数之前固定source帧、音频样本区间与条件。
- 两个download脚本从NVIDIA/Ubuntu官方来源下载资源，不执行系统安装。

本机合同测试仅需numpy：

```powershell
python -m unittest discover -s tests/experiments -p test_synctalk_may_gain_inference_20261016.py -v
```

正式压缩包保留FFV1+PCM mux及所有score数组，排除重复的无音轨FFV1与全段driver/data软链接。
driver可从已保存原PCM精确重建，原始Ubuntudriver已经独立核验。完整May包在此前报告47的resources目录。
