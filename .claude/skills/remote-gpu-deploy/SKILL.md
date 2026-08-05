# remote-gpu-deploy

GPU 服务器部署工作流：环境探测、权重下载、conda 环境管理、常见代码修复、多模型兼容性矩阵。

## 服务器探测（Connect 前必做）

### GPU 可用性五步检查

Seetacloud 等平台可能分配 CPU-only 容器。仅 `nvidia-smi` 能找到不等于可用：

```bash
# 1. nvidia-smi 是否为占位文件
file /usr/bin/nvidia-smi       # 若为 empty，则容器无 GPU 挂载

# 2. 设备节点
ls -la /dev/nvidia*            # 若 "No such file or directory" → GPU 未挂载

# 3. 驱动库
ls -la /usr/lib/x86_64-linux-gnu/libcuda.so*  # 若为 0 字节 → 仅为占位

# 4. PyTorch 探测（比 nvidia-smi 更可靠）
python -c 'import torch; print("cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())'

# 5. GPU 型号
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1
```

**已知踩坑：** `nvidia-smi` 可能是 0 字节占位文件，此时 `/dev/nvidia*` 也不存在，但 `/usr/lib/x86_64-linux-gnu/libcuda.so.x.x.x` 有版本号占位。这种情况直接输出——容器需要重新创建或重新挂载 GPU。

### 数据盘容量检查

| 模型 | 权重大小 | 备注 |
|---|---|---|
| JoyVASA | ~10GB | |
| IMTalker | ~8GB | |
| Hallo3 | ~52.2GB | CogVideoX-5B + T5-XXL + 音频/人脸模型 |
| Hallo2 | ~15GB | |
| LatentSync | ~5GB | |

```bash
df -h /root/autodl-tmp   # Seetacloud 数据盘挂载点
df -h /autodl-pub        # 公共 NFS（通常是只读）
```

### Conda 环境枚举

```bash
CONDA_BIN=/root/miniconda3/bin/conda
"$CONDA_BIN" env list
for prefix in $("$CONDA_BIN" env list | awk '/\// {print $NF}'); do
  "$CONDA_BIN" run -p "$prefix" python -c '
import sys; import torch;
print(sys.executable, "| torch", torch.__version__,
      "cuda", torch.version.cuda if hasattr(torch.version, "cuda") else "none",
      "ok" if torch.cuda.is_available() else "no-gpu")
' 2>&1 || echo "skip $prefix"
done
```

## Conda 环境管理

### 策略

- **每个模型独立环境** — 依赖版本冲突无法避免（OpenCV 4.9/4.10、numpy、torch 版本）
- 不修改已有环境（JoyVASA、IMTalker 等）
- 命名：`<model_name>`，例如 `hallo3`，无需加 `-env`

### 已知环境对照

| 环境 | Python | PyTorch | CUDA | 备注 |
|---|---|---|---|---|
| `joyvasa` | 3.10 | 2.2.2+cu121 | 12.1 | OpenCV 4.9.0.80, numpy 1.26.4 |
| `imtalker` | 3.9 | 2.0.1+cu118 | 11.8 | |
| `halloo3` (新建) | 3.10 | 2.4.0 | 12.1 | 需过滤 torch/nvidia/opencv 包 |

## 依赖安装常见修复

### OpenCV 冲突

Hallo3 的 `requirements.txt` 同时装了三个 OpenCV 包（`opencv-python`, `opencv-contrib-python`, `opencv-python-headless`），这会导致 numpy 绑定 bug。**必须收敛为一个**：

```bash
awk -F== '!($1 ~ /^(torch|torchvision|triton|nvidia-|opencv-contrib-python|opencv-python-headless)$/)' requirements.txt > filtered.txt
pip install -r filtered.txt
pip install opencv-python-headless==4.9.0.80   # 替代所有三个
```

- OpenCV 4.10+ 有 numpy 绑定 bug（`cv2.imdecode` 等函数抛异常）
- **永远不要 co-install** `opencv-python` + `opencv-contrib-python` + `opencv-python-headless`

### DeepSpeed

Hallo3 需要 `deepspeed==0.14.4`。在消费级 GPU（单卡）上安装失败不阻断推理。记录日志后继续即可。

### MultiScaleDeformableAttention

JoyVASA 动物模式需要 C++ 扩展编译。成功前提：
- CUDA 工具链可用
- Ninja 已安装

### NumPy 兼容性

| 框架 | 兼容版本 |
|---|---|
| OpenCV 4.9.0.80 | numpy 1.26.4 |
| scipy | 与 numpy 大版本必须匹配 |

如 `pip check` 报 `numpy.distutils` 冲突，删冲突的 dist-info 目录后重装：

```bash
pip uninstall -y numpy scipy
pip install numpy==1.26.4 scipy==1.13.1
```

## 权重下载

### Hugging Face（网络加速 + Xet 禁用）

```bash
source /etc/network_turbo
export HF_HUB_DISABLE_XET=1
huggingface-cli download <repo> --local-dir <path> --local-dir-use-symlinks False
```

`HF_HUB_DISABLE_XET=1` 必须设置，否则 Meta 大文件下载失败（`huggingface_hub>=0.27` 默认启用 Xet）。

### 人脸模型（某些 HF 仓库遗漏）

Hallo3 的 HF 仓库包含 `buffalo_l.zip`，但解压和 face-landmarker 下载需手动：

```bash
FACE_DIR=pretrained_models/face_analysis/models
mkdir -p "$FACE_DIR"
unzip -j -o "$FACE_DIR/buffalo_l.zip" '*.onnx' -d "$FACE_DIR"
wget -qO "$FACE_DIR/face_landmarker_v2_with_blendshapes.task" \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### LivePortrait 动物检查点（JoyVASA）

```bash
huggingface-cli download KwaiVGI/LivePortrait --local-dir <path>
```

## 验证清单

部署完成后的最小验证：

1. **PyTorch GPU**: `torch.cuda.is_available()` → `True`, `device_count()` ≥ 1
2. **模型导入**: `python -c 'from <model> import <entry>'` → 无 ImportError
3. **推理结果**: 单样例生成 MP4 → `ffprobe` 显示 video + audio 流
4. **显存峰值**: 不超过物理 VRAM（RTX 4080 = 32760 MiB）

## 常见问题

| 症状 | 原因 | 解决 |
|---|---|---|
| `nvidia-smi: empty` | 容器无 GPU 挂载 | 重新创建实例 |
| `libcuda.so` 存在但 0 字节 | GPU 占位文件 | 同左 |
| OpenCV 报 numpy 相关错误 | 多个 OpenCV 包或 numpy 版本 | 收敛到 opencv-python-headless==4.9.0.80 |
| `huggingface-cli` 下载大文件断开 | Xet 协议未禁用 | `HF_HUB_DISABLE_XET=1` |
| `MultiScaleDeformableAttention` 编译失败 | CUDA 工具链或 Ninja 缺失 | `apt install ninja-build` |
