# ditto-talkinghead
## requirements
### 首先使用environment.yaml下载基础环境
conda env create -f environment.yaml
### 额外下载一些库
python -m pip install onnxruntime-gpu mediapipe einops
### 下载Flask相关库
python -m pip install -r /data/zhaohuiyang/nas/241/HammerChat/requirements.txt
### tensorrt及两个附属包需要单独pip
python -m pip install tensorrt==8.6.1 tensorrt-bindings==8.6.1
python -m pip install  tensorrt-libs==8.6.1 --extra-index-url https://pypi.nvidia.com
### trt模式，需要切换到libcudnn.so.8
#### 下载cuDNN
到https://developer.nvidia.com/cudnn-archive?utm_source=chatgpt.com下载cuDNN 8 for CUDA 12.x的 Local Installer for Linux x86_64 (Tar)
#### 再放到固定目录并解压
cd ~/Downloads/cudnn8
tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
#### 随后创建存放目录，并复制库文件到目录：
mkdir -p $CONDA_PREFIX/opt/cudnn8
cp -av ~/Downloads/cudnn8/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib $CONDA_PREFIX/opt/cudnn8/
cp -av ~/Downloads/cudnn8/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include $CONDA_PREFIX/opt/cudnn8/
#### 最后配置环境变量
export LD_LIBRARY_PATH=$CONDA_PREFIX/opt/cudnn8/lib:$LD_LIBRARY_PATH
#### 再检查是否成功
ls $CONDA_PREFIX/opt/cudnn8/lib | grep libcudnn
## exapmles
CUDA_VISIBLE_DEVICES=3 \                                                                        # 视频输出，pytorch模式
python inference.py \
    --data_root "./checkpoints/ditto_pytorch" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./example/output.mp4"

CUDA_VISIBLE_DEVICES=3 \                                                                        # 视频输出，trt-online模式
python inference_online.py \
    --data_root "./checkpoints/ditto_trt_Ampere_Plus" \
    --cfg_pkl "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl" \
    --audio_path "./example/audio.wav" \
    --source_path "./example/image.png" \
    --output_path "./example/output.mp4"