---
title: echomimic-v3-flash
type: deployment
permalink: tts-exp/docs/deployment/echomimic-v3-flash
status: VERIFIED
server: 10.108.16.192
model: EchoMimicV3-Flash
verified_date: '2026-10-30'
tags:
- server192
- tfg
- echomimic
---

# EchoMimicV3-Flash：192个人目录部署

✅ 2026-10-30已完成真实单图+音频推理及中文20源80逻辑视频（41唯一输入），九评分器和本地证据复核全部通过。科学结论与完整配对数值见 [[77-echomimic-v3-chinese20-nine-scorers]]；本记录说明可执行环境与本次受控配置。

## 路径和范围

- 主机：wangjiajun@10.108.16.192:22，实际HOME /home/data2/wangjiajun。使用前按全局server-192 skill重新读取当次飞书1048手册并检查资源。
- 模型根：/home/data2/wangjiajun/EchoMimicV3-Flash；独立Python：env/bin/python。
- 官方源码：repo/，revision 7e89489ca51c0d008fc1963ec6c03fc5bd0b9397；ZIP/source receipt保留。
- 资产：models/transformer、models/chinese-wav2vec2-base、models/Wan2.1-Fun-V1.1-1.3B-InP。
- 受控批量代码：/home/data2/wangjiajun/av-evaluation/experiments/echomimic_v3_chinese20_20261030/；本机同名scripts/experiments/目录。
- 已完成run：/home/data2/wangjiajun/av-evaluation/jobs/echomimic_v3_flash_chinese20_rms24_20261030_attempt01。
- SSH凭证走本机DPAPI运行器的运行时解密，记忆与脚本不含明文。

## 环境与资产

共享/home/user/anaconda3/bin/conda仅用于创建个人克隆环境，源为本人av-evaluation/envs/eval-base；包缓存、pip/HF缓存与TMP均为个人路径。实际torch2.7.1+cu128、transformers4.57.3、diffusers0.35.2、numpy1.26.4、cv2 4.11.0；完整pip-freeze随证据保留，未给原评分环境升级依赖。

下载前有限检查公共model；本次19资产共20793443133字节。HF官方revision绑定，ModelScope同作者/基础模型镜像逐文件大小/SHA一致后下载；小文件再核对HF Git blob ID。Flash为echomimicv3-flash-pro完整检查点（非2025 preview），SHA256 5ebdbb2fc709108bf2a1728fd92eb2874804e4bc0324e92a2cd55425968c85a4。中文wav2vec2 HF revision3991242c806928916fff4a8c0e4f76acf661b743，pytorch_model.bin SHA be2da40c9e7ae26bfc904a3ed79ebb9e8f060bec6dba85d6a6ae86114bc38901。基础模型HF revisionfc913c34361f4ec879e2f9c78b4f11ae50a937d1；T5/CLIP/VAE/tokenizer实际加载。

完整Flash可按其config构建并strict加载全部键（实测1,863,772,992参数），不用先加载随后会全被覆盖的基础DiT。未用基础DiT的部分下载已在确认无进程使用、路径位于个人目录后清理；初始/选定清单均保留，部分adapter检查点不能沿用这种省略。

## 可执行入口与兼容处理

本次实际入口是实验目录的prepare.py/generate.py/controller.py，orchestrator为模型根run_experiment_v2.sh；它们绑定已冻结的特定run。已完成run保持只读；新实验使用新attempt并显式更新路径/协议，不直接重跑旧orchestrator覆盖分析或归档。上游原infer_flash.py作为来源保留，不能假定在此依赖版本下直接运行其默认入口等同于本包装器。

Diffusers0.35.2缺少上游旧load_model_dict_into_meta，T5低内存加载回退FP32。私人包装器采用CPU加载后显式BF16，并在offload前断言Transformer、VAE、T5、CLIP四模块均为BF16；主机RAM需求需要单独核对。最初诊断中止，原日志/worker/显存与修复绑定完整保存，没有复用其未完成媒体。其它三模块及T5实际加载均零missing/unexpected；Flash另做strict全键校验。

FlashAttention未安装，使用PyTorch SDPA；私有attention适配保留真实key/query长度语义，非因果全注意力，未给原本None的text context_lens凭空加mask。独立CPU sliced-softmax测试及GPU BF16对FP32参考通过（最大差0.00411725），实际生成验收通过。关闭编译，不改系统CUDA/驱动。

## 本次模型合同

单静态3.png，512²/25fps/seed42，8步Flow_Unipc、text CFG6/audio CFG3、shift5、BF16、model CPU offload、TeaCache.1/skip5。VAE encode/decode自动清缓存，scheduler set_timesteps重置求解历史；每片新建CUDA generator和TeaCache，批量保留权重而不保留前片运动状态。

官方run_flash示例最多81帧，本次按每条完整音频计算原生4k+1向下帧数，范围61–197。完整归一化PCM不改，条件前缀/未支持尾部逐片记录，最大未支持尾部154.125ms；未为五秒评分器外部补帧/补音或时间拉伸。静态图输入不是将参考视频喂给唇形重配模型。

沿用前轮RMS−24dBFS WAV，绕过上游额外−23LUFS；实际中文W2V preprocessor do_normalize=False，实测特征提取器逐值保留float32输入前缀。内部LN/GN另层处理，不能推断完全没有尺度消除。该入口有别于作者默认LUFS和2025预览版英语W2V，本次配对结论限受控设置。

## 资源与验收

单A6000 UUID GPU-de7ff5ff-3510-739f-917c-cb98892de759，Torch allocator22GiB上限，4逻辑CPU affinity并设置Torch/OMP4；nohup运行，GPU阶段顺序执行。成功生成NVIDIA驻留显存5秒采样峰值15214MiB（14.86GiB），Torch allocated/reserved峰值约13.26/14.51GiB，41唯一输入计时累计43.36分钟、不含模型加载。这是A6000上限定内存的实测配置，不是物理24GB卡另一次测试或任意长度的保证。

80/80媒体分辨率/帧率/帧数、非静态、ALAC解码PCM逐样点一致通过；首对N/T联系表已查看。3项服务器测试、12脚本编译通过；五个原评分环境pip-freeze集合与前轮完全一致。原始七项归约、本地2376内容文件/80导出媒体、25统计分层复算通过。实验GPU进程已退出，环境与资产保留。

## 复现材料

[代码说明](../../../scripts/experiments/echomimic_v3_chinese20_20261030/README.md)与results/echomimic_v3_flash/chinese20_rms24_20261030_attempt01/提供输入/协议、资产与源码快照、raw特征矩阵、逐视频与配对表和图。评分入口沿用server192-nine-scorers的独立环境；包装器需要GPU_UUID而非仅CUDA_VISIBLE_DEVICES，生成与评分阶段分别绑定本人的缓存根。完成证据包SHA256 ca8cd17cb6974016e69e3470c9cd01fb9927bb2d9d0b8eefca2f51ff3f09380b。

已将前端配置核对、dtype回退检查、批量缓存边界与原生帧数/显存验证经验加入本机全局server-192/references/personal-deployment.md，quick_validate通过；经验不取代当次手册。

## Observations

- [result] 原生单图EchoMimicV3-Flash在192个人目录完成80逻辑视频/41唯一输入，九评分器真实前向与归档读回通过。
- [verification] BF16与model CPU offload有效推理采样峰值14.86GiB；原评分环境集合保持不变。
- [scope] RMS24受控入口绕过官方LUFS；do_normalize=False和4k+1尾部限制均随协议留档。

## Relations

- validated_by [[77-echomimic-v3-chinese20-nine-scorers]]
- relates_to [[server192-nine-scorers]]

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Record verified private EchoMimicV3-Flash deployment, compatibility fix and controlled Chinese20 run | October 30, 2026 | user |


## 官方配置对照（2026-10-30）

✅ [[78-echomimic-v3-official-control]] 已完成6来源单变量及官方组合诊断，独立run为av-evaluation/jobs/echomimic_v3_official_control_20261030_attempt01。LUFS、81帧上限和seed43分开对照，共50真实生成/74 SyncNet评分；旧20来源归档不变。新评分前缀必须验证显示帧与PTS，H264数据包截取的B帧顺序问题已通过独立无损裁剪修复并留存失败证据；代码及LF启动器见scripts/experiments/echomimic_v3_official_control_20261030/。已执行run不可直接重跑。
