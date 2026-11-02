---
title: Patrick RMS归一化同初始化独立复跑
type: experiment
permalink: tts-exp/experiments/patrick-rms-归一化同初始化独立复跑
status: paused
version: patrick_ave_rms_coarse_repeat_v1
contract_sha256: b44d67ce4daa7076e88648adefceb663118dd877a5b2a527a7bcf2d5439ee53b
execution_status: PAUSED_BY_USER
scientific_status: IN_PROGRESS
normalization: whole_recording_RMS0.1_before_same_AVE
seed: 42
tags:
- synctalk
- patrick
- rms
- repeat
controller_revision: controller_revision02_metadata_handoff_v1
---




# Patrick RMS归一化同初始化独立复跑

## 问题和新授权
用户在RMS原试验数值失败、零更新密度定位及强制恢复短试跑116次全skip后，明确要求“从头开始一次带归一化的试试”，并询问是否与主流编码器一致。这是独立新执行版本，不重启任何失败controller，不使用失败权重、EMA或恢复试跑scaler，不覆盖旧预算/归档。沿用原先授权的Patrick seed42三档−6/0/+6各coarse1–10×6000attempts，最多180000formal。

## 冻结设计和来源
OUT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_repeat_v1；remote RUN=/dev/shm/synctalk-multispeaker-data-20261017/trials/patrick_ave_rms_coarse_repeat_v1。新contractSHAb44d67ce4daa7076e88648adefceb663118dd877a5b2a527a7bcf2d5439ee53b，19冻结源。新wrapper仅重定向新RUN/版本、明确旧输入绑定并核parent合同；原14训练数学源精确保留。RMS仍为先gain后整录音float64 RMS0.1、再float32进入同AVE，无去均值/epsilon/clip/limiter；6份既有已资格AVE缓存freshSHA复验后只读复用，不冒称重跑AVE。41447原inputs全pin、6featureSHA、新源19SHA、原完整initial freshSHA通过，SHM free约12.19GB；不动共享环境/CloudDrive。

同一seed42完整随机initial SHA d6ce1f369bd2ee1695288c93f1602fcf181f0329313c710388e6b7c70071be5c、原initial tensorSHA1bd087fb0ad9e62cddf5e92ec142507664ba0dfe99ccf6607a144967b2cae35c。原FP16/defaultGradScaler/loss AdamW EMA/65536rays maxsteps16/原LR和采样不改，恢复试跑的cache rollback不带入。初始资格的6实际更新不进入formal初始权重/optimizer或预算。新版本保留全部中间full checkpoint，便于对照定位，不做退休删除。首numerical/job terminal或lost先F封存再停全部新job，不换gain、不replay、不自动重启。

## 当前执行证据
用户最新明确要求“完成这一轮后先保存断点并中断实验”，覆盖此前继续coarse10/评测授权。当前+6gain coarse7已自然训练、验证、完整保存结束，三档均完成coarse1–7，共21/30完整epochs，各42000attempts。最新0/−6/+6累计成功更新为41991/41990/41991，普通AMPskip按实际记账；更新不等保留、不补步，phase最终资格尚未验收。

F三档最新coarse7完整RAW checkpoint、全部model/optimizer/scheduler/scaler/EMA/counters及原逻辑trace/curves均保留。三份最新training archive全部成员与standalone fullcp再次freshSHA通过，最新+6ep7 raw6000trace/finite loss与7个跨gain18000行sampling/LR block独立复核passed；未新GPU重算或加载全tensor。持久证据OUT/user_pause_v1/paused.json、remote_pause_verification.json、controller_exit_classification.json及independent_reviews；暂停时间2026-10-19T12:43:35.174715+00:00。三份cpSHA：0=497916c17d6966bfb35062f634b96fabb0d90a13facfd8e5bfde0759602c3d0a；−6=e042410f62a58b8c61e676f81c9322f20602294ba701534fd06528b4ac90e7bb；+6=788fd29c2563b658c749b9dcc3973611b7c5f8077ad5c5de578b8d9dbc5eba43。

用户pause_request在当前作业运行时写入，允许它完整结束但禁止下一job；host229476在F持久和block核验后由原pause guard退出，0GPU信号/0host信号。原controller_revision02_metadata_handoff_v1/controller_failure.json含预期用户pause AssertionError，原样保留并独立分类为显式暂停，不是训练数值terminal。Linux无本任务训练/评分进程，formal job恰21个、无coarse8 job、无新score_27_v1。OUT/active_controller.json已指向user_pause_v1/paused.json，原authority在previous_authority.json；旧root receipt/metadata故障和冻结19源/contract b44d67ce4daa7076e88648adefceb663118dd877a5b2a527a7bcf2d5439ee53b保持，不重启旧host、不重放。未来恢复需人类新授权和逐receipt新审计handoff，不能删除pause flag后盲重启。

本repeat粗训练和新27格目标均未完成，0新视频/评分，不能称科学成功/失败或RMS总体因果结论。评分helper仍仅score_receipt_candidate_v1的6CPU测试候选，未接入实际renderer或GPU资格。旧RMS数值失败/恢复probe116全skip、旧coarse CONTROL_FAILED/Brown失败全部保持；旧fine14/Chris/Cathy和原24full仍暂停。自动巡检在暂停发布核验完成后删除，不自动继续训练/评测；不归档chat或标总目标done。

## 归一化与编码器原生合同的边界
当前x→0.1x/sqrt(mean(x²))是为电平消除干预额外加入的RMS，非AVE原生预处理，也不等于逐输入z-score。(x−mean)/sqrt(var+1e−7)会减均值、近单位方差，统计范围取实际送入processor的片段/录音且需处理padding；不能把不同位置和范围都称同一归一化。此次不同时换算法，否则无法观察同一设计是否重现失败。

2026-10-19再次核官方小配置/源码：facebook/wav2vec2-base-960h与facebook/hubert-large-ls960-ft的do_normalize=True，采用Wav2Vec2FeatureExtractor的零均值单位方差；TencentGameMate/chinese-wav2vec2-base与microsoft/wavlm-base-plus的HF预处理配置均False，不能按家族统一推断实际入口。Whisper原生是log-Mel动态范围下限和固定affine，不是逐录音RMS0.1或z-score。配置说明processor行为，直接喂模型绕开processor时需单查调用链。既有归一化调查中的Tencent中文wav2vec2False纠正保持。

来源：[Wav2Vec2配置](https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/preprocessor_config.json)、[HuBERT配置](https://huggingface.co/facebook/hubert-large-ls960-ft/raw/main/preprocessor_config.json)、[零均值单位方差源码](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py)、[腾讯中文wav2vec2配置](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base/raw/main/preprocessor_config.json)、[WavLM配置](https://huggingface.co/microsoft/wavlm-base-plus/raw/main/preprocessor_config.json)、[Whisper源码](https://raw.githubusercontent.com/openai/whisper/main/whisper/audio.py)。仅文档/小源码核对，没有新编码器推理或更换AVE。

## Observations
- [status] paused
- [decision] 同RMS0.1、同seed42完整随机初始状态另立repeat，绝不继承失败权重/缩小scaler/cache fallback。
- [result] 用户要求当前+6coarse7结束保存后暂停，已21/30完整epochs、三最新fullcp/F全验，无新coarse8或评分作业；属于人为暂停而非数值失败。
- [blocker] 用户主动暂停，coarse8–10及此次新27格尚未执行；phase最终控制资格/数值全程稳定性与RMS训练轨迹因果未结论。
- [next_step] 保留暂停与全部state/trace证据；等待人类新方向，若显式恢复则逐receipt新审计handoff，绝不自动重启、重放或评分。

## Relations
- repeats [[Patrick同AVE逐录音RMS归一化三档粗训练与27格评测]]
- implements [[运行Patrick的RMS归一化AVE三档粗训练并评测]]
- relates_to [[048-synctalk-patrick-same-ave-rms-coarse]]
