---
title: PEAVS与SyncReward官方推理审计及批量掩码问题 2026-10-27
type: note
permalink: tts-exp/experiments/peavs-与-sync-reward-官方推理审计及批量掩码问题-2026-10-27
status: CONTROL_FAILED
engineering_status: AUDIT_COMPLETE
scientific_status: OFFICIAL_SETTINGS_ALIGNED_LOCAL_TIMING_CONTROLS_STILL_FAIL
run_id: av_metric_official_audit_20261027_attempt01
training: false
tags:
- peavs
- syncreward
- audit
- synchronization
- upstream-bug
---

# PEAVS与SyncReward官方推理审计及批量掩码问题 2026-10-27

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

## 问题与结果
用户质疑PEAVS/SyncReward未通过±400ms控制是否源于调用或预处理错误，要求核实是否严格官方方法。本次实际完成媒体、源码、全部关键权重、官方入口、精度和批量的差分审计。旧调用并非完全官方默认：PEAVS用batch1而发布默认64；SyncReward用FP32而官方CUDA选择BF16。PEAVS批量差异对部分样本有实质影响，定位到发布代码的有效长度掩码错误。更正设置后仍未通过本项目预设32/40资格门，不将其称作论文官方benchmark失败。

## 冻结对象与媒体核验
原20来源自然视频各±400ms，共40控制，来自runs/av_metric_preference_20261026_attempt02。未恢复用户取消的跨自然/TTS换轨评测，只审计同一音轨的已知移位。全40逐PCM验证移位±6400采样点和零填充；视频像素逐帧相同，起始PTS/采样率/帧率一致。全40 PEAVS视频I3D特征逐值相同，音频VGGish特征改变。官方MP4→AAC128k→WAV后，两代表控制仍精确保留±6400采样点互相关偏移。SyncReward官方16k→48k解码后，代表波形内部逐采样误差0；中心起点0.565s，正负400ms整个4.75s窗口均没有边缘补零，故不能以补零解释该反例。

PEAVS45源码/配置Git blob与9cf014ba2c02551f4494cae685607ca3b5328f43匹配；6仓库模型权重Git blob匹配；VGGish/PCA官方发布SHA匹配。SyncReward tracked source无修改，版本f29fbdc2bb2dd3d59644ffd3c60d207dd3e62615，Stage2 6,635,671,828 bytes SHA6c09263ac41d59baf78a181f17e4116128a337d70a03637886de3734450a32a2再次验证。没有错误模型、缺失保留参数、错配音轨或分数方向读反证据。

## PEAVS：批量依赖是真问题
直接运行未改写metric.py（仅拦截最后删除缓存），batch1的60输出与旧包装器最大差0；官方默认64与batch1最大单项差0.8118796。默认64控制下降25/40，旧batch1为26/40，均低于32/40。

最小复现仅两条视频：目标BAC009S0008W0121_N音频特征5x128、视频4x1024，单独2.395576，与一条更长视频同批2.789705，媒体/特征完全不变。发布utils/dataset.py把cur_dur_v写成len(cur_aud)，混合批次中被补零的视频位置可能被当有效位置。Wav2Lip视频尾部短于音频，再加I3D丢弃不完整窗口/AAC补齐，会形成长度差。仅作诊断把两侧特征截等长后，单独/同批结果2.40754485/2.40754461，差2.38e-7；该裁切不是官方分数，也未改写正式实验。

CuPy RawModule兼容核与独立Torch逐点定义最大差2.38e-7。两代表失败样本重新完整提取/评分与旧分数完全一致。审计脚本首次CuPy在Torch之后导入碰到MFA CUDA库冲突，只修正审计子进程导入顺序以匹配原bootstrap的CuPy先导入；原生产bootstrap和历史分数未修改。

默认64对全部80原生文件（20N+60cloud，保存排序/批次组成）复算：
|版本|N|cloudT|T-N|T更高|对Sync-C反向|
|---|---:|---:|---:|---:|---:|
|历史batch1|2.217287|1.813750|-0.403537|6/60|42/60|
|官方默认64，保留上游掩码错误|2.252396|1.844311|-0.408085|6/60|42/60|

严格同参考12来源默认64：N2.414602、T1.939770、Δ-0.474832，T更高3/36、冲突30/36。九条cloud特征音视频长度不等。偏好方向没有变化，但默认64仍依赖批次组成，不能把它当可靠性问题已修复。

## SyncReward：BF16更正与官方核心对照
“V100无原生BF16，所以不能执行BF16、必须FP32”的推断过强。当前Torch2.6在V100(capability7.0)可软件执行包括完整模型的BF16，is_bf16_supported(including_emulation=False)=false，但含仿真=true。本次完整执行官方CUDA BF16 autocast，保留官方允许的低显存encode_batch_size1（默认18）。仍用发布load_av/processor/prepare_inputs、48k、18x0.5s、stride0.25、共享中心4.75s和发布模型/严格全Stage2参数。

直接执行官方score.main的FP32兼容模式，原生S0002=1.450853109、提前400ms=1.478407383、延后=1.492405653，与旧包装器三项逐值差0。FP32编码batch1与2最大差2.38e-7，不能解释该反例。恢复官方BF16后，代表三项为1.453125、1.4765625、1.4765625，仍是错位更高。

|精度|下降|持平|上升|本项目门槛|
|---|---:|---:|---:|---:|
|历史FP32|16/40|0|24/40|32/40|
|官方CUDA BF16|14/40|7/40|19/40|32/40|

完成60控制媒体及3个官方demo派生控制的BF16评分。FP32/BF16最大绝对分差0.065010667，不能把精度差一概说成可忽略；但它不是控制失败的全部原因，因为BF16也失败。软件BF16并非H20硬件bit-exact复现。旧原生SyncReward均分/胜率仍是FP32版本，本审计没有重评完整60条cloudT为BF16，不伪称完成。

加载优化仍标注：按官方配置构建骨干（恢复num_classes1024），再由发布build_reward_model严格加载完整Stage2，避免下载会被覆盖的9GB初始骨干。这与原始下载流程不同，但所有保留参数都由Stage2覆盖。没有偷偷改评分公式或数据窗口。

## 适用范围和决策
PEAVS官方README明确AVS基准排除人脸；论文第7节说明未在VoxCeleb/LRS等talking-face上实验。I3D/VGGish按0.96秒抽特征，不能直接把它当中文唇同步裁判。SyncReward包含Speech，但本审计只能确认它在这批静态头像中文视频上的400ms排序不可靠，不能单独证明训练域或语义线索是唯一原因。
- [decision] 不把PEAVS/SyncReward对TTS的低分或与Sync-C分歧当作SyncNet有偏好或真实口型优劣的裁决。
- [decision] 保留旧冻结输入、包装器及历史结果，明确batch1/FP32标签；新入口采用发布默认批量/精度，同时保留低显存支持参数和加载优化的说明。
- [decision] 不为让资格门通过而按分数调时移或私自修官方模型掩码。等长裁切仅作最小诊断。

## 产物、调用与验证
run：runs/av_metric_official_audit_20261027_attempt01，Ubuntu /home/wjj/tts-audio/tts-exp，Windows .有报告/JSON/官方源码快照；原始完整输入及重特征以远端和审计证据包为准。
主报告official_inference_audit_report_20261027.md；audit_summary.json；media_and_cached_features_audit.json；source_and_weights_audit.json；peavs_all_repo_weights_audit.json；vggish_weights_audit.json；peavs_post_aac_timing_audit.json；syncreward_official_parity.json；syncreward_bf16_controls.json及scores.jsonl；peavs_official_parity.json；peavs_minimal_batch_dependency.json；peavs_native_default64_reanalysis.json。

版本化新入口部署在/home/wjj/av-eval-20261026/：
- score_peavs_official64_20261027.py：默认--metric_batch_size64，保留batch组成警告；真实两视频新入口smoke通过，S0008=2.7897048，与原始默认64一致。
- score_syncreward_official_20261027.py：默认--precision bf16，--precision float32显式重放历史；真实新入口S0002=1.453125，precision=bf16，与官方BF16控制相同，exit0。
两者在scripts/experiments/有版本化副本。旧score_peavs.py与score_syncreward_v100.py不覆写。tests/test_av_metric_official_audit_20261027.py五个真实artifact integration函数直接执行通过；py_compile通过；未运行pytest。官方源码未改，没有DEBUG仪表留在模型中。未新训练、TTS调用或TFG生成。

## 审计证据包

完整审计证据包（不含可按SHA重新取得的6.6GB SyncReward权重）保存在${USER_HOME}/.codex/tts-exp-artifacts/av-metric-official-audit-20261027/audit_evidence.tar.gz，36,148,970 bytes，SHA256 79715c0c65db09b7dc9787bd8a871c799f4b7fd90ffa02973b12edf0b0e1e96a。431成员逐文件读回核验通过，包含报告、代码、控制媒体、120组PEAVS音视频特征和PEAVS评分头。C是本机备份，不是百度网盘。原始重特征的其他工作目录不冒称已全归档。

## References
- [PEAVS官方仓库与非人脸基准说明](https://github.com/amazon-science/avgen-eval-toolkit#avs-benchmark-dataset)
- [PEAVS论文](https://arxiv.org/html/2404.07336v1#S7)
- [SyncReward发布代码](https://github.com/WeChatCV/SyncReward)

## Relations
- relates_to [[peavs]]
- relates_to [[syncreward]]
- relates_to [[云端Qwen Audio3.1Plus的20来源原生四指标分歧 2026-10-27]]
