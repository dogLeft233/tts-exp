---
title: Chinese-LiPS 官方测试集 21人×5句
type: guide
permalink: tts-exp/docs/reference/chinese-lips-test21
status: qualified-with-warnings
updated: '2026-10-31'
---

# Chinese-LiPS 官方测试集 21人×5句

## 数据和位置

从官方 [Chinese-LiPS](https://huggingface.co/datasets/BAAI/Chinese-LiPS) 的test划分，以revision `db96948538811029011eee44602438a26710ecd9` 抽取21说话人×5个不同目标片段；另每人一条不同ID参考片段。105目标音频8.035750–11.957875秒，均值10.134520秒，总1064.125秒。数据卡标注 CC BY-NC-SA 4.0。

192个人数据区：`/home/data2/wangjiajun/av-evaluation/datasets/chinese_lips_test21_20261031/`；包括原48kHz stereo WAV、人脸原始MP4、16kHz mono PCM16副本、源视频1秒图、转写、选择和检验凭据。macOS `data/chinese_lips_test21_20261031/` 有16k音频、图、预览index、manifest及检验摘要；不含原始视频。以 `manifest_release.csv` 和 `prepared_final/` 为当前清单，不用早期 `manifest.json` 或 `prepared/`。各说话人的参考与目标ID不同，尚未独立核查它们在源录制的实际时间区间。

## 选择与质量边界

按source WAV长度估计8–12秒、文本规范化去重、句尾标点及同人编号间隔至少3，固定seed选择每人6个，前5目标、后1参考；不根据模型或评分器分数选样。后来只按源PCM饱和筛查每名受影响说话人的48个候选，074替换2条无饱和片段，保留冻结清单与修订记录。没有声称所有句子人工听辨或转录校验。

最终有13/105目标、3/21参考源PCM饱和，分别来自027（6条）、074（4条）、141（6条）；原始数据已经饱和，调整音量无法恢复。027和141各检查48个备选没有找到零饱和，074找到2个。预设严格无饱和分析需排除相应目标或说话人，不能把现有105对全部称为高质量主分析集。样本保留、不隐瞒。

126条视频内音轨和对应WAV的波形经±100ms窗口比较最小相关0.959，峰位范围−0.0625至+3.5625ms；未移动媒体，音轨匹配不能证明视觉嘴音同步。静态目视21人及105目标首图，102人像暗、141低清、074戴眼镜且低对比，Haar有未检出；I2V或评分器的实际检测资格未验证。未来TTS生成结果可能短于5秒，必须按生成后共同支持集计数。中文官方转录未经逐条核听；两臂音量尚未统一。

## 校验与来源

HTTP Range 从官方镜像读取 ZIP中央目录和各成员，126对原始媒体均通过ZIP成员CRC；没有完整源ZIP校验，`source/test.zip`是已中止的续传残件，不是完整文件。服务器504个原始/派生文件SHA复核零失败；本地包SHA传输一致，本机126音频与126图片SHA复核零失败。未进行TTS、TFG、SyncNet或其余评分器前向，未核实任何模型组件的完整训练集。初始和最终清单、音轨审计、静态总览、筛查记录和脚本保留在服务器。

## 后续已有克隆音频与当前人像入口（2026-10-31）

上文“未进行TTS/TFG”等描述是初始数据准备检查点，不代表后续状态。会话“查找九评分器实验中的模型差异”已完成 qwen-audio-3.1-tts-plus 的104/105成功克隆、21人各独立ID参考、每目标一次成功draw；使用既有数据集转写，0 ASR。缺024_29_F_TY_139：原文及格式变体均Engine411，失败保留，不补造音频。104条6.16–13.28秒，20条provider峰值触顶标记，未逐条听辨/ASR核查；原有源饱和标记继续保留。

本机 `runs/chinese_lips_cloud31_20261031_attempt01/`、192 `/home/data2/wangjiajun/av-evaluation/experiments/chinese_lips_cloud31_20261031_attempt01/`；`records/`、`audio_manifest.json`与`verification.json`为输入入口，provider_audio为原始24kWAV，tts16为16k单声道副本；保留未统一电平原件。记录中macOS绝对路径按上述目录映射，不直接作为Linux路径。

当前用于PC-Talk试跑的人像分配是数据根下 `hdtf_gender_matched_20261031/assignments.json`，21位音频说话人各一张不同HDTF人像，每人的5句固定同图、自然/TTS同图；按数据集F/M标签组内seed20261031随机分配。旧 `hdtf_portraits_20261031/` 是20张/21人且两人共图的历史版本，本轮不用。HDTF人像可能与生成器训练来源重合，不能称完整无泄露队列。

[[PC-Talk Chinese-LiPS21配对九评分器2026-10-31]] 已从104成功中按每人ID字典序取首句，21对42唯一视频及九评分器均完成；不是105对全量。推理输入另存RMS−24派生，不修改本数据根或云TTS原件。正式结论见[[84-pctalk-chinese-lips21-nine-scorers]]，本数据入口不替代实验报告。

## Relations
- relates_to [[中文多说话人 I2V 测试数据候选 2026-10-31]]
- relates_to [[83-pctalk-chinese20-nine-scorers]]