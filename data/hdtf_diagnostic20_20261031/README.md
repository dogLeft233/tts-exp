# HDTF 20说话人诊断子集

✅ 已采集，2026-10-31。本子集用于检查 TalkLikeYou 在 HDTF 条件下的绝对评分；不是论文官方测试划分的复现，也不是未见语料证明。HDTF 是 TalkLikeYou 训练来源之一。

## 数据

- 20位按官方姓名归并的不同说话人，每人2个互不重叠的10秒窗口，共40段／400秒。
- `images/`：40张对应窗口起点图片，保留镜像原片画面尺寸。
- `audio/`：40条16kHz、单声道、PCM16 WAV；没有增益归一化、降噪或移音。
- `clips/`：40段自然视频，25fps／250帧，H.264+AAC，仅供浏览和原生视频诊断；无损音频请读独立WAV，不能把这里的AAC音轨当作WAV逐字节副本。
- `original_videos/`：保留完整镜像片段；包括20条选用源和明确排除的短候选，可从原片重新提取原采样率音频和完整语句。
- `index.html`：图片、音频和视频浏览入口；`contact_sheet.jpg`：40张图片总览。
- `manifest.json`：每段时间、说话人、路径、SHA、来源及排除记录；`samples.csv`：简版清单；`validation_receipt.json`：独立核验。

## 来源与选样

视频来自第三方镜像 `global-optima-research/HDTF` 的 `videos.zip`，版本固定为 `6f9ed465f761b8378c690be4f86f090634281584`。按HTTP Range只取所需ZIP成员，逐成员通过ZIP CRC并计算本地SHA256；未下载整套ZIP，不能声称核验了完整ZIP的SHA256。上游声明的完整ZIP SHA仅作为来源信息保存。

官方元数据固定到 `MRzzm/HDTF` commit `bbde889bbab954c9a9ebcabe9962885c29f2165e`；15个元数据文件和官方目录清单均保留在 `provenance/`，每个选用视频均映射回官方YouTube链接与片段时间。不是重新从YouTube下载的原始全长视频。

固定seed20261031；仅取WDA/WRA具名说话人，跨类别与尾部编号合并身份；每人先选字典序首片，再在两类别内打乱并交替读取。源时长至少40秒，依次取 `floor25fps((duration−10)×0.2)` 和 `floor25fps((duration−10)×0.6)` 起的两个10秒窗口。短片按固定顺序排除并补选。未按SyncNet或任何TFG输出选择样本，没有性别平衡配额。

## 转录与后续使用

查询的HDTF官方发布只提供视频URL、片段时间、分辨率与裁剪信息，没有附逐句语音转录。本子集没有运行ASR，所有转录字段为null、状态为 `NOT_PROVIDED_NOT_ASR`；镜像中的视觉caption和音频embedding不能当作人工语音转录。

固定10秒窗口可能切断单词或句子。正式做同文本TTS前，应从保留原片补ASR、核对文本，并按完整语句重新切分；不能将本批窗口直接称为人工核对的完整句样本。40段来自20位说话人，后续统计应按说话人聚合或聚类。

## 复现与核验

采集脚本：`scripts/acquire_hdtf_diagnostic_subset.py`；测试：`tests/test_acquire_hdtf_diagnostic_subset.py`（2项通过）。媒体环境版本在 `provenance/local_media_environment.json`；独立核验与预览生成代码在 `provenance/validate_subset.py`。采集脚本可用已下载源片按CRC复用，重建派生媒体；应在新目录复现，避免覆盖后续人工整理。

```bash
uv run --no-project --with imageio-ffmpeg==0.7.0 --with pillow==13.0.0 python scripts/acquire_hdtf_diagnostic_subset.py --output data/hdtf_diagnostic20_replay
uv run --no-project --with imageio-ffmpeg==0.7.0 --with pillow==13.0.0 --with numpy==2.5.4 python data/hdtf_diagnostic20_20261031/provenance/validate_subset.py
```

此轮仅在本机采集和处理媒体，未运行ASR/TTS/TFG/九评分器，未登录192服务器。

来源：[HDTF官方](https://github.com/MRzzm/HDTF)；[镜像说明](https://huggingface.co/datasets/global-optima-research/HDTF)。
