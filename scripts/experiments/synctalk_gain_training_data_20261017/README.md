# SyncTalk 训练电平数据准备（2026-10-17）

✅ May工程先导的准备/核验程序。用户本轮授权保存spec和准备数据；生成器训练为0。
正式任务规格在 Basic Memory `tts-exp/tasks/准备-sync-talk-训练电平先导数据`。

## 数据与资格

固定官方commit `43079fb3f349675f6cb33031bb07262f7566cdd5`、官方May.zip和AVE原权重。
完整母版包含42518个文件；6073个25fps图像/姿态帧与16k原PCM。
May为已查看的英语单身份公开素材，官方姿态/背景等预处理可能使用全段；只用于工程先导，
不称为中文、多身份、独立确认或未污染上游预训练数据。
监督全部来自真实自然讲话视频，音频为原自然语音的纯电平干预；没有TTS换轨、TFG合成监督或帧重排。
素材字节与官方一致，划分与官方不同：原train5520帧/220.8秒、val553帧/22.12秒；
新train监督4150帧/166秒，以换取独立val/test和上下文隔离。
数据量未被证明足够训练收敛：READY_ENGINEERING_PILOT只表示数据/loader可用，
不表示从头模型已充分收敛、数据量已充分或已达到正式科学证据资格。
启动训练前另冻结成本、学习曲线与数据量资格；如需声称训练结论不依赖小数据，
须在同等规则下增加训练素材或预定训练时长消融，并观察独立validation的性能和差值稳定性。

| split | 原音频支持（源帧，右开） | 监督源帧（右开） | 监督帧/秒 |
|---|---|---|---|
| train | [0,4200) | [25,4175) | 4150 / 166 |
| val | [4250,4850) | [4275,4825) | 550 / 22 |
| test | [4900,6050) | [4925,6025) | 1100 / 44 |

块间50帧/2秒弃用，块内首尾25帧/1秒只供音频上下文。
每split的−6/0/+6dB是原channel0 PCM切片的DOUBLE纯标量；不重采样/归一化/限幅。
源立体声两声道逐值相同；评分PCM单独保存，所有模型和增益共用。
主clip另避开provider监督序列首尾4行的时间注意力补零，val2段、test4段，每段250帧/10秒。
尾部未成整段的帧仍保留作重建评价，不静默丢失。

## 持久位置与云端

- 本机数据：`F:/Datasets/tts-exp/synctalk_gain_training_20261017`，避免占满E盘。
- Ubuntu工作副本：`/dev/shm/synctalk-gain-training-data-20261017`，易失，不是备份。
- 百度网盘：`tts-exp-research-models/synctalk-gain-training-data-20261017/`。
  从Windows B盘上传，Linux另一客户端独立读回SHA；实际状态以cloud receipts为准。
- May.zip原SHA：`008fe3fb46430a92beff334e26f37eab229c717ff5b577dfbba90f46c3848d4b`。
- 派生归档 `prepared_data_verified.tar.gz` 包含音频/划分/共享链接/全部AVE数组/QA，
  不重复包含1.369GB源图像，恢复时还需要May.zip。

## 重建与验证

`prepare.py`需numpy，源文件原样提取。Windows资产目录用junction，Linux用相对symlink；
二者均只读共享一个母版，不复制三份视频。

```bash
python prepare.py --archive May.zip --protocol protocol.json --output /path/to/fresh_dataset
python qualify_provider.py --dataset /path/to/fresh_dataset --official /path/to/official_checkout --output /path/to/fresh_features
python validate.py --dataset /path/to/fresh_dataset --archive May.zip --features /path/to/fresh_features --output /path/to/new_qa.json
python audit_source_clock.py --dataset /path/to/fresh_dataset --output /path/to/new_clock_qa.json
```

`qualify_provider.py`需已有torch/librosa/scipy/OpenCV环境及CUDA，仅执行官方AVE与数据loader。
AST取原函数避免导入renderer扩展；仅两处显式修正：test读test JSON，val不截100帧。
strict原AVE加载、按原batch64与eval编码，保存padded与按aud_id选出的aligned特征；
每split的0dB再完整编码一次，原数组保存供逐位核验。first/middle/last实际collate均验证。
新训练入口必须复用这两处修正及独立root，禁止未修正官方`--test`、`--test_train`、`trainval`，
禁止把旧aud_ave.npy或其他增益的缓存混用。未来训练可重新提取相同特征或使用带完整identity检查的缓存，
不能把“文件存在”当缓存有效。

独立validator不导入producer，逐文件比较封存ZIP和提取字节，完整解码17400幅监督图像，
重读5800个landmark和姿态；读回九个驱动、原PCM、所有AVE数组及0档重复。
clock审计核对源MP4的9个抽取帧索引与三个12秒音频区间；证明未引入抽取偏移，
不把它称为原视频人类感知同步已证明。分层contact sheet固定等距抽样，不能据此声称全帧人工审核。

## 已知失败与当前边界

provider_attempt01因隔离命名空间漏DataLoader、attempt02漏packaging.version，在完成loader资格前失败；
原日志保留。attempt03补齐原依赖后成功，不修改原音频/网络计算。
Windows产生的JSON使用CRLF，Linux为LF；跨端join保留原件并单列两端SHA，
仅在差异严格等于换行且解析内容一致时接受。源素材、WAV和NPY仍要求字节同。
初次Linux云端查May.zip时上传尚未完成，未创建假成功receipt；以后续完整独立SHA读回为准。

中文正式队列须每身份至少300秒train/60秒val/120秒test，另加隔离上下文，真实单人AV，
来源/录制段/身份/语言/暴露与质量留档；不能用AISHELL纯音频或缺失的HDTF路径充数。
本次没有启动TFG训练、TTS替轨、净化、HuBERT生成对照或多身份确认。
