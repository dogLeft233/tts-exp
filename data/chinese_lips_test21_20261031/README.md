# Chinese-LiPS 官方测试集 21人×5句

状态：已准备自然音视频候选，非全部质量通过；未生成TTS、未运行TFG或评分。

- 来源：https://huggingface.co/datasets/BAAI/Chinese-LiPS ，revision db96948538811029011eee44602438a26710ecd9；原作者标注许可 CC BY-NC-SA 4.0。
- 21位说话人，每人5条目标、1条不同ID的同人参考，总126条。105条目标8.036–11.958秒，平均10.135秒。参考片段与目标ID不同，未独立验证源录制中的绝对时段不重叠。
- 固定seed、中文文本去重、WAV字节数估计8–12秒、编号间隔>=3，得到 selection_frozen.json；之后只按原始PCM饱和情况用两条无饱和候选替换，详见 selection_changes.json 和 selection_qa.json。48条备选/受影响说话人只按饱和检查，不根据TFG或评分器分数选择。
- prepared_final/target 与 reference 是16kHz mono PCM16 WAV与源人脸视频1秒处PNG；没有统一电平、变速、补静音、去噪或生成。源WAV是48kHz stereo PCM16；raw/保留被替换文件，最终条目以 manifest_release.csv 为准。
- 13/105目标、3/21参考的源PCM存在饱和，涉及027、074、141三位；各人逐个追加筛查48条仅074找到2条无饱和可替代。饱和源保留，未被修复；主分析应预设处理或另报无饱和子集。16kHz转换本身亦有重采样/混音饱和，参见manifest。
- 126对WAV与视频内音轨在最大±100ms搜索后，最小归一化相关0.959，观察最大偏移约3.56ms；仅验证音轨对应性，不证明真实嘴音同步，也未对媒体进行时移修复。
- 1秒图像的Haar未识别人脸仅为自动检测警告，21人首图及105目标首图已做静态人工视觉查看；102较暗，141低清，074眼镜/低对比，可能影响人脸检测和I2V。未逐片人工听辨、未验证转录完整性/人声音画真值，未评估模型数据泄露或生成后5秒窗口。
- 逐项哈希与音轨审核：manifest_release.json/csv、audio_track_audit_final.json、summary.json；目录中 input原始 zip 成员 CRC 与 504 文件 SHA 读回复核通过。source/test.zip为已中止的未完成下载，只保留续传状态，不是完整源ZIP；按HTTP Range取成员，无法声称整包SHA核验。
- index.html 可离线逐条查看图片、转写、标志并播放16k音频；若只复制prepared_final，则raw/原视频留在服务器。
