# 独立视觉执行：源码指纹转录更正

原视觉设计 `visual-design-root-freeze-20261002.md` raw SHA256 `0394781ac97369b1dc8d3eaf79ec43f61b888d52cac8ff585eff4f0a14f896c8` 保持不变。该设计及接口草案02的 video_features 指纹将 `cbbc8` 误写为 `cbcc8`；本补充只更正文字转录，不改变提取算法、几何定义、支持集、bootstrap、校准或实验选择。

2026-10-02 ROOT 从实际本地文件独立计算的 SHA256：

- `scripts/experiments/lrs3_tts_visual_advantage/video_features.py`：`2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4`。当前源码保持原提取 API；工作树没有该文件的改动。
- `checkpoints/mediapipe/face_landmarker.task`：`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`。这是64位十六进制指纹；未执行 packet05 中部分请求多出的末尾 `4` 是另一个转录错误。

执行计划须同时绑定原设计和本补充的实际文件字节；所有本地源码、上传目标、实际服务器源码及提取协议统一核对上述实际指纹。ROOT 已读取新 SyncNet 统计结果；本补充属于执行指纹修复，不能称为效果解读前新科学设计。新视觉几何结果尚未产生或读取。

未执行的 extractor05/plan05/packet05 原样保留。Fresh06修复 helper 要求的 `(mediapipe module, landmarker handle)` API调用、动态 dataclass 模块注册与执行侧状态/指纹门限，使用独立源码与输出名称。该修复不修改已生成800视频、冻结的54视觉端点或先前11776统计族。
