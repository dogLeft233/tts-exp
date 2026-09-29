## Why

最新 content-residual 续跑已完成：控制通过，但 CORRECT/N 的 ΔSync-C=+0.008，95% CI=[−0.029,+0.040]，固定 anchor 也未建立改善，终态 NO_INCREMENT_ESTABLISHED。历史 shift、phone-aligned mel、MAG/ENV 均未建立稳定 replacement 收益；masked 条件优于 NAT_ONLY 不能代替优于完整 natural。

继续训练通用生成头缺少正面依据。下一步先回答更基础的问题：已有 MFA-linear/TTS 驱动视频，相对 natural 驱动视频，是否在相同真实时间上更接近真实口型？已有 133 条视觉缓存，但父实验仅115条合格、21/23组有合格数据，原门槛122条未达到。用一次CPU缓存审计给出缺失数据的保守界限，有助于判断是否值得投入新的视觉教师确认。

## What Changes

- 新增独立、只读父资产的视觉诊断；固定133条/23组，使用三臂共同真实帧支持。
- 保留原 eligibility；新增共同支持要求只会减少可观测记录。所有不可观测记录以最坏/最好界限进入原分母。
- 输出有限历史队列的方向界限、缺失原因、可审计决策和BM结论。
- 零GPU、零生成、零训练、零SyncNet评分，不修改父BLOCKED结论。所有实现任务留给下游。

## Impact

新增 capability `lrs3-visual-teacher-missingness-audit`，建议实现于 `scripts/experiments/lrs3_visual_teacher_missingness/`。这是回顾性研究优先级诊断，不是replacement实验或旧Stage01的门禁修复。现有缓存是已见数据；即使正向，也只建议另行设计独立视觉确认，不启动旧Stage02或音频头训练。
