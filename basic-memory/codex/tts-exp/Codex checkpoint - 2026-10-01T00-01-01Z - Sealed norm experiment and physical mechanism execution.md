---
title: Codex checkpoint - 2026-10-01T00-01-01Z - Sealed norm experiment and physical
  mechanism execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t00-01-01-z-sealed-norm-experiment-and-physical-mechanism-execution
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: b5cbd8b0bcfc732c59e3af53e9657dfd9871f9b5
started: 2026-10-01 00:01:01+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-01T00-01-01Z - Sealed norm experiment and physical mechanism execution

## Summary

完成80条方向/范数四格的320行封存，纠正评分窗口与搜索增益的解释，启动1440条WORLD物理波形，并把已确认的生成端音量作用进一步分解为mel内部与裁剪边界坐标。

## Story

用户要求从本地、Git、BM和百度盘恢复研究后，自主完成为什么TTS改善TFG的机制解释；最新授权继续使用阿里云Qwen TTS。男声100条、女声40条已经完整生成，当前重点是干预和独立消费。Root维护理论，三个gpt-6-luna/max代理分别负责数据/CPU、GPU实验和独立审计，不在排除项、方案或部分运行处结束。所有项目笔记在本地tts-exp，密码与API key不写文件或知识库。

已确认的正路径仍是固定RMS经生成端改变输出：男性80 clips/40说话人，原自然音频评分时主脸ΔC=+0.192，99%[+0.111,+0.276]；第二脸+0.135[+0.071,+0.207]，单位化特征也为正。静态CAL来源方向orth+−baseline=-0.019[-0.048,+0.011]，四个主确认门都未通过。这两个结果分别限定音量路径与统一静态方向，不能把后者当全部逐音素TTS特征的结论。

Root已直接读取纠错后的次要分析，140537检查/0错误。完整轨迹raw原生T−N在40人的C=+0.041[-0.125,+0.199]，预设guard15为+0.416[+0.269,+0.555]；FULL单位化C=+0.036[+0.017,+0.054]。正确search_gain=D0−D，FULL原生差+0.975[+0.865,+1.090]，表示零延迟到最小距离的差值，不能当C或独立口型质量。旧08字节不改。窗口边界/距离曲线几何需要与生成端音量分开解释；后续native缓存时间置换和320原生视频视觉测量已冻结，不能只选择显著的中心窗口。

方向/范数新实验已全量完成，80 clips×4条件=320行，0失败，1765条实际forward事件都记录benchmark=False、deterministic=True、确定性算法启用。q10保存z1方向配z0范数，q01保存z0方向配z1范数，按全帧规则，不额外要求混合条件inactive身份。独立消费者仍在修复真实封存结构的引用边缘：score NPZ无d/u，factor/source NPZ才有u/d；retry3的继承JSON多数保留retry2父路径，一条短T样本result已用child路径但JSON仍是parent，必须限定continuation_parent精确两路径且SHA一致。失败尝试都在打开效应之前结束并保留。attempt06已在CPU启动，尚无完整审计结论。

WORLD CAL的9060参数/波形与3810能量重建检查均PASS。原物理overlay绑定的源SHA含多一个字符，已以独立corrector重绑实际64字符SHA，100记录和7027旧SHA字段不动；loader从实际alignment_rows读取qualification、OOV和grid SHA而不是conditions。metadata-only attempt04核验160原grids、2修复grids与2LAB全部一致，不读EVAL波形。Root freeze v3继承原CAL及v1/v2，仅绑定这些执行元数据修复，不调科学公式、阈值、样本或能量残差筛选。CPU attempt02因缺少精确绑定的MFA normalizer源码退出2，未生成波形；补齐同SHA只读源码后attempt03真正运行，首样本RT/EX/SP/BOTH/F0/AP及QC波形已落盘，目标80 pairs×两recipient方向×九臂=1440。后续仍必须真实Wav2Lip/SyncNet GPU与独立消费。

旧随机方向实际float32剂量有51/22428帧超1e-6公差，不能静默放宽。新候选每次cast F32后再求实际剂量的QA完成80 contexts/240 random arms/22428活跃帧，0失败，最大误差约1.9e-8；辅助640新GPU行正在做资源/执行预检。女性40条N/T与80 MFA grids完成，38/40结构合格；潜变量主复核仍用全部40且首forward须用新剂量算子。

新mel机制规格在已读取主音量结果后冻结，明确是informed follow-up。使用实际Wav2Lip幅度mel前端，M0原音频、M1固定RMS；U为两者均严格在(-4,4)内部的bin，S为补集（含没有变化的边界bin）。q10只移植U，q01只移植S，q00=M0/q11=M1逐元素精确。不能把U称语音、S称静音或把mel干预说成纯物理能量分解。原9/27代码的等价规则已恢复但没有真实统计；新CAL20 CPU及EVAL80/320将实际运行。静态规格已审阅PASS，local_recovery在WORLD启动和native打包后负责实现，remote_recovery唯一启动GPU。

## Working State

服务器准确入口root@connect.weste.seetacloud.com:12349已记在服务器笔记，密码/API key只来自原用户消息和运行时；桥接器token/PID是易失状态，不入BM。新私有输出根/root/tts-research-system-20261001/为0700，不是/root/autodl-tmp/tts-research-system-20261001/，不写公共/autodl-pub。原50G数据盘仍紧张，已SHA/权限验证迁移2.6G纯下载缓存并以symlink保留旧路径，科学产物不删。最近系统盘约13.47GiB可用，WORLD预计累计6.75GiB；新aux/female/mel/WORLD GPU必须事先预算、共用只读输入、临时JPEG移到封存科学产物以外，并用SHA/source_path归档完成科学输入后再决定空间安排，不能再次等到盘满。

本地E盘约19.1GB、C约37.8GB、F约30.7GB。新大归档授权写C:/Users/dogle/.codex/tts-exp-artifacts/20261001/，仓库runs留transfer manifest/source_path映射；已存在core继续E盘。最多两传输通道，core占一条。core 301片中最新267片存在且本session逐片验证失败0，仍有34待传，不能称全部本地恢复。

Norm完整私有run：source_axis/norm_angle_radius_20261001_attempt08_formal_80x4，完成2026-09-30T23:06:42Z。protocol f1c082f7c255345ce027f231275dcf2917b4c390895517a8ed8aeca8d65c6562；rows36727fde4ffb652ac592dd37a11a8a112cfb430c37cb7b4c96b3c28fd57571a3；results5c389dfc421080b00e478b455bb5ec919b9ae7b5e22748835a3df84945d7a024；status28543b6a0cefd5a065cf05d368c912d9e61f2be43ab15bd0d444f92a7b2bec0f；all-files ba11e903d18868781f562554b867d5fe062f58f3b65b1b9eab414518c2bcb50a；code-seal a0c066f43e0f4e0223698d4d561e7b35a7c0efae75f16c5f92126c794b86458a。独立消费attempt06位于private independent_validation/norm_consumer_20261001_attempt06，GPU disabled，supervisor exitcode在output外。

WORLD actual runner scripts/experiments/world_phone_transfer_20261001_overlay_attempt04.py SHA8143f5d51831693d03a4b9ce1f8f3a42134b5d903ed9111f749e61cc878b3e95；local freeze runs/recovered_20260930/independent_validation/world_cal_qc_freeze_20261001_v3.json SHA0a5b694975efeb07b9d1e9c0beb10f0142bb4203ba4a026d7ff965477991ab9d。corrected overlay raw5760271d0292cfb5215585c9d8e5e015902f956de6e625eeb293431d8bdac074；preflight add6abf57a4ae38dfa4f611495f7edc22037be6ccf1991b93ba8ad516877dc9a；execution mapping05 raw309701eaea62665f0521504c2b77fb77333a04606ffab64f946926b48b08c1f9；exact normalizer a5ae2fa9c5f57e25a090dc0432738fd1d6b29655fa1139ff45a0098d3b4a79b3。CPU output world_phone_transfer_20261001_overlay/out/world_physical_eval_attempt03，control siblingoutsideoutput，PID858778/5秒alive并有首波形；运行PID与磁盘状态须重新确认。

## Changed Files

新mel规格openspec/changes/confirm-mel-level-partition-20261001/spec.md SHAabeced0f6f50a94fe49fc085af40358071dacc5091c7c796c26c5edafbe63ee8，local commit326bb7bad35f198855766b57b46e2f0e90b03ee0。CONTEXT.md更新为91行、最新主结果和私有路径，local1394459。WORLD attempt04 local b5cbd8b，exact已执行corrector commit9f1d5ca的blob SHA19ae41f4ad6f90cfe7355008233ccfead7074a019cdc02c40b97f621fc3604b9，工作树CRLF字节不同需区分。

dirty主consumer/validator/sealed writer及旧test，untracked新norm/sham/aux/random/female/WORLD/MFA脚本由代理维护；不要整体stage。仅记录实际实验QA和静态检查，没有新增或运行pytest。ignored runs/basic-memory、.codex/scratch、C归档和活跃process均为机器本地，不是已推送成果。未创建PR，当前新提交未推送。

## Verification

Root已直接读取主独立summary55988932b41c0410f10acdb87db43ac30d21315d9ef1a5e06f423462c5403f1c与纠错补充768934c1ab011bf12dfb53d0ec49592802895d74018c594b1f4f088e39ccc86a，完整指标/窗口和四门检查如上。WORLD CAL两审计c3c2865a2d2fdade5c8aae280bbed4a0fcbcc785fada32caf3e42cd578aad8d1、f05b23eda4e8d6ff213cffa1bf441979ec21c2ba8fe1009a232e094cfba4e590，v3字段、100原record、严格metadata SHA与AST最小函数差异已核；数学路径未改。新剂量QA和WORLD首波形/CPU进程由代理提供实际产物证据，root后续读完整QA/封存文件。

GitHub只读REST确认canonical repo、c4a1998 pushed baseline、当前branch PR查询空；同agent/session搜索8个coding_session/codex_session、legacy无命中，直接读H核对精确agent/session/started。Router、实验指令、Coding Session schema和checkpoint/writing技能已加载。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- runs/recovered_20260930/independent_validation/source_axis_consumer_attempt08/independent_consumer_summary.json
- runs/recovered_20260930/independent_validation/secondary_reanalysis_20261001_attempt03/independent_secondary_reanalysis.json
- runs/recovered_20260930/remote/world_phone_transfer_20261001_overlay/input_seal_attempt04/metadata_preflight_attempt04.json
- openspec/changes/confirm-mel-level-partition-20261001/spec.md

## Observations

- [result] 生成端固定RMS作用已跨两脸/raw-unit确认，静态来源方向主门未通过；完整与中心窗口raw原生ΔC显著不同，必须共同报告。
- [result] Norm80/320已完整封存但尚无成功独立消费；WORLD1440 CPU已真实生成首波形，新F32剂量80/240操作QA已PASS。
- [decision] 科学设计不因metadata/path失败调整；失败attempt保留，新attempt只修实际封存schema和精确来源引用。
- [decision] melU/S为已读结果后的坐标机制follow-up，精确拷贝原/固定两个mel端点，不给U/S附加语音/静音语义。
- [decision] 保持原支持集/完整分母/独立条件资格；不把RMS控制差值缩小写成中介比例，也不把SyncNet当可见/感知质量。
- [blocker] 仍需norm独立320审计、aux640、female640、native320视觉/缓存置换、mel320和WORLD GPU全流程；磁盘预算与真实metadata继承引用需完成。
- [next_step] 完成已封存norm的全量独立消费并核查正在生成的WORLD1440进度，按私有磁盘预算启动修正随机辅助GPU及随后物理/女性/mel主线。

## Relations

- continues [[Codex checkpoint - 2026-09-30T23-03-09Z - Verified mechanisms and private execution recovery]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[TTS 机制确认分离生成端与评分端]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]
- relates_to [[句级音量的生成响应经过哪些 mel 坐标]]
- relates_to [[来源实验的边界与搜索增益独立复核]]