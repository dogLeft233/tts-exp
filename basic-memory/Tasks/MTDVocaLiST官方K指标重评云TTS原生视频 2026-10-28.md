---
title: MTDVocaLiST官方K指标重评云TTS原生视频 2026-10-28
type: task
permalink: tts-exp/tasks/mtdvoca-li-st-官方-k-指标重评云-tts-原生视频-2026-10-28
status: complete
---

# MTDVocaLiST官方K指标重评云TTS原生视频 2026-10-28

用户在完成69审计后明确授权：按MTDVocaLiST官方方法重新评分既有云TTS与自然驱动视频，统计T胜比例、均差与Sync-C冲突。不是新EARS，不重发API，不重生成TFG。

## 冻结设计

run /home/wjj/tts-audio/tts-exp/runs/mtd_official_cloud_rescore_20261028_attempt01；80媒体=20自然+60云，60对、20独立来源。输入和原Sync-C配对直接复用cloud_mtd_synchformer_20261028_attempt01的已冻结inputs/sync_pairs。主K5，辅K7/9/11/13/15全部报告，分数[0,1]转换均差百分点。严格异号冲突，以全部60和非平局可比支持分别给分母，平局不能称一致。strict12和来源bootstrap另报。

直接AST加载官方test_stu.py原封不动的Dataset/calc_pdist/eval_model；测试用student_thin_200官方类strict加载公开pure权重。sys.setprofile仅在eval_model返回读取作者samplewise_acc列表，不改其公式；网络前向暂停profile减少开销。记录calc_pdist矩阵，作者六指标逐条与独立截断移动均值重算assert完全相同。原生全片 min_length−5窗口，作者31shift与−1000无效候选保留。不做lag+3校准、媒体时移、guard剔除或时间拉伸。

人脸统一原512逐帧S3FD，默认JPEG及作者Dataset Resize96下半48，去除旧256检测适配；首3自然可复用69审计的原512 JPEG（波形逐点检查原音频），所有80网络新前向。benchmark=False避免S3FD搜索workspace，不关外来GPU任务。权威脚本scripts/experiments/mtd_official_cloud_rescore_20261028.py，对应tests同名，依赖审计helper脚本，两机均在当前项目。

## 执行与恢复

两个实际测试PASS：K上下文扩大不等于扩大±1帧容忍；真实512矩阵直接官方eval_model六指标与独立归约完全相同。freeze成功，protocol含script与official源码SHA。GPU当前V100外来进程约8018MiB，不终止。当前score exec session64693，已3/80，保存scores.json/status.json/matrices，每条可断点恢复。每条约10秒加逐帧人脸检测。先执行score完成，再analyze；勿再freeze（协议脚本SHA被冻结）。

```bash
/home/wjj/.venvs/mtdvocalist-20261028/bin/python /home/wjj/tts-audio/tts-exp/scripts/experiments/mtd_official_cloud_rescore_20261028.py score
/home/wjj/.venvs/mtdvocalist-20261028/bin/python /home/wjj/tts-audio/tts-exp/scripts/experiments/mtd_official_cloud_rescore_20261028.py analyze
```

## Relations

- follows [[69-mtd-official-parity-and-shift-control-audit]]
- replaces_score_adapter [[68-cloud-mtd-synchformer-native-preference]]
- uses [[MTDVocaLiST部署]]

## 最终完成

session64693退出0，80/80新网络前向、原512人脸裁剪及作者六项归约完成。report [[70-mtd-official-native-cloud-preference]]。K5主T胜21/60=35%、N胜39/60，平均T−N−0.565642个百分点（N3.771425%、T3.205783%），来源CI[-1.320332,+.326934]百分点跨0，与Sync-C冲突33/60=55%（SyncT/MTDN30，反向3）。K7/9/11/13/15 T胜分别15/12/6/10/6，平均差−1.571506/−1.912478/−2.398864/−3.078399/−3.749404百分点，冲突45/42/39/44/48，K7/K11各3平局；其余无平局。严格12子集K5 T胜15/36，差−.716549百分点，冲突24/36。

独立CPU从80矩阵重算480指标及360配对统计PASS，无额外GPU。80PCM原音轨逐点一致，每条作者输出与独立归约exact。小JSON/CSV及80矩阵已同步Windows run，report两机保存。源码快照在远端source_snapshot，source_manifest含驱动/helper/test_stu/测试SHA。全程无lag+3校准、guard剔除、API、TFG重生。绝对正确率低、非零原生峰与±1零判据不合，不能把偏自然直接当感知真值或SyncNet偏置证明。