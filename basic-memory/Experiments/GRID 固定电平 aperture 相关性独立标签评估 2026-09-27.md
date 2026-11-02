---
title: GRID 固定电平 aperture 相关性独立标签评估 2026-09-27
type: experiment
permalink: tts-exp/experiments/grid-固定电平-aperture-相关性独立标签评估-2026-09-27
status: concluded
---

# GRID 固定电平 aperture 相关性独立标签评估

## Observations
- [status] concluded
- [question] 英语受控 GRID 固定时间配对 aperture 形状相关；lag0 唯一主要，旧 cal-only lag−3 次要，不确认幅度或完整嘴型保真。
- [design] 16 固定标签先 REAL-only QC，16/16 eligible；三域9视图×两lag单一41帧支持。全部16进入统计，包括测量失败s13；无输出筛样、无子集分析。100000标签等权配对bootstrap，共用抽样，seed20260927。
- [result] 总体测量门15/16≥12 PASS，全必需r可算。lag0 BASE/BASE Δr=+0.005123，95%CI[−0.016918,+0.027786]，99%CI[−0.023438,+0.035164]；lag−3 Δr=+0.009024，95%CI[−0.007883,+0.025420]，99%CI[−0.013370,+0.030478]。两lag各81组合95%CI全跨零，稳健变化NOT_CONFIRMED，不称零效应或等效。
- [auxiliary] 全部输入下调7.193–15.032dB，固定canonical RAW评价音轨；G-only ΔSync-C=+0.143，95%CI[+0.041,+0.242]，99%CI[+0.009,+0.274]。ΔD=−0.113，95%CI[−0.225,−0.011]，99%CI[−0.264,+0.020]；offset差0。Sync仅同标签辅助，不冒称同帧或救主端点。
- [conclusion] 电平响应有输入范围依赖，并非加大音量必然更高分；未证明固定目标最优、模型训练电平分布或物理质量改善。英语受控/全下调不直接复制中文多上调或TTS优势。
- [boundary] 端点受到旧8cal结果启发后冻结，8cal是开发集不是独立验证；旧完整几何3/8 FAIL保持。测量失败s13全分母保留，未调整阈值。cal-only −3不确认物理时延来源。
- [verification] 独立复核PASS：432视图、5184相关、162CI、10569标量max7.94e−15，68448距离exact，1016hash及16PCM/32scalar-mel绑定。receipt SHA f0130dc45046520c54a5a7fc5102088009f2282f9fe421a343a893785ee2f6a6。
- [resources] 科学cap40→37MiB经独立保守界收紧；科学完成后仅收尾6文件≤572KiB，独立批准运行时cap33MiB；另8MiB/5GiB底/1GiB RAM/64MiB临时不变。GPU已释放，新完整生成视频临时文件已移除；本地科学资产完整，CloudDrive本轮归档0。
- [report] runs/grid_shape_correlation_evaluation_20260927/report.md；SHA256 919cf7e27b8f193385de810b1b0c6b2e3c97e4bc03ea572e942540819dd6f08e。protocol v2 SHA 4a6da1c680812a73f277b966c96d907022c7e0124b399347d4459f5949fa1aa7。完整95/99CI/逐标签结果见geometry_results.json与sync_results.json；最终hash/resource见final.json及final_verification.json。
## Relations

- follows_up [[GRID aperture 轨迹形状相关开发门 2026-09-27]]
- relates_to [[GRID 固定电平连续口部几何校准 2026-09-27]]

## Changelog

- September 27, 2026：root批准另行冻结16eval，创建planned；正式协议/代码待审，尚未读取eval媒体或模型分析。

- September 27, 2026：零eval访问时按reviewer封存v2纯工程修订；protocol SHA 4a6da1c680812a73f277b966c96d907022c7e0124b399347d4459f5949fa1aa7。显式Sync deterministic/seed、PSF源码版本、PTS rtol=0及分析环境/线程；v1保留，科学门未变。

- September 27, 2026：全部16标签完成并独立复核PASS，planned→concluded；测量15/16但全16保留，两lag各81组合未确认相关性改变，固定RAW评价音轨的G-only Sync-C +0.143且99%CI正。保留s13与旧几何FAIL，不增实验或阈值。纯资源附记在独立上界证明后收紧40→37→33MiB，科学字节不变；报告/最终hash及GPU释放、CloudDrive零归档状态封存。