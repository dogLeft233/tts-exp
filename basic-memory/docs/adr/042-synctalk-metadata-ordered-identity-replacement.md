---
title: 042-synctalk-metadata-ordered-identity-replacement
type: decision
permalink: tts-exp/docs/adr/042-synctalk-metadata-ordered-identity-replacement
status: open
project: tts-exp
decided: 2026-10-17 13:58:18.668891+00:00
---

# 042-synctalk-metadata-ordered-identity-replacement

## Decision
用户明确授权：按官方元数据固定顺序换一位说话人，保留失败、不放宽门槛，更新spec后继续准备和核验，四人全通过自动完整训练及27格。冻结identity_replacement_metadata_order_v1，policy SHA 180603ad4e9d2a5d92edfb6e74a68fb824d1a10c9fe7746e966f1f7440391efd；以固定HDTF commit bbde889bbab954c9a9ebcabe9962885c29f2165e的WDA、WRA、RD annotation文件依次读取，身份按第一次出现顺序（不另按字母/下载情况排序），录制按文件出现顺序。排除已保留Brown/Patrick/Cathy、已失败Debbie及May；只用每原录制首段连续标注与镜像_000首chunk，不拼断续区间或挑后chunk。

每身份取首个连续≥278秒的main，最多取首300秒；test为元数据顺序首个不同原视频ID、≥64秒且使总素材达到360秒者，取min(420−main秒, test首连续秒)，总360–480秒。main/test角色在任何TFG或SyncNet结果前固定，不扫描clip。原数据资格完整通过者为首保留身份；若自然素材/原录制独立/原几何等资格失败，完整记录该身份与失败后才按已封存候选队列前进，不回选分数更好者、不改变门槛/预算。只补第四个身份，已通过Brown/Patrick与Cathy唯一已冻结候选保持，不缩为3人。

## Context and rationale
Debbie原pair test flow4.483/6.690px失败；按ADR040唯一反转的完整val750 flow median2.0101095402098133px仍>2，p953.938487529220711通过5也不放行。两个方向及固定门均保留，官方metadata无第三Debbie原录制。反转48108成员2142144023原B/33000派生alias完整F SHA通过，archive da2695fcabf7e6408530759b8be24427a3e3bd5f309d2e79834d64db133e4f45，原pair完整失败bundle也保留。仍0正式TFG optimizer更新/0视频，替换规则没有使用模型处理效应。

固定顺序输出15个metadata长度/来源合格候选及所有排除项。首个BenCardin：WDA annotation行12/13，0官方hSvTbV4wW8A标00:15–05:40（325秒），1官方OTg7ngyHFWw标00:11–04:55（284秒）；计划首300秒main0、首120秒test1=420秒。此前Amy与Barbara缺≥278秒连续main，未把多个不连续区间相加。镜像revision固定6f9ed465f761b8378c690be4f86f090634281584，尚待下载CRC/SHA/ffprobe/独立重复和完整预处理资格，不将annotation长度冒称实际媒体长度或自然身份已核验。

## 固定传输目录资格与当前待核验身份
固定镜像revision的完整ZIP目录SHA5fdcd80d087214ec385c3817d2965ee7a448ed77f29c71d1496c32adae9f268f实际核验：首metadata候选BenCardin缺WDA_BenCardin0_000，第二CarolynMaloney缺WDA_CarolynMaloney2_000，均记录SOURCE_UNAVAILABLE_IN_FIXED_MIRROR（未预处理/未几何/未TFG），原metadata候选和acquisition attempt01失败保留；不是科学阴性。按原封存队列继续第三ChrisVanHollen，两个首chunk齐备；没有因分数重排、换角色或另找后chunk。

Chris源0/1 CRC1a2faaba/479e55ed，SHA18aa6f842e8b2e42474115a0ea43c2a072599f9b782977d35e996945668642a7 / 30b00b54bdfbe73aaa81b474b6bf0cdc30e72b97bf9771ad94308e6d966c36b0均通过。实际ffprobe328/293秒，计划首300/120秒=420秒，25fps；原发布YouTube/Senate Democrats标题与dateText分别2019/06/24、2018/01/19已直接读取并保存原HTML SHA（这是发布日期，不冒称拍摄日）。十抽样帧单人清晰自然讲话、稳定同木背景/相似衣服、灯光和取景不同，身份标签来自官方metadata；相似布景不当成独立性充分证据，后续全量PCM/RGB重复、语境/自然监督及全部资格门仍必须通过。镜像大range传输曾断线，attempt02错误保留；attempt03仅改4MiB分段与有界重试，原ZIP CRC/源SHA不变。

独立版本F/ROOT dataset_revisions/ChrisVanHollen_metadata_order_v1在预处理前封存selection SHA86050b50da540ca1d32a438ce55f81e1d649c0882525a0f0276ce7c84bb62b4c、revision_contract SHAade2b500c2c254800ebba115a33b62996a83f84db7315d8dd92ef37fc4c59541。第四待核验身份由Chris替代Debbie；原两方向失败和全部源保留。当前候选四人为Brown/Patrick/Chris/Cathy，只有全部完整资格通过后才写统一准入并自动24模型和27格；尚未预处理Chris或训练。Cathy live六模块/远端冻结副本不可改，Chris独立helper版本化，最终全部执行ID/输入路径/初始化/ROI/原PCM/汇总/source pins一致后才能开训。

## Alternatives and consequences
保留失败配对训练违反用户门；放宽2px或再扫描Debbie片段改变资格自由度；只剩三人违反四人规模。固定非评分顺序比任意换人可复核，但这是资格后cohort适配，须在最终报告披露所有被替换/失败/排除与完整队列，结论条件于通过几何资格的四身份，不声称随机身份总体效应或无上游污染。同期Cathy405秒唯一候选继续运行，不能修改其live source capsule。

原6000/750/1500监督、六固定10秒位置、整录制音频上下文、v2 SHA37f0714cecd41cca0f531bb062b6029bad2e96fe5bc19bd3ad0827d61f3dd005及2/5px、2°/5mm等门全保留。24模型/两seed/三训练增益、FP16/65536rays/原loss/optimizer/scaler/EMA、60000/102000完整attempts与实际成功更新公平性保持。评分0原PCM16、±6 float64原PCM单位，source-owned新test ROI、guard20主/full次。新身份必须全量预处理、独立持久/archive、整录制AVE、三评分参考及独立18矩阵、loader90/资源18，然后四人统一版本fresh准入；不能继承Debbie初始化/ROI/PCM或改旧false总receipt。

## Affected artifacts
F identity_replacement_policy_v1/policy.json、ordered_candidates.json、selected_candidate.json已在下载/预处理前固定。新候选用独立dataset_revisions/BenCardin_metadata_order_v1及全部新身份标签，不把Ben数据冒称Debbie。训练/渲染/评分/汇总统一身份列表和源pin需新版本；原失败代码/记录和执行source snapshots保留。Spec与Task更新后按授权自动继续，不再重复一般训练许可。

## Observations
- [decision] 官方固定metadata文件/条目顺序选择首合格新身份，记录所有资格排除与失败，不能以SyncNet/TFG效果择优。
- [decision] 首metadata候选BenCardin0/1计划300+120秒；真实媒体及完整数据资格尚待验证，未准入训练。
- [limitation] 本轮为资格条件化的四身份探索，替换不追称最初预注册，不推人口/感知或数据充分性。

## Relations
- refines [[040-synctalk-recording-role-fallback]]
- refines [[041-synctalk-eligible-recording-fallback]]
- affects [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- affects [[准备并核验四个新身份的 SyncTalk 完整训练数据]]