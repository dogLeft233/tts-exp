---
title: Codex checkpoint - 2026-10-17T13-37-43Z - Independent recording candidates
  queued and cloud storage verified before cohort admission
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t13-37-43-z-independent-recording-candidates-queued-and-cloud-storage-verified-before-cohort-admission
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17T13-37-43Z
username: laptop-e133gsle\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-17T13-37-43Z - Independent recording candidates queued and cloud storage verified before cohort admission

## Summary
四身份完整训练仍为零TFG更新：两个原跨录制配对未过冻结几何门，失败完整封存，唯一候选分别在预处理和GPU队列中；云冷备份与大包零payload暂存读回已实际核验。

## Story
用户要从May先导扩大到四个新身份，每人6–8分钟自然素材，保留同人不同原录制test，全部资格通过后执行官方完整流程，比较训练、驱动、评分三个−6/0/+6dB轴。最终为4身份×seed42/2026×3训练增益共24模型，按身份/seed和四身份等权交付27格、曲线及原矩阵。官方固定commit43079fb3f349675f6cb33031bb07262f7566cdd5，原AVE预训练冻结、TFG共同随机初始化；原FP16/65536rays、60000粗阶段至累计102000完整epoch预算、原loss/optimizer/scaler/EMA保持不变。必须比较实际成功optimizer更新，记录AMP skips，不补步或改scaler掩盖控制失败。评分0原PCM16，±6为float64原PCM数值单位增益；source-owned ROI，guard20主/full次。用户允许百度盘解决空间，但未核验原数不能删除。

本轮查到Cathy原配对也失败，test flow median2.372264813512586px（门≤2），p953.936536358146548px（门≤5）；Debbie原失败median4.48346127047664/p956.69030642173168仍保留。训练参考点v2方法和阈值未改，不能把资格失败称为音量科学阴性。ADR040只准一次角色反转；Cathy原test完整243秒不足278秒main最小需求，ADR041允许按已下载官方metadata顺序选择下一同身份独立录制2一次，不按SyncNet/TFG结果或帧扫描选条件。Cathy主录制1仍295秒，新test2为110秒，总405秒6:45符合用户6–8分钟，spec已如实修订，不能称420秒；四人总1665秒。所有下游仍每人6000/750/1500帧及六个10秒test clips。适配后的n4实验只做条件性探索性结论，不声称总体充分性或人类感知改善。

策略是先封存每次失败、用完全独立namespace准备唯一候选，再对来源、完整原PCM/整录制AVE上下文、冻结几何、评分参考、持久归档、真实loader90条件与65536ray资源18条件重新核验，最后才版本化组合四人。源码、Task、spec、ADR038–041与测试已封存F execution_sources/independent_candidates_attempt13，156文件全SHA/AST通过。训练/渲染/评分代码存在不等于运行完成；仍0本轮TFG optimizer更新、0正式生成视频、无四人全局准入。

## Working State
权威任务为 `Tasks/准备并核验四个新身份的 SyncTalk 完整训练数据.md`；方案为 `Experiments/SyncTalk 四新身份官方完整训练电平实验 Spec.md`；几何及一次性fallback约束看ADR038/039/040/041。ROOT=`/dev/shm/synctalk-multispeaker-data-20261017`，FROOT=`F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017`。Ubuntu SSH `wjj@192.168.50.2` 使用BatchMode/已验证key；GPU V10016GB，isolated runtime_exec.sh，不修改共享环境。/dev/shm为脆弱运行缓存，F/已独立读回云对象才是持久证据。

Brown和Patrick原版本全部独立预处理、6AVE、评分参考、loader90/资源18及三个F归档preflight通过；SHM监督各仅72控制，训练前必须完整恢复。旧global runtime worker110572已核对无子进程后结束并保留原receipt/transition proof，不再等待永远失败的旧Debbie；原stage2 false不得改写成新版本pass。

Debbie候选ROOT/dataset_revisions/DebbieDingell_recording_roles_v2：合同在预处理前F封存；新main源1=290秒/test源0=130秒。全10500素材独立QA和8250 stage1通过，0 review flags。原准备worker仅因等待旧Cathy成功的跨身份依赖失败，非候选几何失败；原source/log/false不覆盖。geometry_resume_worker_receipt.json精确验证该原因和原material QA后只恢复未跑stage1及后续。13:38UTC真实PID429102，phase=train_flow，子进程457739 flow_stream.py已有7分钟；无TFG训练。早期flow持久observer PID41952/session85456会独立Fgzip核验再回收dense cache，不重复归档。post_geometry_worker_attempt02_receipt.json（WindowsPID129032/session62179）等待候选几何，继续全部监督/整录制AVE/新ROI+三评分参考/独立QA/归档/loader90/resource18/preflight，完成即停；attempt1 false保留。

Cathy候选ROOT/dataset_revisions/CathyMcMorrisRodgers_recording2_v2：官方新test YT21mNN49Pp34、main yDi_pmbanwc，F保存官方metadata/110秒ffprobe/五帧素材人审/405秒contract和selection。prepare_cathy_test_replacement.py于13:38UTC仍由flock PID435526排队共享ROOT/preparation_GPU.lock，尚不存在preparation_worker_receipt.json是正常排队状态。session52906；不得重复启动。保留原main并逐字节验6750原JPEG/FAN/parser与原PCM/RGB来源，只允许SHA匹配的原train/val BFM/BA/flow复用；新test与全量v2资格重做。post_geometry_worker_attempt02_receipt.json（WindowsPID128476/session85058）允许等待尚不存在worker receipt，第一次错误退出保留。Cathy原train flow Fgzip已独立验证且没有生成第二dense副本，flowpersistreceipt明确reused。两个候选补齐同权威OpenCV4.10 decode_canonical_rgb.py只读依赖并有独立SHA证明；旧sourcecapsules保留。活跃post控制器各捕获6个已加载本地helper，完成前不要编辑这些6个源码。

两个候选通过仍不能直接训练：四人统一版本contract/actual data path绑定、全局独立准入及版本化监督器恢复尚待实现。现有训练provider/full-supervision restore/render原PCM+landmarks/score ROI+audio/persistence+retirement都仍有BASE路径；必须全部绑定同一selected版本，保留旧failed data和false receipts。不得只替换数据目录却沿用旧test评分参考。可另写新versioned admission/helpers，勿编辑正在执行的6个helper。任何激活/旧namespace移动必须等Cathy读取旧main复用资格结束后，严格校验窄路径和relocation manifest，再重验实际加载、像素/ray、全来源和源hash。此段是未完成工作，未声称已决定某种alias设计。

空间已实际闭合两项：四人F stage1重复loose cache仅在tar/member/loose全SHA核验后退休3257085264B；两失败配对SHM canonical四视频合计1761757067B仅在F原素材和完整失败bundle核验后退休。Debbie完整失败61成员archiveSHA796861f6aaedeabcf168c9d41a262cd22857093b7b2e7a86e47d657518f72734；Cathy60成员archiveSHA624a46a0f1135e819b7238f1c8282f4f004bdea40b37b8d7b481de475227b9e5。Cathy seal manifest的外部dependency标签错写Debbie，仅加固定原manifestSHA的纠正overlay，不改原manifest/archive/numerical bytes。

failed_material_cold_backup_receipt.json实际passed=true、verified4：四F canonical共1761757067B在Ubuntu云挂载逐片及全SHA独立读回、F凭据核验后才退休本地重复副本；原视频字节云保存、原下载MP4与原PCM/metadata/失败数组保留。material/各失败身份/failed_pair_cold_storage_locator.json指向cloud_delivery/failed-v1-身份-role/transport_attempt01；未来重解码旧MKV须按locator显式恢复。不要假设原F manifest中的MKV仍在本地。Debbie cache retirement原stdout JSON与远端pretty JSON字节不同，旧stdout attempt01保存后精确远端重新复制，校验凭据已通过，没有覆盖失败证据。未动CloudDrive配置/服务/数据库。

Brown stage1云试验三片597734675B已跨挂载独立读回attempt02通过。新stream reader对相同已上传3片又独立读回attempt03实际通过，85.55秒，payload_staged_bytes=0，SHM before/after3026432000B。旧reader with_suffix造成并发bundle.partial冲突的失败/日志保留；原attempt02 canonical成功凭据不覆盖。stream版本按4MiB块校验每片与ordered whole SHA，仅保留metadata，解决后续大media归档需要第二份SHM整包的问题。最新13:36实际df SHM1.3GiB free/root568MB，Debbie dense flow进行中；F上次约10.11GB，按真实df动态判断，不盲删缓存。

analysis_contract_v1.json已在任何TFG更新前冻结并保存ROOT/F/spec SHA。analyze_factorial_gains.py将seed先在身份内平均，四身份为探索单位；保留±6/0六主轴对比、driver/scorer分解及交互，C/B/Dmin/D0/offset分开。描述性平台规则固定driver0 val MSE、coarse5–7vs8–10、fine12–14vs15–17≤2%，仅事后描述、不早停/选checkpoint/增epoch。媒体监督尾部为assemble→独立aggregateQA→analysis→final scientific review pending，不能靠源码存在宣布通过。heartbeat synctalk-27仍ACTIVE10分钟，仅关键完成/实际故障/需输入通知；全部真实结果独立QA和持久交付后才停用，保留聊天。

共享工作树1035条status，绝大部分其他任务；本任务源码和测试未提交/未跟踪，当前本地SHA未验证GitHub可解析。不要全stage、commit或清理其他任务文件。已核对同head GitHub PR查询为空。

## Changed Files
本轮主要在 `scripts/experiments/synctalk_multispeaker_full_20261017/` 新增候选namespace准备/恢复和post资格控制器、unchanged-main资格、failed-canonical退休、cloud_archive_stream.py/cloud_archive_readback_stream.py/resume_cloud_readback_stream.py及analyze_factorial_gains.py/analysis_contract_v1.json；render_delivery_supervisor.py接入stream及描述分析，assemble_results.py移除旧Cathy四clip错误标签并披露ADR040/041和405秒。对应tests/experiments十个测试文件、Task/spec/ADR041均在F snapshot13 source_manifest.json逐文件固定；更早attempt11/12保留。大型持久产物不在Git，勿把SHM缓存或scratch当成正式结果。

## Verification
此轮实际Linux namespace3控制通过0skip、Windows同组2通过1skip（symlink权限）；failed-cache3控制、Cathy unchanged-main3控制、factorial3控制均通过。stream2控制实际0.132秒通过：跨4MiB边界/ordered whole SHA/零payload暂存与同长度篡改拒绝；真实Brown597MB/三片attempt03也通过。四failed canonical真实云读回/本地退休4项通过，F source snapshot13 156文件SHA/AST通过。此前实际Ubuntu20完整训练控制0skip、Windows锁/408模拟epochs2控制、IO轮换attempt02和独立aggregate3控制仍有效；这些是toy控制，不能计24模型训练或真实27格。原Cathy/原Debbie v2几何失败完整保留。候选Debbie全素材+stage1通过，仅到train_flow；候选Cathy排队，尚无候选几何通过证据。

## References
- [当前仓库](https://github.com/dogLeft233/tts-exp)，本轮read-only GitHub repository API确认canonical；当前head PR搜索为空。
- 本地、尚未验证推送SHA `2b71639c3b555293375d248350999377168d515d`；不构造commit链接。
- FROOT/execution_sources/independent_candidates_attempt13/source_manifest.json
- FROOT/failed_material_cold_backup_receipt.json
- FROOT/cloud_delivery/stage1-SherrodBrown/transport_attempt01/readback_attempt03/persistence_verification.json
- FROOT/dataset_revisions/各候选/post_geometry_worker_attempt02_receipt.json
- ROOT/dataset_revisions/DebbieDingell_recording_roles_v2/geometry_resume_worker_receipt.json

## Observations
- [result] 本轮正式TFG optimizer更新0、正式生成视频0；四人全局准入尚不存在。
- [result] 两个原配对冻结几何失败完整封存，阈值及数值方法未放宽；两个候选独立准备正在运行/排队。
- [result] 两失败配对四canonical共1761757067B已实际独立云读回后退休F重复副本；Brown597734675B新stream读回零payload暂存通过。
- [decision] ADR040/041只允许一次规定顺序fallback，首合格保留，不按SyncNet或TFG测试选数据；最终实验披露资格后适配和n4条件性边界。
- [decision] 原FP16/65536ray及实际optimizer预算控制不变；不补步或修改GradScaler掩盖不等更新。
- [blocker] 候选冻结几何和完整post资格未完成，统一四人versioned data binding/admission尚待实现，旧false总receipt不得复用为pass。
- [next_step] 复用真实PID/receipts推进两个候选，同时实现独立版本化四人来源与训练/渲染/评分路径准入；只有全部fresh gates通过后才能启动已授权24模型。

## Relations
- continues [[Codex checkpoint - 2026-10-17T12-43-24Z - Failed geometry sealed and single recording fallback queued with independent cloud reread repair]]
- tracks [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- implements [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[041-synctalk-eligible-recording-fallback]]