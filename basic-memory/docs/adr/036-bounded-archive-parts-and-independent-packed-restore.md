---
title: 036-bounded-archive-parts-and-independent-packed-restore
type: decision
permalink: tts-exp/docs/adr/036-bounded-archive-parts-and-independent-packed-restore
status: open
project: tts-exp
date: October 11, 2026
decided: '2026-10-10T23:35:00Z'
tags:
- archive
- recovery
---

# 036-bounded-archive-parts-and-independent-packed-restore

## Context

用户已授权在当前20轮自然结束且可恢复闭包完整保存后关闭192.168.50.2。20轮、全tell原数组数学及journal封存均已实际通过。B17完整inventory/ROOT QA08/fresh06准入通过，但archive17/05实际SSH、worker及wrapper均自然rc2，远端stderr0。partial880050176B/SHA695467666eb7ca98e36da192043a4e1127acec967c1929a09f9e4fdf3d1ea079截断于634017302B inventory META body中。C盘写入期间降为0，退出后恢复约852MB；这是缓存压力的相关证据，不是已确证底层错误原因。

B是CloudDrive百度网盘挂载，3.753TB逻辑容量不代表C缓存容量。现有逐文件restore会产生约596376个云对象。最窄历史重复C包删除被自动审批策略拒绝，未启动进程或删除；不换途径重试。不能改变CloudDrive配置、DB、进程或安全策略。

## Decision

采用独立新executor18负责有界传输、完整TAR内容保存、独立完整body恢复包装及验证。原worker17、inventory17、QA08和所有历史失败保持不可变，不能改写producer身份或复用过期prestream06/admission。ADR035全部科学、来源、进程和关闭门继续有效；本决策仅改变本地完整内容包装及其来源绑定。

### Identities and admission

保留qualified inventory17及完整response/receipt/manifest和producer17 pin；新executor、wrapper、读取器、恢复器及协议各自固定来源，ROOT admission v2明确绑定两类身份与fresh07实际扫描、私有捕获、读回和seal。原manifest的archive_tool字段不改成18。ROOT QA读取既有QA08实际证据；新执行代码必须复核完整来源与exact19roots重新inventory的body/stat/hash一致性。

新fresh07全部create-only，禁止旧06时间窗复用。实际全UID scanner结束→实际archive SSH Popen须≤300秒，capacity/source等耗时步骤后在Popen前再次计算，不以本地wrapper启动代替。fresh scan仍须全部19roots及linked/external scope完整、稳定双通道、0refs/incomplete/races、实际rc0。保留reserved final04，在完整备份和恢复合格后单次运行，再独立ROOT关闭准入。

### Exact archive stream

把实际远端PAX TAR stdout按原字节顺序写入有序独立新文件；每块≤67108864B，使用O_EXCL及不跟随链接，记录顺序、逻辑offset、字节数和SHA256。原TAR header/PAX/body/padding/EOF全部保存，不重编码成员。此处“exact”指本次producer实际stdout，不要求与旧失败partial逐字节相同。关闭每块后flush/fsync、完整重读并hash及稳定stat校验。保存整个逻辑TAR的总长度和SHA，并重开所有块独立重算。通过read/seek/tell正确的拼接读取器解析，不重新组装巨大单文件。

传输stdout/stderr/stdin并行，有界≤1MiB缓冲，保留实际SSH身份/时间/rc/stderr。允许有界pipe回压，不能用communicate聚合TAR输出。大inventory response不得同时聚集raw、JSON、压缩Base64多份副本；采用有界/分帧请求或等价明确内存边界。不得丢弃META大body来规避问题。

### Independent packed logical restore

逐TAR成员解析并实际复制所有普通文件body到新的独立bodypack，bodypack使用同样≤64MiB create-only块。包括所有内部META完整body；硬链接别名按原TAR规则各有完整body；零长度普通文件保留独立index行及SHA256(empty)。目录与符号链接保留显式类型和描述；符号链接描述body若原格式存在也必须实际复制与hash。不得只建立指向TAR块的索引并称为restore。

使用有序分片索引，每片≤16MiB。每个原manifest ordinal一一对应唯一确定性短ID，保留原绝对POSIX路径/root alias/relative、type/mode/uid/gid/ns timestamps/dev/ino/nlink、全部PAX及link字段、恢复ID/bodypack offset/length/bodySHA。内部META成员另有明确namespace和全body记录；不能冒充原manifest行。禁止路径穿越、Windows casefold碰撞和静默省略行。

fresh恢复后重开独立bodypack，按索引逐body全量读取/hash，核验所有part/逻辑bodypack/index hashes、TAR member与manifest ordinal双射、零文件和全部65package12来源pins。新executor/wrapper/reader/admission等运行来源也作为create-only companion闭包保存并完整读回散列，不能只保存来源路径。任何预期计数、body bytes或hash不符均不合格。

结果称为content-verified packed logical restore，是保存所有真实body及可确定重建原POSIX绝对路径/属性的便携内容恢复；不是已物理创建的Linux树，也不证明目标主机可立即运行。源绝对SHM映射、link解析目标与所需metadata需实际验证可表达、闭合且无歧义。明天只按缺失SHM重建，HOME/system保留pin并重新测RAM/tmpfs和容量。

### Storage and failure behavior

保留B空间门：完整逻辑TAR+完整独立bodypack+index/companion+8GiB reserve，计入块/索引开销。每次准备写TAR/bodypack/index块前，独立读取C free，须≥536870912B+下一块声明上限；块关闭/fsync/hash后只在固定有限时限内等待余量恢复。该门是本地容量保护，不是远端云提交证明。无需本地大文件暂存或远端root大包。所有缓冲有界；失败保留partial及真实捕获，不续写、覆盖、自动retry或删除。

CloudDrive配置不等于远端持久化证明；没有独立服务端确认时cloud_sync_verified=false/NOT_CHECKED单独记录。不能把即时B读回等同于已云同步。

## Rationale and alternatives

分块限制单个活动云文件的写入体积，独立bodypack避免约60万物理云文件而保留每个原件全部字节。whole stream/part/body/ordinal/sourcepin多层核验仍可检测遗漏、错序、截断和来源漂移。新协议将inventory身份与新传输实现分开，避免历史清单被重新标记。

继续单个巨大TAR保留了已观测失败形态；逐源文件multipart仍产生约60万云对象；manifest-only或省略媒体无法恢复；使用Z不提供独立副本；通过删C历史包或改CloudDrive缓存配置绕过容量不采用。增加逻辑索引与未来Linux重建步骤是本方案的代价，不能宣称物理树已还原。

## Consequences and affected work

ROOT冻结本决策、admission、独立实际QA和最终正常关机；原Luna Signal负责executor18，Dev负责capture及独立源审阅，第三代理可作独立consumer复核。源码与schema接受后才允许新fresh scan/归档。今晚不启动新GPU任务或重复训练/math/seal。科学门、原自然PCM/原时钟、cohort/policy01/ADR031、未来540 Dev和封闭Confirm保持。

## Observations

- [date] October 11, 2026
- [status] open
- [decision] 新executor采用64MiB有序完整TAR块及独立全body packed logical restore，保留producer17与新executor分离来源和ADR035全部进程/完整内容关闭门。
- [rationale] 限制活动写文件并减少云对象，同时保留完整原字节、元数据、绝对SHM重建映射及可逐body验证的独立恢复。
- [alternatives] 巨型TAR、逐源小文件、manifest-only、同服务器Z副本、策略拒绝的C删除或缓存配置修改均不采用。
- [verification] 原Luna Dev和Signal完成只读源码/设计审阅；ROOT QA02实际完整hash并核验archive17失败，没有运行实现测试。

## Relations

- extends [[035-privileged-process-attestation-for-shutdown-archive]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]