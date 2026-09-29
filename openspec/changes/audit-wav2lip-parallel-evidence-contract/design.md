# E0：三路并行结果的契约与数值审计

先读 `../../parallel-replacement-next-20260909.md` 与历史复核。本分支是CPU只读证据审计，新媒体、新评分、新TFG、训练均0；不运行父runner/validator（它们会写旧run），不修改父代码/run/spec。独立小包写自己的旁路报告，查到问题不是默认重跑授权。

## 固定输入

A=`runs/wav2lip_natural_temporal_contrast_20260909_v1`；
B=`runs/wav2lip_reference_conditioning_interaction_20260909_v1`；
C=`runs/wav2lip_residual_local_response_20260909_v1`。
另复用公共P/Q/D。以下为设计时文件字节SHA，必须在prepare核对：

| 文件 | SHA-256 |
|---|---|
| A/protocol.json | 516b3c52c1267e587540395cbf122e867fb1b02732369df50988bb985612a69a |
| A/drivers/manifest.json | 9af82f121940613441a59b5c13d0e87e3ac01200f93bfd4eb5274c2267f73949 |
| A/control_analysis.json | 8607989e557a1a7ff5a8e2c32f8d0b20113f1743130f5d26896ccfbdefdbd7ef |
| A/candidate_scores/manifest.json | c92b60b67da5a4f301fb8ba1e9b09b462d67153bd66d198387aacabc2960ff6e |
| A/analysis.json | c63ca572b3fdcaf93e4a63f63da66e6aca7efd7968973c6fd179540c6cf448ff |
| B/protocol.json | 51231bd182b8555db3c0b81a57a1c5a8cf495f0a186b3584b9f9f8bf99abe6cc |
| B/reference_manifest.json | bbd7f121e8b383c308f20f1eb5300c760ae2214a154ff81ec6e1dda654a2c89e |
| B/control_scores/manifest.json | 9e5a724d3634054d4328c0d35816f06db0065ccac41368e22bdb71291adc6264 |
| B/control_analysis.json | dc79b2f53bdc5a31395e596b2ddb26824e592fb096ca2de0806d8d577b51cda0 |
| B/final.json | 5cc7bdaee6d7e21142bf949aef0250a4a3be411ebe19e4890f5d770e6b696f28 |
| B/validation.json | 74fbdf7865a28bd0264b82ac32688d1ef797225a40673ab2befefbfff0ba9206 |
| C/protocol.json | 3a20ff1f4a7a5693212655332b064a586830d4872224c6e6a9a115bc9d8b4870 |
| C/exposure.json | 666e19efec2daa811088f4e72a35e7872d8fe124006f9b9b58dcb4a1c13f4e3b |
| C/reused_manifest.json | 99d8867d41f46c062ff529d1cb42993af2cfe297f07ad5dabaf977963bac8110 |
| C/analysis.json | 283f0c2605ab291a86ec6eae79b152fef33e36ac8f08299634e9f02b3f25b076 |

A replay/parity列表在control_analysis.json，不存在A/control_scores/manifest.json；不要补造。所有score经worker到visual/audio_embedding/matrix及sha256。A/B媒体可CPU解码验证PCM/PTS/帧数/像素；C的原媒体不读，只沿P/Q既有绑定声明来源局限。
额外冻结旧三个change的proposal/design/spec、旧公共契约与当前三个实现包源码hash，比较父protocol中历史spec/code绑定。缺少历史代码绑定记not_proven，不用当前hash填补历史。当前源码缺陷可被证明，旧产物的执行源码身份另列。

## 三项独立计算

必须完整读旧A/B/C各design；原数值筛选按其原seed20260910、20000次、99%CI，父控制seed20260909、10000次、95%。禁止以新E2/E3/E4 seed替换历史。

A：从原M独立实现binomial核/幅度/clip，两候选数组与chunk逐值复算；从embeddings重建全部N/SMOOTH/SHARP距离与U/k0/C/D/A；复算全部组均值、联合正组数、CI与标签，比较A/analysis。重新检查2 replay/2 parity、原P匹配域控制、PCM/PTS/媒体身份。不能导入A.transform或其统计helper。

B：先复现存盘legacy 13/16及损伤mean=1.2678824125656059、95%CI=[0.975656394793519,1.5851212186472756]。然后按原B design而非旧实现重算F46已知延迟：
`B[r,j]=dist(V_N[r],A_N[r+j-15])`；
`T[r,j]=dist(V_N[r],A_DELAY[r+j-10])`；
U=30..57、j=0..30，自然offset=15-argmin(mean B)，delay offset=10-argmin(mean T)。
全部真实支持q=15..72 / 20..77；同一视觉embedding逐值相同；音频须精确补零3200 samples。损伤仍来自legacy delay曲线在F46_N自己的k0处，不使用补偿T算损伤。报告旧失败ID/预期峰越界/新通过ID、embedding平移误差及曲线差。新通过数未知，不能硬编码16/16。
控制仍要求≥14/16 offset、损伤CI下界>0且≥7/8组正；边界解释完整另要求旧失败全恢复且16/16。独立重查F0 replay像素/矩阵对P、F46 repeat、parity和F46参考来源。
只报告 `f46_control_as_spec=PASS|FAIL|not_available`；F46_C仍不存在，不计算交互I，不改旧final的CONTROL_FAILED，不发放candidate授权。

C：从D used_masks独立构造K、官方93个chunk、5帧视觉支持并集B_r与e_r，不能导入C.support。注意最后chunk起点应为292，不是无条件int(92*3.2)=294；主U虽不碰尾部，完整support产物仍需验。按旧design独立重算4臂矩阵/父三contrast、beta、local_gain、CI。联合正组数用同一组同时beta>0且local_gain>0；不得用两个边际各≥7替代。每一个e_r=0行都验visual及31列distance，不只是整条e全0。不可辨识组保留null诊断，不用0填未观测估计。报告各差异是否影响当前阴性标签。

同时检查A/B analyze隐式生成、final未绑定validation、独立验收缺失等契约项，输出有文件/行号/反例的checklist。只用mock/临时合成fixture验证路径，不调用真实父analyze。

## 输出与边界

最小 `runner.py`（prepare/analyze/all）、`validate.py`、少量纯NumPy函数。CLI包 `wav2lip_parallel_evidence_contract`、run-id `20260910_v1`，仅CPU threads2。
protocol冻结后计算；旁路产物input_audit、per_record、recomputed_analysis、discrepancies、validation、review、final、result.md。独立validate必须从原数组第二条实现复算核心结果，不能只看discrepancies或producer布尔；矩阵1e-4、统计1e-6、ID/索引/标签精确。保存同一固定bootstrap索引供复验，同时独立生成核对其身份。

每一路分别给 `numeric_reproduction`、`original_spec_compliance`、`recomputed_decision`；不把旧validator返回PASS当独立证据。已发现契约违例即使数字不变也输出 `CONTRACT_VIOLATION_REPRODUCED`，完全无违例才 `EVIDENCE_REPRODUCED`；缺损导致无法审计为BLOCKED并保留已完成子项。工程审计可以PASS而被审计旧实现不合规，二者字段分开。

数值不变只说明相应旧阴性仍可复现；若数字/标签变化，旧值和新值并列，科学结论以新旁路审计及其局限说明。所有replacement/训练/泛化授权false。B即便新控制通过，也仅建议另写纠错续跑协议，不补16个F46_C；原问题尚未回答。

最低测试：old k=27/28峰越界而平移域恢复；错误10-j/15-j符号；“6组联合正、7组各自正”必须不过门；尾chunk292；逐行零曝光漏检；篡改analysis后重签仍失败；像素不同但U相同不算replay；缓存缺失不是科学阴性。

BM唯一实体 `Wav2Lip parallel evidence contract audit 2026-09-09`。主agent最终将审计结果同步原A/B/C实体；本分支并发执行时只写本实体，避免多人改旧笔记。
