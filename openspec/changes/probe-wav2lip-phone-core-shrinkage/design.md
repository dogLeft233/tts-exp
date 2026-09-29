# E3：保留边界的音素内部稳定化

先读公共 `../../parallel-replacement-next-20260909.md`。过去语义/Fp研究不是没做过；本轮只测试音素内部轨迹规则，不声称句义语义。旧natural-only训练policy不续跑。与旧A全时域5点平滑不同，本轮保持phone-core边缘、只移动内部，并设同支持普通平滑对照。

## 支持预先冻结

新增D/03_data/mask_manifest.json字节SHA=`956d7eb5bd7fe6b12a658191ac0ab80a9a8a6af77a0fce2aebcec30be561deec`。用D/05_drivers/drivers.json中每条各seed/condition共有的used_masks，通过mask_sha256与mask_manifest一对一连到label和natural_core_start/end。以used_masks的global_start/end为实际写入区间，并断言等于对应natural_core区间；不把padding的mask_start/end当核心。

取所有既存、非空label且label不在{sil,sp,spn,<eps>}、长度L>=5的core [s,e)，范围0..308。重复mask去重按身份，区间重叠则BLOCKED，不事后挑一个；不足5列的core保留原M、不填新边界。对第j=0..L-1列定义权重 `w_j=min(1,j/2,(L-1-j)/2)`，两端0，中间渐入；区间外0。不重新MFA/ASR，不增删音素，不扫描最有利边界。
每组必须在公共U的至少一个chunk支持中有w>0；否则全队列INPUT_DEGENERATE，不删零曝光记录或改U。该支持判定在读新分数前完成。

## 三臂

M=float64 natural mel。每个core/频率的模板mu是该core全部列M的算术平均，不使用TTS或其他样本。构造
`R_phone[:,s+j]=w_j*(mu-M[:,s+j])`。
普通平滑S为M沿时间reflect-pad2的[1,4,6,4,1]/16结果；
`R_generic[:,s+j]=w_j*(S[:,s+j]-M[:,s+j])`。
按全record Frobenius范数将generic乘 `norm(R_phone)/norm(R_generic)`。任一范数<=1e-12则INPUT_DEGENERATE，全队列停止。
共用 `a=min(0.25,0.5/max(abs(R_phone),abs(R_generic)))`。
PHONE_CORE=float32(clip(M+a*R_phone,-4,4))；
GENERIC_CORE=float32(clip(M+a*R_generic,-4,4))；N原M。
core外及每core两端必须bit-exact；所有臂形状/时钟不变。保存区间、标签、w、mu、两个范数、a、postclip范数比、边界身份与曝光。每条postclip两扰动范数比需在[0.95,1.05]且非零，才能解释机制对照；不满足不调整范数或clip追结果。

这是音素位置条件下的稳定化，不是严格分离了所有频谱/动态效应：phone模板与普通平滑仍改变不同结构，必须保留该局限。两个候选均重新生成，不能复用旧A的SMOOTH（支持/幅度不同）。

## 运行与判定

公共控制后生成32候选，总34视频/36评分，参考P/F0与完整N评分。主假设只预指定PHONE_CORE/N，按公共gain；GENERIC_CORE/N为完整披露的对照，不因它阳性改主假设。

主假设未过→NO_PHONE_CORE_GAIN_ESTABLISHED，即使对照偶然好也不判本假设成功。主通过且等范数有效，并PHONE_CORE相对GENERIC_CORE的C/D/A三个99%CI下界全>0、至少7/8组三项同正→PHONE_CORE_SPECIFIC_SIGNAL_TO_CONFIRM；主通过但机制对照未通过→NATURAL_GAIN_MECHANISM_UNRESOLVED。机制比较不另加meanC>.05。没有差异不等于两者等效。

所有16条8组保留，全U公共k0，不按曝光重定义主端点。阳性后另做独立确认与波形可达性；当前不是可播放波形头。

## 验收/BM

包 `wav2lip_phone_core_shrinkage`，CLI/文件/资源/独立validator按公共契约。测试短core不改、L=5权重[0,.5,1,.5,0]、重叠/错mask身份拒绝、频率维不混淆、边界不移动、范数及clip控制、整组无曝光、主未过而胜generic不能阳性。独立validator从D索引和原M重做所有driver与主/对照统计。
BM唯一实体 `Wav2Lip phone core shrinkage probe 2026-09-09`，记录保留边界规则、曝光分布、范数有效性、主结果和机制边界。
