---
title: 47-synctalk-amplitude-frontend
type: experiment
permalink: tts-exp/docs/experiments/47-synctalk-amplitude-frontend
status: concluded
tags:
- synctalk
- amplitude
- frontend
- reproducibility
---

# 47 — SyncTalk 音量前处理与真实权重响应核查

日期：2026-10-16。✅ 前端审计完成；未训练、未渲染图像或视频、未计算 SyncNet。工程执行机 Ubuntu/V100，代码与其他实验隔离。

| 修订 | 日期 | 内容 |
|---|---|---|
| Initial | 2026-10-16 | 固定官方代码和权重，实际前向、原始数组独立复核及资源限制 |

## 结论与推荐起点

**推荐 SyncTalk AVE + May 开始后续训练电平×测试电平预检。** AVE 没有逐段波形增益归一化；纯标量增益通过绝对 log-mel、真实 AVE 权重以及已训练 May 的音频 MLP/注意力传到生成器条件入口。HuBERT 则在整段波形上进行零均值/单位方差处理，几乎抵消此干预，适合作为近似幅度不敏感的对照，不能宣称严格不变。方法可实施不代表训练能带来质量收益。

| 编码器 | 前处理操作 | 幅度是否传入编码 | 实际验证状态 | 是否适合训练电平对照 |
|---|---|---|---|---|
| AVE | 单声道16k、预加重、绝对dB、固定范围映射/clip；编码器BN用固定统计 | 是；编码相对L2变化8.783%–14.628%；May音频MLP/注意力后11.993%–18.643% | ✅ 官方11.3MB权重严格加载；13条件及重复；May真实语音和checkpoint入口前向 | 首选；从WAV重算，冻结AVE，只训练生成器 |
| HuBERT large | 整段波形零均值/方差归一化，分母含1e-7；网络内逐时间步LayerNorm | 大部分抵消；实际hidden相对L2变化0.003816%–0.052536%，仍高于零重复误差 | ✅ 官方1.262GB权重严格加载；13条件及重复；CNN/投影/hidden均保存 | 适合作近似不敏感对照；不适合把原始增益当强条件分离 |
| DeepSpeech 0.1.0 | int16转换、MFCC含log-energy、上下文拼接后整矩阵单一mean/std | MFCC前端仍有响应；PCM16诊断相对L2变化2.032%–5.848%；完整编码未知 | ⚠️ 前端实际运行；TF冻结图及logits未运行 | 暂不首选；先固定PCM16合同并部署真实图 |

范围仅为5条实际语音、13安全条件、8个非零增益比较；不当作5位独立说话人，不作语言/人群泛化或收益显著性结论。项目历史语音只用于处理链诊断，不恢复其独立效果证据资格。

## 固定代码、环境与已取得资源

- 官方仓库：[ZiqiaoPeng/SyncTalk](https://github.com/ZiqiaoPeng/SyncTalk)，commit **43079fb3f349675f6cb33031bb07262f7566cdd5**。Ubuntu直连GitHub超时，由本机官方clone→Git bundle→Ubuntu checkout；远端官方checkout未修改。
- Ubuntu独立根目录：`/home/wjj/tts-audio/synctalk-amplitude-audit-20261016`；官方代码在 `official/`，独立解释器 `env/bin/python`。环境通过`.pth`只读复用已有包，没有安装或改动既有环境。
- 实际版本：Python3.10、torch2.5.1+cu121、librosa0.11.0、soundfile0.14.0、transformers4.39.2、python_speech_features0.6。与官方测试的torch1.12.1环境不同；不宣称旧环境逐位复现。
- AVE：官方Git内 `nerf_triplane/checkpoints/audio_visual_encoder.pth`，11,300,706 bytes，SHA256 `1f75cdb5556938590c64221f6c47a8d20ab3e368d53c166023eedb12279db8c4`，严格加载无缺失或额外参数。
- [May.zip官方来源](https://drive.google.com/file/d/18Q2H612CAReFxBd9kxr-i1dD8U1AUfsV/view)：完整下载1,225,104,190 bytes，SHA256 `008fe3fb46430a92beff334e26f37eab229c717ff5b577dfbba90f46c3848d4b`。完整包保存在本机；Ubuntu仅解压36,069,525 bytes最小子集：完整WAV/cache/bs/背景/原split JSON及每split最前4帧，不称完整训练数据已落到Ubuntu。
- [trial_may.zip官方来源](https://drive.google.com/file/d/1C2639qi9jvhRygYHwPZDGs8pun3po3W7/view)：完整下载17,612,143 bytes，SHA256 `5fa488118885c53e463581121c3bf60aee602a8812ce5a1ad33c11edf0b3b445`，两个checkpoint均已解压到Ubuntu。实际使用 `ngp.pth` SHA256 `79e5c0056702357e176e9895b237bfcab1a517b85f4e468586282954e7379e79`。
- HuBERT：[facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft)，revision **ece5fabbf034c1073acae96d5401b25be96709d8**，bin大小1,262,057,559 bytes，SHA256 `9cf43abec3f0410ad6854afa4d376c69ccb364b48ddddfd25c4c5aa16398eab0`。配置、tokenizer与权重完整取得；运行副本在 `/dev/shm/synctalk-amplitude-audit-20261016/hubert-large`，这是易失存储，持久完整副本在本机 `runs/synctalk_amplitude_audit_20261016/resources/hubert-large`。
- 完整本机资产目录：`E:/Documents/tts-exp/runs/synctalk_amplitude_audit_20261016/resources`。逐文件大小、SHA和路径见 `resource_manifest.json`。

## 源码追踪：AVE

以下行号均指上述固定commit，Ubuntu前缀为 `.../official/`。

1. `data_utils/process.py:14–18`用ffmpeg抽取16k WAV，没有loudnorm/limiter/峰值归一化，也没有显式`-ac 1`；默认WAV转换会有PCM量化。主实验绕开这一步，使用DOUBLE WAV，避免把PCM量化混为纯增益。
2. `nerf_triplane/utils.py:1580–1581`的librosa.load默认转float32、mono平均、重采样到16k。整数PCM转浮点属于固定单位换算，不是按内容调节增益。主实验输入已经mono16k；May原文件是双声道16k，实际基线沿官方mono平均路径。
3. `utils.py:1584–1606`：线性预加重 `y[t]=x[t]−0.97x[t−1]`；STFT n_fft/win800、hop200（50ms窗、12.5ms步）；80维mel，55–7600Hz。当前librosa默认center=True、Hann窗，边界补齐；这些线性操作不消除纯标量增益。
4. `utils.py:1609–1615`：`20log10(max(1e-5,mel_amplitude))−20`，然后固定仿射映射 `0.08S+4` 并clip[-4,4]。它不是减去每条音频的最大dB，也不是每窗mean/std。双臂都不触及floor/clip的格子，增益d dB应使mel增加`0.08d`；±6dB即±0.48。floor对应原mel幅度1e-5；映射下界对应S≤−100，即原幅度≤1e-4，因此floor比例和下界clip比例不同。
5. `utils.py:1618–1653`：AudDataset按25fps位置裁16个mel步，窗口为[1,80,16]；长度`int((mel_T−16)/80*25)+2`。窗口是约200ms内容；它没有每窗增益归一化。
6. `network.py:9–27,60–88`：卷积/BatchNorm2d/ReLU残差堆叠，输出512维；没有向量L2归一化。预训练BN在学习时按batch×频率×时间统计，在本项目官方provider每次均`.eval()`并`no_grad()`，用预训练running statistics，**即使生成器处于训练模式也不更新AVE的BN**。
7. `provider.py:171–188,193–209`：batch64，首尾各复制2帧；训练self从`aud.wav`、外部驱动从`--aud WAV`重新提取；`aud_ave.npy`读取行已注释。缓存是数据集对象内的内存tensor，构造后修改WAV不会刷新它，重建数据集才重算。
8. `provider.py:218–220,343–344,430–437,534`及`utils.py:48–83`：按aud_id取特征，转[N,1,512]，att=2拼8个邻近时间行并边界补零。`network.py:120–137,306–321`的AudioNet_ave是512→256→128→32的MLP，AudioAttNet是对8个时间行的softmax加权求和。softmax归一化的是时间权重，**不把输入音频或编码向量归一化为单位幅度**。
9. `network.py:368–394`中density将enc_a乘由空间位置产生的aud_ch_att，再与空间/眼部特征拼接送sigma_net；无额外音频L2归一化。返回项的`.norm()`是ambient统计，不是把enc_a除以其范数。渲染部分本次只静态追踪。

## 源码追踪：HuBERT

- `data_utils/hubert.py:1–9,12–26,77–93`：sf.read解码为浮点；CLI用librosa.resample到16k；helper对二维音频取**第一声道**，与AVE的平均不同。CLI对非16k二维[N,C]使用默认末轴resample存在轴不匹配风险，本次只验证mono16k；今后应显式先选声道再重采样，单列工程修复，不能混入既有结果。
- 官方指定Wav2Vec2Processor与HubertModel，公开preprocessor `do_normalize=true`。本机transformers `models/wav2vec2/feature_extraction_wav2vec2.py:81–99,233`：每条完整输入先执行 `(x−mean(x))/sqrt(var(x)+1e-7)`，不是每20秒chunk分别归一化，也不是整个batch共享统计。理想epsilon=0时，正标量增益严格抵消；实际epsilon使极低电平残留尺度变化。13条件与此公式逐值一致，最大误差0。
- `hubert.py:27–63`：在归一化后切约20秒段，核400、步320、80样本边界覆盖；不足400的末尾跳过，按expected_T裁/补。输出50Hz、1024维；`make_even_first_dim`去除奇数末行，再reshape[-1,2,1024]写`_hu.npy`，得到约25Hz缓存。本次片长均<20秒，长段拼接规则仅静态核查。
- 模型配置 `feat_extract_norm=layer`、stable LayerNorm、24层、eps1e-5。实际依赖 `modeling_hubert.py:210`是卷积输出每个时间步跨通道LayerNorm；`:370–384`是feature projection LayerNorm；`:762`及各transformer block为hidden维LayerNorm。它们不是整段波形RMS操作，也不提供严格增益不变证明。
- 官方from_pretrained默认eval；helper有torch.no_grad，没有train调用。预计算阶段dropout、LayerDrop及训练态SpecAugment关闭；生成器训练只读缓存，不会让HuBERT转训练态。若未来单独finetune HuBERT，那是另一实验。
- `provider.py:169–170,214,235–249`读取`aud_hu.npy`或显式NPY并调维，不根据WAV的mtime/hash刷新。修改WAV后必须显式重跑提取，保存输入SHA和模型revision，禁止乘旧缓存冒充波形增益。

## 源码追踪：DeepSpeech

- `data_utils/deepspeech_features/deepspeech_features.py:48–61`：scipy.wavfile.read保留文件dtype；多声道取第一声道。`:141–153`用resampy到16k，最后**直接astype(int16)**；非16k分支的`np.float`在新NumPy已弃用，需要单独兼容修复。
- FLOAT/DOUBLE WAV是±1量纲，直接cast后全零；本次13个安全DOUBLE文件在此步全部变为0。这是输入合同错误，不是“模型对音量不敏感”。官方常规PCM16文件是±32768量纲，行为不同。大幅增益后cast可能溢出，不能靠int16 cast当limiter。
- `deepspeech_features.py:206–211`调用python_speech_features.mfcc(numcep26)。本次版本0.6的`base.py:8–32,35–64`默认25ms/10ms、预加重0.97、512FFT、26个mel功率滤波器、log、DCT/lifter22；零能量/filter值以机器epsilon保护，c0替换为log-frame-energy。
- `deepspeech_features.py:213–237`每隔一帧保留、左右各9行零上下文、拼成494维；**整条二维上下文矩阵使用单个mean和std**，无std保护epsilon。增益对log-energy列的偏移以及边界零上下文，不等同整个矩阵统一仿射变换，故不能假定此步骤严格抵消增益。
- `deepspeech_store.py:31–54`指定TF0.1.0冻结图`deepspeech-0_1_0-b90017e8.pb`，预期SHA1 `b90017e816572ddce84f5843f1fa21e6a377975e`。本次未下载/加载该图，也未部署TensorFlow；图内层、归一化和logits幅度响应**未验证**。
- `deepspeech_features.py:77–82,155–181,64–74`读取29维logits、插值到50Hz、零补上下文、每2行取16步窗口，写25fps NPY。`provider.py:165–166,214,235–249`只读缓存。MFCC及标准化前端已实际运行；完整DS特征长度/logits只是源码核查。

## 三路径的生成器入口边界

HuBERT/DeepSpeech的NPY经provider调为[N,1024,2]或[N,29,16]，再由get_audio_features按att=2取8个邻近行；可选--emb会先argmax并查embedding，这不是默认连续特征合同。本次不启用--emb。network.py:90–117的AudioNet先按win_size裁最后一轴，再以四个Conv1d/LeakyReLU和MLP压到32维；它没有BatchNorm/LayerNorm或向量归一化。HuBERT缓存仅有2个步，不能把注释的16步当实际长度。之后与AVE共享network.py:306–321的8行时间注意力及368–394的空间门控/density入口。HuBERT/DeepSpeech这一段仅静态追踪，没有用AVE的May生成器checkpoint冒充另外两种前端的受训生成器权重。

## 冻结干预与实际结果

运行前固定候选−6/0/+6dB，按峰值安全门拒绝越满幅条件，不按响应筛选样本。5条mono16k语音为官方demo/test.wav及既有`data/data/audio/{1,2,3,100}.wav`。三条低电平RMS分别0.011582/0.012973/0.008322；demo和100.wav为0.105408/0.128501。文件哈希与完整路径在statistics.json，未核验为独立说话人。

| 源 | 原peak | 运行条件 | AVE未补边长度 |
|---|---:|---|---:|
| 官方demo | 0.583740 | −6/0；+6拒绝 | 329 |
| 1.wav | 0.074463 | −6/0/+6 | 141 |
| 2.wav | 0.069672 | −6/0/+6 | 136 |
| 3.wav | 0.060089 | −6/0/+6 | 127 |
| 100.wav | 0.874451 | −6/0；+6拒绝 | 128 |

从原始解码float64逐样本乘`10^(dB/20)`，写无损DOUBLE WAV，再读回要求逐位相等。13条件纯标量最大误差0、样本数/16k/时轴保持、peak<1；不含limiter/自动归一化/有损编码。官方librosa的float32转换另列阶段，不能声称该阶段也保持float64逐位关系。DeepSpeech另做明确标记的PCM16舍入诊断，量化臂不算纯浮点增益臂。

保存阶段：DOUBLE波形、float32读取、预加重、STFT幅度、mel幅度、绝对dB、clip后的mel、AVE窗口/512编码/补边/att2输入、HuBERT processor/CNN/feature LayerNorm/投影/hidden/paired cache，以及DS的float错误cast/PCM16/MFCC/标准化上下文。逐阶段含shape/dtype/hash、RMS/peak、均值/std、最大绝对差、相对L2、余弦及长度变化。

- 8个非零比较：AVE相对L2 0.087827–0.146275，cos 0.989408–0.996392；May已训练音频MLP/注意力后相对L2 0.119932–0.186432，cos 0.982792–0.993133。高余弦不等于编码不变，方向和尺度均应报告。
- 未clip格子的±0.48公式最大绝对误差1.32642e-5；下界截断占比0.006%–30.221%，上界截断均0。增益改变下界截断支持，因此全矩阵不能简化为统一平移。
- HuBERT processor相对L2 9.0968e-6–0.00214251，cos接近1；真实hidden相对L2 3.8162e-5–5.25364e-4，cos最低0.999999862。输出/paired缓存长度在各安全增益档不变。
- DS PCM16诊断标准化上下文相对L2 0.020316–0.058480，cos 0.998290–0.999794；包含显式量化影响，未将其当作纯浮点端到端编码结论。
- AVE mel/编码、HuBERT hidden和May音频MLP/注意力重复最大绝对误差均0。观察到的差异高于本次重复误差，不等于已证明会改变视觉质量。

## May官方基线资格与缓存差异

May原WAV实际为16k双声道PCM16、242.923秒。官方mono读取后全段peak0.475616、RMS0.047606；**全段+6dB理论peak约0.94898，当前资产可安全承载此档**，训练前仍须对最终全split文件重新检查，不能推广到其他数据。

从完整May原WAV按官方AudDataset生成mel，实际19434×80、6070窗口；运行前32窗口AVE，使用官方ngp.pth严格加载AudioNet_ave和AudioAttNet，运行24个内部条件帧，输出[24,32]，全部finite，重复逐位相同。这是官方模型的音频条件基线前向，**不是完整NeRF渲染基线**。

包内旧aud_ave.npy为[6072,512]，当前provider计算后补边将为6074行。前32新编码与旧缓存行2:34相对L2为0.000125085、maxabs0.00133491，近似而非逐位一致；长度也相差2。当前官方provider本来不读这个旧AVE缓存；后续不可为了对齐旧cache偷偷修改前端，应冻结本次环境并审计aud_id、最后几帧及共同支持。

## 失败、阻塞与存储边界

- HuBERT attempt01在tokenizer尚未传完时停止，0科学结果。attempt02因torch2.5/transformers4.39的旧weight_norm参数名不兼容，自动漏载位置卷积的g/v，**结果作废**；保留原数组于共享内存的`hubert_attempt02_INVALID_MODEL_LOAD`，不参与结论。
- canonical attempt03只把官方checkpoint旧weight_g/v映射为parametrizations.weight.original0/1，剥去hubert前缀、排除原本不属于HubertModel的CTC头。全部模型参数strict=True载入且逐值对齐；有效位置卷积权重maxerror0，没有随机缺失参数、重训或前处理修改。
- Ubuntu初始根盘仅714MB空闲。CloudFS虽显示百度盘3.5TB空闲，但实际上传会占根盘写缓存；May上传到约302MB时失败并耗尽根盘，未取得完整云备份。已移除本任务的不完整上传，移动无效试跑到SHM后恢复空间。没有删除历史实验、改变现有模型环境或停止其他任务。
- GPU运行前空闲，终检无本任务compute进程。已按本机持久副本SHA核验并移出本任务66.74MB重复传输包，终检根盘可用约355MB，仍紧张；官方C++渲染扩展未构建、nvcc不在PATH，完整图像基线与训练未运行。完整May和HuBERT权重持久副本在本机；SHM不能当长期备份，百度盘不列为下载或备份成功。

## 后续训练的最小准备清单

1. **先解决持久空间。** 完整May已下载，可从本机资源目录传输；Ubuntu当前仅36MB子集。选择独立数据盘或配置有独立缓存空间的云盘，传输后核验manifest，确保完整5520 train/553 val帧与配套文件，不把8帧subset当训练集。
2. **建立独立渲染环境并做一帧基线。** 从固定commit/当前权重开始，补CUDA编译工具及必要renderer依赖；固定版本和初始化，首先完整渲染一帧与同环境重复，检查音频/帧时轴，然后才估算训练预算。本次没有为此扩装大环境。
3. **修复split合同，保留官方旧文件。** `process.py:377–412`默认前10/11 train、最后1/11 val；实际May为0–5519及5520–6072。`provider.py:126`默认test复用val；`:153–154`只有type=val截前100帧，type=test读取同一val JSON的全553帧，不能误写test也截100。独立产生train/val/test连续段JSON，并让provider明确读取test；`main.py --test_train`会回训练帧，正式评估禁止使用。
4. **隔离上下文并固定aud_id。** AVE16步mel+8行注意力跨边界，建议先保留至少0.5秒guard并核算实际感受野；不能把相邻帧随机划分当独立测试。HuBERT整段归一化和transformer上下文更广，须先按split隔离音频再编码，单一guard不能代替隔离。剪段后显式重建aud_id与原始时轴映射。
5. **从波形生成每档训练与测试条件。** 不改官方AVE的dB/clip；每个冻结电平使用独立数据根与独立workspace，self `aud.wav`替换为对应安全增益的无损波形，重建dataset并保存输入SHA、脚本SHA、模型SHA及特征SHA。`--aud`是外部驱动接口，不能把训练self/GT映射未经检查直接当训练电平切换。随机增益因AVE预计算冻结，不能在当前provider上只改WAV而不重算，须预计算冻结的gain银行/实现显式采样接口并先验证，另列协议。
6. **固定比较与判断标准。** 同初始化、预算、身份/姿态/划分，比较训练电平×同一测试句低/原/高档；AVE保持冻结，生成器训练。建议先单身份小预算可行性，再多身份多seed；归一化基线是单列条件，不能暗改所有档。使用保留真实帧的嘴部重建/同步测量及合格盲评判断质量；SyncNet用固定原自然PCM作辅助评价，同时报告距离/原时钟支持，不把涨分直接写成质量改善。

这份清单尚未执行；无新完整训练被授权启动或实际启动。

## 复现与产物入口

脚本在本机 `E:/Documents/tts-exp/scripts/experiments/synctalk_amplitude_audit_20261016/`，Ubuntu同名文件在独立根目录。AST只排除renderer依赖导入，执行的是固定commit内原函数/类定义；selected source哈希和行号随statistics保存。权重兼容映射仅限HuBERT参数名，已单列验证。

```bash
R=/home/wjj/tts-audio/synctalk-amplitude-audit-20261016
# 使用全新输出名重跑；不覆盖canonical产物。
$R/env/bin/python $R/audit.py --root "$R" --output "$R/results/frontend_reproduction" --inputs "$R/official/demo/test.wav" /home/wjj/tts-audio/tts-exp/data/data/audio/{1,2,3,100}.wav
$R/env/bin/python $R/audit_hubert.py --root "$R" --frontend "$R/results/frontend_attempt01/statistics.json" --model /dev/shm/synctalk-amplitude-audit-20261016/hubert-large --output "$R/results/hubert_reproduction"
$R/env/bin/python $R/audit_conditioning.py --root "$R" --output "$R/results/conditioning_reproduction"
```

canonical Ubuntu结果：`$R/results/frontend_attempt01`、`$R/results/hubert_attempt03`（SHM成功产物的持久镜像）、`$R/results/conditioning_attempt02`。HuBERT原statistics中的路径保留执行时SHM位置，镜像和本机文件由validator映射读取，不改写原证据。三个statistics SHA分别 `0c9e05b3…`、`aef2a302…`、`9c2be5c2…`。

本机交付目录 `E:/Documents/tts-exp/runs/synctalk_amplitude_audit_20261016`包含完整原始数组、压缩包、下载资源、resource_manifest和独立验证。`validate.py`从原数组独立核对299个stage及299个比较，再核对2个May数组，共301数组；全部SHA/形状/finite/RMS/peak/差值/标量关系/重复控制通过。损坏数组拒绝的unittest通过。不是只检查输出文件存在。

## Observations

- [status] CONCLUDED：前端数值审计完成；完整渲染和训练未运行。
- [result] AVE保留幅度，HuBERT近似消除且有epsilon残差；DeepSpeech只确认前端响应，完整图未知。
- [conclusion] AVE+May可作为训练电平研究的工程起点；本次不能回答最佳电平或质量收益。
- [report] `runs/synctalk_amplitude_audit_20261016/report.md`及同名正式报告。

## Relations

- relates_to [[研究主线与证据地图]]
- relates_to [[46-amplitude-gain-independent-validation]]
- follows [[Codex checkpoint - 2026-10-16T10-52-00Z - Training amplitude design and Ubuntu history verified]]