"""Audit full endpoint geometry and save a report, chart, source and hashes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import json,hashlib,shutil,sys,csv,zipfile,re
import numpy as np

ROOT=Path('F:/Datasets/tts-exp/encoder_groupnorm_swap_20261020_v1')
REPO=Path('E:/Documents/tts-exp')
DELIVERY=Path('C:/Users/dogle/Documents/Codex/2026-10-20/new-chat/outputs/encoder_groupnorm_swap')
ARMS=('native','swapped','self_permuted')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    full=json.loads((ROOT/'terminal_full_geometry.json').read_text());rows=json.loads((ROOT/'layer_metrics.json').read_text())
    protocol=json.loads((ROOT/'protocol.json').read_text());assert sha(Path(__file__).with_name('run.py'))==protocol['source_sha256']
    centered={};cov={};qa_errors=[]
    for mid in ('hubert_base','wav2vec2_base'):
        for split in ('train','test'):
            for arm in ARMS:
                a=np.load(ROOT/mid/split/arm/'scale1/last_hidden_state.npy').astype(float)
                a/=np.sqrt(np.sum(a*a,axis=1))[:,None];a-=np.mean(a,axis=0)
                centered[mid,split,arm]=a;cov[mid,split,arm]=a.T@a
    def independent_cka(ka,kb):
        cross=centered[ka].T@centered[kb]
        return float(np.sum(cross*cross)/np.sqrt(np.sum(cov[ka]**2)*np.sum(cov[kb]**2)))
    for row in full['cross']:
        value=independent_cka(('hubert_base',row['split'],row['hubert_arm']),('wav2vec2_base',row['split'],row['wav2vec2_arm']))
        qa_errors.append(abs(value-row['unit_direction_CKA']))
    for row in full['within']:
        value=independent_cka((row['encoder'],row['split'],'native'),(row['encoder'],row['split'],row['arm']))
        qa_errors.append(abs(value-row['unit_direction_CKA']))
    assert max(qa_errors)<1e-12
    qa=json.loads((ROOT/'qa.json').read_text());qa.update(full_terminal_cka_covariance_checks=len(qa_errors),full_terminal_cka_covariance_maxerror=max(qa_errors))
    write(ROOT/'qa.json',qa)
    # Compact plot reports all-frame endpoint geometry; the larger layer plot uses 128 samples for CKA.
    sys.path.insert(0,str(REPO/'.codex/scratch/report25_plotdeps_attempt01'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for i,mid in enumerate(('hubert_base','wav2vec2_base')):
        values=[next(x['mean_angle_degrees'] for x in rows if x['encoder']==mid and x['split']=='test' and x['arm']==arm and x['scale']==.5 and x['kind']=='scale_response' and x['stage']=='last_hidden_state') for arm in ARMS]
        axes[0].bar(np.arange(3)+(i-.5)*.34,values,width=.34,label=mid,color=['#1565c0','#c44422'][i])
    axes[0].set_xticks(range(3),['Native GN','Swapped GN','Own GN permuted']);axes[0].set_ylabel('Mean direction change (degrees)');axes[0].set_title('RMS 0.5 versus 1, test chunk');axes[0].legend(fontsize=8)
    pairs=[('native','native'),('swapped','native'),('native','swapped'),('swapped','swapped'),('self_permuted','self_permuted')]
    values=[next(x['unit_direction_CKA'] for x in full['cross'] if x['split']=='test' and x['hubert_arm']==h and x['wav2vec2_arm']==w) for h,w in pairs]
    axes[1].bar(range(len(values)),values,color=['#777777','#6ba0d2','#e39471','#705ca1','#c1b1cf']);axes[1].set_xticks(range(5),['Both native','H swapped','W swapped','Both swapped','Both permuted'],rotation=25,ha='right');axes[1].set_ylabel('Unit-direction CKA');axes[1].set_ylim(0,1);axes[1].set_title('Cross-encoder geometry, all 1000 frames')
    for i,v in enumerate(values):axes[1].text(i,v+.015,f'{v:.3f}',ha='center',fontsize=9)
    fig.savefig(ROOT/'swap_summary.png',dpi=160);plt.close(fig)
    # Assign the next report number only after computation, avoiding stale numbering.
    note_path=ROOT/'report_identity.json'
    if note_path.exists():identity=json.loads(note_path.read_text())
    else:
        numbers=[int(m.group(1)) for p in (REPO/'basic-memory/docs/experiments').glob('*.md') if (m:=re.match(r'^(\d+)-',p.name))]
        identity=dict(number=max(numbers)+1,slug=f'{max(numbers)+1}-encoder-groupnorm-swap')
        write(note_path,identity)
    slug=identity['slug']
    def metric(mid,split,arm,kind,scale=1.,stage='last_hidden_state'):
        return next(x for x in rows if x['encoder']==mid and x['split']==split and x['arm']==arm and x['kind']==kind and x['scale']==scale and x['stage']==stage)
    drift_lines=[];gain_lines=[]
    for split in ('train','test'):
        for mid in ('hubert_base','wav2vec2_base'):
            d=metric(mid,split,'swapped','baseline_drift');p=metric(mid,split,'self_permuted','baseline_drift')
            drift_lines.append(f"| {split} | {mid} | {d['mean_cosine']:.6f} | {d['mean_angle_degrees']:.3f} | {p['mean_angle_degrees']:.3f} |")
            arm_rows=[metric(mid,split,a,'scale_response',.5) for a in ARMS]
            gain_lines.append('| '+split+' | '+mid+' | '+' | '.join(f"{x['mean_angle_degrees']:.6f} / {x['relative_l2']*100:.5f}%" for x in arm_rows)+' |')
    cross_lines=[]
    for split in ('train','test'):
        for h,w in pairs:
            x=next(x for x in full['cross'] if x['split']==split and x['hubert_arm']==h and x['wav2vec2_arm']==w)
            cross_lines.append(f"| {split} | {h} | {w} | {x['mean_cosine']:.6f} | {x['unit_direction_CKA']:.6f} |")
    report=f'''---
title: {slug}
type: experiment
permalink: tts-exp/docs/experiments/{slug}
status: concluded
execution_status: SUCCEEDED
scientific_status: FROZEN_ENCODER_PARAMETER_TRANSPLANT_DIAGNOSTIC
inputs: report58 same May train/test first20.005s chunks; scales0.5/1/2
outputs: F:/Datasets/tts-exp/encoder_groupnorm_swap_20261020_v1
model: report58 SHA-pinned HuBERT-base and wav2vec2-base; only firstGN gamma/beta transplanted in memory
code_paths:
- scripts/experiments/encoder_groupnorm_swap_20261020
- tests/experiments/test_encoder_groupnorm_swap_20261020.py
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
---

# {slug}

✅ 已完成40次冻结编码器前向、832个完整层比较；0训练、0TFG、0SyncNet。交换首层GroupNorm的gamma/beta会明显改变两模型本身的特征方向，但没有互换两模型的RMS敏感性。模型间末层时间几何CKA增加，但自身参数打乱也产生相似效应，不能当作变成供体的语义表征。

## 交换的究竟是什么

用户要求互换HuBERT与wav2vec2的首层GroupNorm，观察各自交换前后及彼此之间的方向。两者GN的512组、512通道、affine=True、epsilon=1e-5相同，且GN无BatchNorm式running statistics，因此完整模块的有效互换等价于按原通道序号交换已学习gamma与beta。保持各自卷积、后续网络、输入和epsilon不变；只在内存换参，不写原权重文件。另设自身gamma/beta用同一固定seed20261020排列共同打乱的对照，不搜索最优排列、不做跨模型通道对齐。

每模型两段May音频，沿用报告58先全split标准化、后取前320080采样20.005秒，再乘0.5/1/2。三臂native/swapped/self_permuted共36完整前向，另4个默认重复，总40。模型为facebook/hubert-base-ls960 revision dba3bb02fda4248b6e082697eee756de8fe8aa8a、facebook/wav2vec2-base revision 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8；同报告58三文件及权重SHA全核对。模型eval、FP32/eager、TF32关闭、无梯度与optimizer；本机4060峰值reserved见completion.json，没有Ubuntu调用或重训。

## 交换本身对方向的影响

下表固定输入尺度1，比较末层全部1000帧的单位特征方向。角度是逐帧夹角的平均，非平均余弦的反余弦。

| 片段 | 接收模型 | 交换后对自身原模型余弦 | 交换后平均夹角° | 自身打乱后平均夹角° |
|---|---|---:|---:|---:|
{chr(10).join(drift_lines)}

这是一项很强的表征干预，远大于单纯RMS减半的微小变化。自身打乱参数也造成类似偏移，说明结果包含打破已学习通道匹配的效应；不能仅凭偏移把接收模型叫作供体模型。first-conv输出在换参前后逐值相同，所有非GN参数SHA不变，确认变化由GN换参进入。

## 交换后对RMS的响应

每臂分别以自己的尺度1为参照；每格为平均方向夹角° / 全特征相对L2变化%。这是音量响应，与上一节换参本身的基线偏移不同。

| 片段 | 接收模型 | native | swapped | self_permuted |
|---|---|---:|---:|---:|
{chr(10).join(gain_lines)}

测试片段HuBERT换参后方向响应0.02138→0.04227度，wav2vec2为0.41737→0.33016度，仍相差约7.8倍。wav2vec2训练片段方向响应反而从0.22720升至0.24865度，所以不能称接受HuBERT参数就统一减敏。其相对L2在两个片段下降，是另一种范数加权端点，应分开描述。自身打乱后的wav2vec2方向响应比供体换参更低，表明减敏并非HuBERT供体参数的特异证据。

首层GN的相对L2在换参后两边都下降：测试片段HuBERT0.08669%→0.07040%，wav2vec2 0.31476%→0.22678%；但HuBERT末层响应反而上升。因此后续网络及换参后的工作点也是决定因素，不能把所有差异归于GN参数大小。解析GN公式在24个尺度对照上重建最大误差1.461e-8。

将每帧单位特征的RMS改变量Delta u展平，交换前后该改变量的整体余弦，测试半尺度HuBERT为+0.02267、wav2vec2为-0.01171，接近0；说明换参还改变了音量响应方向，不能看成只把原响应乘一个系数。这里每个差值都在同一接收模型坐标中计算。所有0.5/2结果见terminal_full_geometry.json。

## 两模型相比于彼此

两个独立训练模型的768维坐标不天然对齐。直接同维余弦保留为描述数值，不能解释为语义相似度。主要补充单位化特征的中心化线性CKA：比较同一批时间帧之间的几何关系，允许正交旋转和整体尺度变化。完整末层使用全部1000帧，既无采样也无拟合旋转；逐层补充CKA使用预先固定128个时间点，分开存储，不混用数值。

| 片段 | HuBERT参数臂 | wav2vec2参数臂 | 原坐标余弦 | 单位方向CKA |
|---|---|---|---:|---:|
{chr(10).join(cross_lines)}

测试片段两者均交换的CKA从0.38488升到0.71772，训练从0.35386升到0.66569；但两者各自打乱也升到0.66137/0.61257。只让wav2vec2接收HuBERT GN时，其对原HuBERT的CKA提高；只让HuBERT接收wav2vec2 GN时，其对原wav2vec2几乎不变或略降。没有对称的“互换表征”现象。

这些CKA增高确实依赖共同时间排列：固定时间打乱控制降到约0.011–0.019。但时间结构接近不等于音素或语义保留更好，尤其交换参数已经严重改变基线。补充全轨迹参与率秩，HuBERT测试39.72→12.29，wav2vec2 7.92→9.88；并非两者统一塌缩。没有ASR、音素标签、生成或SyncNet验证，不能给出语义/感知优劣判断。

## 验证与产物

独立CPU复核832个完整层比较的全部逐帧矩、106496个采样向量及32个完整末层比较。所有12个独特native条件及4重复末层与报告58逐值一致；4个默认重复的所有层逐值一致。换参张量逐值核对供体/排列，前卷积不变，两模型最后精确恢复原state_dict，原权重文件没有修改。另26项全部末层CKA用特征协方差公式独立重建，最大误差{max(qa_errors):.3g}。几何单测2项通过，包括正交旋转保持CKA而改变原坐标余弦，以及正增益不改变单位方向。

代码scripts/experiments/encoder_groupnorm_swap_20261020/{{run,geometry,analyze,full_geometry,finalize}}.py；测试tests/experiments/test_encoder_groupnorm_swap_20261020.py。protocol.json在运行前绑定源、输入、参数网格和比较定义。full_geometry.py追加全帧endpoint分析，没有新增编码器forward；128点CKA在旧逐层JSON保留，正式末层表以terminal_full_geometry.json为准。源和测试快照、raw末层数组、逐层样本/逐帧矩、参数副本、原始统计均保留；manifest.json逐文件SHA，未声明云备份。pretrained helper按原SHA引用报告58代码。

## 结论边界

这次直接验证了：已学习GN仿射参数确实影响特征方向及尺度响应，但按原通道编号互换gamma/beta不足以互换两个编码器的行为。通道所依赖的卷积滤波器、后续网络以及二者共同训练形成的匹配不能忽略。自身打乱对照只有一个预定排列、只有两段May和两个checkpoint；不估总体显著性或供体特异效应。仍不能从这次冻结编码器干预推断报告56训练分差的因果贡献，也不能建议直接换GN部署。

## Observations
- [status] concluded，40冻结前向、832层比较，0训练/TFG/SyncNet。
- [result] 换GN仿射参数大幅改变自身特征方向，RMS响应没有互换。
- [result] 跨模型末层CKA增高也见自身打乱对照，不等于语义向供体迁移。
- [limitation] 原通道未对齐；两段May/单固定排列，不作总体或下游质量结论。

## Relations
- follows [[58-encoder-layer-epsilon-diagnostic]]
- relates_to [[56-may-postnorm-scale]]
- supports [[研究主线与证据地图]]
'''
    (ROOT/'report.md').write_text(report,encoding='utf-8');(DELIVERY/'report.md').write_text(report,encoding='utf-8')
    for name in ['terminal_full_geometry.json','swap_summary.png','qa.json']:shutil.copyfile(ROOT/name,DELIVERY/name)
    sources=ROOT/'source';sources.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):shutil.copyfile(path,sources/path.name)
    shutil.copyfile(REPO/'tests/experiments/test_encoder_groupnorm_swap_20261020.py',sources/'test_encoder_groupnorm_swap_20261020.py')
    progress=json.loads((ROOT/'progress.json').read_text());progress['status']='COMPLETED';write(ROOT/'progress.json',progress)
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in ROOT.rglob('*') if p.is_file() and p.name not in ['manifest.json','delivery_receipt.json']]
    write(ROOT/'manifest.json',manifest);write(ROOT/'delivery_receipt.json',dict(passed=True,files=len(manifest),bytes=sum(x['bytes'] for x in manifest),manifest_sha256=sha(ROOT/'manifest.json'),training_updates=0,report=slug))
    with zipfile.ZipFile(DELIVERY/'reproduction_kit.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in list(sources.glob('*.py'))+[ROOT/n for n in ['protocol.json','qa.json','report.md','manifest.json','terminal_full_geometry.json','layer_metrics.csv']]:archive.write(p,p.relative_to(ROOT).as_posix())
    print(json.dumps(dict(report=slug,files=len(manifest),bytes=sum(x['bytes'] for x in manifest),full_cka_maxerror=max(qa_errors)),indent=2))

if __name__=='__main__':main()
