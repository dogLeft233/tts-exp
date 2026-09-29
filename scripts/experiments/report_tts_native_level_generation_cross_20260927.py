"""Fixed descriptive report/plots for LEVEL GxE, no model or parameter changes."""
from pathlib import Path
import hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_native_level_generation_cross_20260927'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(x):return f"{x['mean']:+.3f} [{x['ci99'][0]:+.3f}, {x['ci99'][1]:+.3f}]"
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=read(OUT/'summary.json');v=read(OUT/'independent_validation_all.json');a=read(OUT/'independent_acoustic.json');h=read(OUT/'independent_hashes.json');p=read(OUT/'protocol.json');control=read(OUT/'cal_controls.json');main=summary['results']['raw/guard20/common71']['C']
    assert v['status']=='passed' and a['status']==h['status']=='PASS' and control['engineering_passed']
    desc=read(OUT/'frontend_description.json');pairs=read(OUT/'pairs.json');invariants=read(OUT/'invariants.json')
    lines=['# 原生全局电平与谱组合：生成×评价四格','','状态：concluded；全部100对波形门与独立复核PASS，26cal工程控制、74eval、共同71/15speaker主支持。所有输入保留，无ASR/score删样。','','## 主结果','','每个数为speaker等权均值 [99% CI]，20,000次speaker bootstrap、seed20260926。没有FWER保证。Sync-C三位小数。', '',f"identity原native T−N：**{fmt(main['baseline_T_minus_N'])}**。",'', '| 条件 | 生成G对T−N效应 | 评价E对T−N效应 | 四格I对T−N效应 | 总效应对T−N | 处理后native T−N |','|:--|--:|--:|--:|--:|--:|']
    for condition in ['EQ','LEVEL','EQ_LEVEL']:
        d=main['conditions'][condition];lines.append('| '+condition+' | '+' | '.join(fmt(d[e]['T_minus_N']) for e in ['G','E','I','total'])+' | '+fmt(d['processed_T_minus_N'])+' |')
    lines += ['', '| 条件 | 分量 | N臂效应 | T臂效应 |','|:--|:--|--:|--:|']
    for condition in ['EQ','LEVEL','EQ_LEVEL']:
        for effect in ['G','E','I','total']:
            d=main['conditions'][condition][effect];lines.append(f"| {condition} | {effect} | {fmt(d['N'])} | {fmt(d['T'])} |")
    lines += ['', '## 冻结的声学操作代数对比','','Δ是对应h相对identity的G/E/I/total效应。EQ×LEVEL = Δ(EQ_LEVEL)−Δ(EQ)−Δ(LEVEL)，是操作交互。EQ_LEVEL−EQ描述谱形接近条件下的电平效应；LEVEL−identity描述原谱条件下电平效应。全部报告，不挑显著。','','| 对比 | 分量 | N | T | T−N |','|:--|:--|--:|--:|--:|']
    for contrast,values in main['contrasts'].items():
        for e,d in values.items():lines.append(f"| {contrast} | {e} | {fmt(d['N'])} | {fmt(d['T'])} | {fmt(d['T_minus_N'])} |")
    lines+=['','## 敏感性','','raw/unit × guard20/valid/guard0主敏感性使用同一71支持；all74 valid/guard0单列。','', '| 视图 | identity T−N | LEVEL总T−N效应 | LEVEL处理后T−N | EQ_LEVEL总T−N效应 | EQ_LEVEL处理后T−N |','|:--|--:|--:|--:|--:|--:|']
    for view,value in summary['results'].items():
        d=value['C'];lines.append('| '+view+' | '+fmt(d['baseline_T_minus_N'])+' | '+' | '.join(fmt(d['conditions'][c][k]['T_minus_N'] if k=='total' else d['conditions'][c][k]) for c in ['LEVEL','EQ_LEVEL'] for k in ['total','processed_T_minus_N'])+' |')
    lines+=['','B、D、固定k3 C_anchor/D_anchor、bestlag全部同样计算，完整99CI见summary.json，不选择端点。','','## 波形及前端检验','','固定算法：r=sqrt(mean(x²))，m0=sqrt(rN*rT)，mcap=min(.98*rN/pN,.98*rT/pT,.98*rN/pEQ_N,.98*rT/pEQ_T)，m=min(m0,mcap)，g=m/r_arm。LEVEL=x*g，EQ_LEVEL=EQ*g。EQ复用父封存float64变换，恢复各臂原RMS。identity直接原PCM/32768转FLOAT32，不STFT回环，baseline不缩放。', '', '| split | n | T/N RMS dB中位 | mcap触发 |','|:--|--:|--:|--:|']
    for split in ['calibration','evaluation']:
        pp=[r for r in pairs if r['split']==split];lines.append(f"| {split} | {len(pp)} | {np.median([r['original_T_N_rms_db'] for r in pp]):.3f} | {sum(r['cap_active'] for r in pp)} |")
    lines+=['','主门在任何新评分前通过；独立显式FFT/OLA重构、标量不变量检查全部1000波形。无clip、同长度、identity逐样本精确。全部输入/输出hash核验。', '', '实际安装python_speech_features '+p['frontend_description']['package_version']+'，默认appendEnergy=True；非零能量/滤波器能量窗的理想gain只使c0加2ln(g)，c1:12不变。以下分开理想float64与保存FLOAT32波形；零能量和滤波器零值例外按窗口显式记录，未删除。', '', '| split | 操作 | MFCC计算 | c0预测残差max | c1:12误差max | 例外窗数 |','|:--|:--|:--|--:|--:|--:|']
    for split in ['calibration','evaluation']:
        for c in ['LEVEL','EQ_LEVEL']:
            dd=[r for r in desc if r['split']==split and r['condition']==c]
            for typ in ['mfcc_float64','mfcc_saved_float32']:
                err=np.array([r[typ]['nonexception_max_error_by_coefficient'] for r in dd]);exceptions=sum(r[typ]['frames']-r[typ]['nonexception_frames'] for r in dd)
                lines.append(f"| {split} | {c} | {typ} | {err[:,0].max():.3e} | {err[:,1:].max():.3e} | {exceptions} |")
    lines+=['','Wav2Lip实际hp为periodic STFT n_fft800/hop200，preemphasis .97，幅度mel→20log10、min_level_db−100/ref_level_db20、[-4,4] clipnorm；未floor/clip单元预测平移=0.08×20log10(g)。各臂预测、观测差、floor/clip比例见frontend_description.json；逐MFCC系数差/零窗mask在frontend_description/。这些是前端描述，无回归或中介比例。','','## 缓存、控制和数值独立性','', f"严格缓存复用{h['reuse_count']}个raw/identity/EQ cell；源PCM路径/hash、图像、模型、框、25fps/FFV1/batch32、MFCC窗口和Wav2Lip mel逐元素相同，帧数/L/query一致。400候选PCM数值逐元素相同；200 identity仅STFT数值重构微差，但两个网络实际输入完全相同。runtime版本完全同。首cal N/T新重复pixel/features严格相等。LEVEL/EQ_LEVEL全部实际新渲染。", '', '生成框[138,90,357,387]，SyncNet ROI[33,18,462,447]→224/JPEG；FLOAT32 WAV→MFCC float64×32768→实际float32模型窗口。joint L=min(F,PCM//640)−5，各臂cell固定同L/query；标准±15 lag，anchor k3冻结。', '', f"RAW/identity/原PCM FLOAT等幅桥接通过；+5帧delay固定共同query控制通过。cal diagonal medianlag迁移标记 anchor_transfer_passed={control['anchor_transfer_passed']}；不据cal改k/support。详细各条件N/T medianlag：`cal_controls.json`。", '', f"独立重算距离{v['distance_entries']}单元，max误差{v['max_distance_error']:.3e}；metric max误差{v['max_metric_error']:.3e}；speaker统计/99CI max误差{v['max_statistics_error']:.3e}；四格闭合max误差{v['max_fourcell_closure']:.3e}。hash验证{h['checked']}项。", '', 'σA/σV/σM/covAV在固定k3/guard20的每个四格保存于scores/*.json的bridge，z仅observer捕获存features，未干预模型；不做拟合/中介比例。共享GPU lease串行，连续资源检查，5GiB磁盘watchdog；无外部compute忽略或终止、无历史删除。', '', '## 解释边界','','LEVEL均衡全局RMS，不等于LUFS或听感响度。测试时使用同对N/T的RMS和谱；不是可部署单输入模型或独立未见数据确认。固定video换audio是评价路径；固定audio新video是生成路径的Sync-C效应，不自动代表真实嘴型提高。EQ保留父校准的短时包络附带变化；标量增益改绝对包络与能量，固定floor/压缩/前端非线性可能响应。', '', '## 产物','','protocol.json/code_snapshot/cpu_seal.json冻结协议、代码、支持；support.json、pairs.json、invariants.json、acoustic_gate.json；reuse_manifest.json与reuse_repeat_validation.json；frontend_description.json；scores/、summary.json、effects.json；independent_acoustic.json、independent_validation_all.json、independent_hashes.json。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4));colors=['#4c78a8','#f58518','#54a24b'];names=['EQ','LEVEL','EQ_LEVEL'];positions=np.arange(4)
    for j,c in enumerate(names):
        vals=[main['conditions'][c][e]['T_minus_N'] for e in ['G','E','I','total']];means=np.array([r['mean'] for r in vals]);ci=np.array([r['ci99'] for r in vals]);axes[0].errorbar(positions+(j-1)*.2,means,yerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o',capsize=3,label=c,color=colors[j])
    axes[0].set_xticks(positions,['G','E','I','total']);axes[0].axhline(0,color='gray',lw=.8);axes[0].set_ylabel('Change in T - N Sync-C (99% CI)');axes[0].legend()
    vals=[main['baseline_T_minus_N']]+[main['conditions'][c]['processed_T_minus_N'] for c in names];means=np.array([r['mean'] for r in vals]);ci=np.array([r['ci99'] for r in vals]);axes[1].errorbar(np.arange(4),means,yerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o',capsize=4,color='#444444');axes[1].set_xticks(np.arange(4),['identity']+names);axes[1].axhline(0,color='gray',lw=.8);axes[1].set_ylabel('Native T - N Sync-C (99% CI)');fig.suptitle('Raw geometry, guard20, common 71 pairs / 15 speakers');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/('level_gxe_effects.'+ext),dpi=180)
    final={'status':'concluded','validation':'PASS','protocol_sha256':sha(OUT/'protocol.json'),'report_sha256':sha(OUT/'report.md'),'summary_sha256':sha(OUT/'summary.json'),'finished_epoch':time.time(),'source_pairs':100,'eval':74,'primary':71,'speakers':15,'no_ASR_selection':True,'no_new_TTS':True,'anchor_transfer_passed':control['anchor_transfer_passed']}
    (OUT/'final.json').write_text(json.dumps(final,indent=2)+'\n')
    hashes={str(f.resolve()):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='artifact_hashes.json'}
    (OUT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n');print(final);print('artifacts',len(hashes))
if __name__=='__main__':main()
