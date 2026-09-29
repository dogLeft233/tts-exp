"""Report only: no tuning, filtering, inference selection or waveform changes."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_native_spectrum_generation_cross_20260926'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(r):return f"{r['mean']:+.3f} [{r['ci99'][0]:+.3f}, {r['ci99'][1]:+.3f}]"
def main():
    s=read(OUT/'summary.json');v=read(OUT/'independent_validation_all.json');assert v['status']=='passed'
    c=s['results']['raw/guard20/common71']['C'];p=read(OUT/'protocol.json');ag=read(OUT/'acoustic_gate.json');cc=read(OUT/'cal_controls.json');assert cc['engineering_passed']
    lines=['# 原生长期谱中点匹配的生成评分四格','',
      '历史static cloud image3的100对原N/T；旧26cal工程、74eval，主要结论与敏感性共同71对/15位说话人。冻结单次EQ与ENVeq，先通过全eval声学门及独立重构，再评分。每个identity/EQ/ENVeq都重新生成；没有使用旧视频特征。',
      '', '## 主要结果（raw guard20，speaker等权，逐比较99%CI）','',f"原配T−N：{fmt(c['baseline_T_minus_N'])}。",'',
      '| 处理 | 处理后原配T−N | 原配差变化 | N的G | N的E | T的G | T的E | T−N的G | T−N的E | T−N的I |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for h in ['EQ','ENVeq']:
        b=c['conditions'][h];lines.append('| '+h+' | '+' | '.join(fmt(x) for x in [b['processed_T_minus_N'],b['total']['T_minus_N'],b['G']['N'],b['E']['N'],b['G']['T'],b['E']['T'],b['G']['T_minus_N'],b['E']['T_minus_N'],b['I']['T_minus_N']])+' |')
    lines+=['','EQ−ENVeq预设同分量对照：','','| 分量 | N响应 | T响应 | T−N响应差 |','|---|---:|---:|---:|']
    for component in ['G','E','I','total']:
        b=c['EQ_minus_ENVeq'][component];lines.append('| '+component+' | '+' | '.join(fmt(b[a]) for a in ['N','T','T_minus_N'])+' |')
    lines+=['','把原生长期谱形拉近约一半后，主要raw guard20的TTS优势没有缩小，反而由0.438增至0.492，差距变化+0.054 [0.005,0.099]。相对ENVeq的预设净对照仍为+0.050 [0.006,0.093]；这不支持“本次处理所拉近的长期谱差驱动原生优势”的预期。', '', 'EQ−ENVeq的T−N响应差：E为+0.030 [0.001,0.061]，G为+0.019 [−0.016,0.048]、I为+0.001 [−0.009,0.011]。可辨认到评分音频端响应；G/I跨零不能称没有效应，也不计算中介比例。固定k3 anchor净总响应+0.049 [0.011,0.088]。共同71的raw valid及unit guard20净总CI跨零，其余视图见全表；不同几何数值尺度不可直接相比。']
    lines+=['','## 全部预设指标及共同支持敏感性','', 'C=median31−min31，B=median31，D=min31；anchor固定k3不重估。四格q00=id/id、q10=h/id、q01=id/h、q11=h/h；G=q10−q00、E=q01−q00、I=q11−q10−q01+q00。所有原配处理后差与变化CI分别给出。','','| 视图 | 指标 | 原配T−N | EQ后T−N | EQ变化 | ENVeq后T−N | ENVeq变化 | EQ−ENVeq总响应差 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for view,block in s['results'].items():
        for metric in ['C','B','D','C_anchor']:
            b=block[metric];eq=b['conditions']['EQ'];env=b['conditions']['ENVeq'];lines.append('| '+view+' | '+metric+' | '+' | '.join(fmt(x) for x in [b['baseline_T_minus_N'],eq['processed_T_minus_N'],eq['total']['T_minus_N'],env['processed_T_minus_N'],env['total']['T_minus_N'],b['EQ_minus_ENVeq']['total']['T_minus_N']])+' |')
    lines+=['','## 声学命中与附带变化','', '| Split | 谱距raw median dB | EQ median dB | 逐pair减少median | ≥25%比例 | ENV误差median/max dB | EQ/ENVeq R变化median |','|---|---:|---:|---:|---:|---:|---:|']
    for split,ac in ag['splits'].items():
        dist=ac['pair_spectrum'];r=ac['R_absolute_relative_change'];e=ac['ENVeq_envelope_db_rmse']
        lines.append(f"| {split} | {dist['raw']['smooth_mean_log_shape_db']['median']:.3f} | {dist['EQ']['smooth_mean_log_shape_db']['median']:.3f} | {ac['spectrum_reduction']['median']:.2%} | {ac['fraction_reduction_atleast25']:.2%} | {e['median']:.4f}/{e['max']:.4f} | {r['EQ']['median']:.3%}/{r['ENVeq']['median']:.3%} |")
    lines+=['','所有长度、finite、RMS、无clip、float64 identity、谱距、包络和R门均PASS；独立FFT/OLA/包络20次重构和门复现PASS。旧26cal的208个FLOAT32 PCM逐样本完全一致。', '', '| Eval条件 | 包络vs raw median dB | common voiced F0 median abs cents | voicing disagreement median |','|---|---:|---:|---:|']
    for h in ['identity','EQ','ENVeq']:
        b=ag['splits']['evaluation']['metrics_by_condition'][h]
        pitch=read(OUT/'f0_descriptive.json')['summary']['evaluation'][h]
        vals=[b['envelope_db_rmse_vs_raw']['median'],pitch['f0_common_voiced_median_abs_cents']['median'],pitch['f0_voicing_disagreement_fraction']['median']]
        lines.append('| '+h+' | '+' | '.join('NA' if x is None else f'{x:.6f}' for x in vals)+' |')
    lines+=['','## 控制与可复核性','', f"原始PCM16与float64×32768 MFCC、RAW/identity、重复视频、+5帧延迟控制：engineering PASS={cc['engineering_passed']}。固定k3迁移标记={cc['anchor_transfer_passed']}；不根据lag改变k或支持。",f"独立距离最大误差={v['max_distance_error']:.3g}，指标最大误差={v['max_metric_error']:.3g}，四格闭合={v['max_fourcell_closure']:.3g}，全部主/敏感性统计与EQ−ENVeq复核最大误差={v['max_statistics_error']:.3g}。", '', '生成使用image3、框[138,90,357,387]，评分框[33,18,462,447]至224，25fps FFV1，JPEG SyncNet前端。每臂全部cell共用官方L=min(frame_count,PCM//640)−5；原71支持在任何评分之前固定。Bootstrap按speaker等权20000次，PCG64 seed20260926，CI为逐比较99%，不声称FWER。', '', 'F0/voicing见`f0_descriptive.json`及其数组：首次CPU波形环境没有pyworld，后续在项目.venv对封存PCM按同冻结函数补测；原文件保留，未改变波形、门或评分。完整数值：`summary.json`、`effects.json`、`scores/`、`acoustic_gate.json`、`cal_controls.json`、`independent_validation_all.json`；波形与模型/代码/输入绑定见`protocol.json`、`cpu_seal.json`、`artifact_hashes.json`。', '', '## 解释边界','', '同对N/T在测试时向谱中点匹配，仅部分谱差被拉近，不是训练模型或未见样本确认。EQ保留每条原RMS且未主动时间变形，但存在包络、F0估计与voicing耦合；ENVeq只控制匹配到的包络变化。G是固定评分音频的生成视频响应，E是固定视频的评分音频响应；不能转换成真实嘴型质量或中介比例。CI跨零不代表等效，也不把部分谱拉近后的残余差距归因于单一剩余因素。']
    # Descriptive representation bridge, no regression or causal allocation.
    bridge={}
    score_rows=[read(f) for f in sorted((OUT/'scores').glob('*.json')) if read(f)['split']=='evaluation' and read(f)['eligible']]
    for a in ['N','T']:
        bridge[a]={}
        for h in ['identity','EQ','ENVeq']:
            vals=[r['cells'][a]['raw'][h+'__'+h]['bridge'] for r in score_rows]
            bridge[a][h]={k:float(np.mean([x[k] for x in vals])) for k in ['sigmaA','sigmaV','sigmaM','covAV']}
    (OUT/'representation_descriptive.json').write_text(json.dumps(bridge,indent=2)+'\n')
    lines+=['','表征σA/σV/σM/covAV的描述见`representation_descriptive.json`（固定k3、guard20；样本均值，不作中介拟合）。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    values=[c['baseline_T_minus_N'],c['conditions']['EQ']['processed_T_minus_N'],c['conditions']['ENVeq']['processed_T_minus_N']]
    for i,r in enumerate(values):axes[0].errorbar(i,r['mean'],yerr=[[r['mean']-r['ci99'][0]],[r['ci99'][1]-r['mean']]],fmt='o',capsize=4)
    axes[0].set_xticks(range(3),['Identity','EQ','ENVeq']);axes[0].set_ylabel('TTS - natural Sync-C (99% CI)');axes[0].axhline(0,color='grey',lw=.7);axes[0].set_title('Original matched audio / video gap')
    for i,k in enumerate(['G','E','I','total']):
        r=c['EQ_minus_ENVeq'][k]['T_minus_N'];axes[1].errorbar(i,r['mean'],yerr=[[r['mean']-r['ci99'][0]],[r['ci99'][1]-r['mean']]],fmt='o',capsize=4)
    axes[1].set_xticks(range(4),['G','E','Interaction','Total']);axes[1].set_ylabel('EQ - ENVeq: T-N response difference');axes[1].axhline(0,color='grey',lw=.7);axes[1].set_title('Prespecified envelope control contrast')
    for ext in ['png','pdf']:fig.savefig(OUT/('spectrum_generation_cross.'+ext),dpi=180)
    plt.close(fig)
    final={'status':'concluded','validation_passed':True,'finished_epoch':read(OUT/'final.json')['finished_epoch'] if (OUT/'final.json').exists() else time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'report_sha256':sha(OUT/'report.md'),'n_eval':74,'n_common':71,'speakers_common':15,'gpu_released':all(read(OUT/('gpu_release_'+s+'.json'))['lease_released'] for s in ['calibration','evaluation'])}
    (OUT/'final.json').write_text(json.dumps(final,indent=2)+'\n')
    files={str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['artifact_hashes.json','report.log']}
    (OUT/'artifact_hashes.json').write_text(json.dumps(files,indent=2)+'\n');print('report and hashes',len(files),flush=True)
if __name__=='__main__':main()
