"""Final report and standalone plots for the frozen waveform G x E study."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_acoustic_generation_cross_20260926'

def read(p):return json.loads(Path(p).read_text())
def fmt(x):return f"{x['mean']:+.3f} [{x['ci99'][0]:+.3f}, {x['ci99'][1]:+.3f}]"

def run():
    summary=read(OUT/'summary.json');p=read(OUT/'protocol.json');support=read(OUT/'support.json');scores=[read(f) for f in sorted((OUT/'scores').glob('*.json'))];effects=read(OUT/'effects.json');gate=read(OUT/'acoustic_gate.json');control=read(OUT/'cal_controls.json');ind=read(OUT/'independent_validation_all.json')
    fields=list(effects[0]);f=(OUT/'effects.csv').open('w');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(effects);f.close()
    main=summary['results']['raw/guard20']['C'];n=len(gate['new_main_ids'])
    report=f'''# 固定真实波形的生成器×评分音频四格

状态：{summary['status']}。历史static cloud image3条件，原26cal工程检查、原74eval中共同guard20支持{n}对；新identity视频为主baseline。本实验在source条件新结果揭示前冻结；不是未见确认。未重新TTS，未做ASR/人眼筛选，也未按结果选择剂量、样本或端点。

identity基线原配 T−N Sync-C：**{fmt(main['baseline_T_minus_N'])}**。下文所有 Sync-C 均保留3位小数，区间为逐比较99% speaker bootstrap CI，未宣称FWER控制。

## 四格与主结果

固定每个来源臂，q00=E(G(identity),identity)、q10=E(G(h),identity)、q01=E(G(identity),h)、q11=E(G(h),h)。G=q10−q00、E=q01−q00、I=q11−q10−q01+q00、total=q11−q00。G-only在同一固定评分音频下改变实际生成视频；E-only固定视频改变评分音频。它们都是评价器响应，不是人类唇形质量真值。

|处理|分量|N效应|T效应|T−N效应差|
|---|---|---|---|---|
'''
    for c in ['Rlo','Rhi','ENVlo','ENVhi']:
        for effect in ['G','E','I','total']:
            d=main['conditions'][c][effect];report+=f"|{c}|{effect}|{fmt(d['N'])}|{fmt(d['T'])}|{fmt(d['T_minus_N'])}|\n"
    report+='''
R处理与同能量包络控制的差异，命名为**该时频处理的非包络部分**。两处理仍有长期谱、F0与STFT一致性差异，不能称纯时间对比度因果效应。

|剂量R−ENV|分量|N效应差|T效应差|两臂效应差的差|
|---|---|---|---|---|
'''
    for dose in ['lo','hi']:
        for effect in ['G','E','I','total']:
            d=main['R_minus_ENV'][dose][effect];report+=f"|{dose}|{effect}|{fmt(d['N'])}|{fmt(d['T'])}|{fmt(d['T_minus_N'])}|\n"
    report+='''
## 支持、时轴和前端

100个原历史pair全部保留，26cal与74eval沿用原拆分。主统计仅按生成帧数和PCM长度的共同joint支持决定；每臂L=min(frame_count,PCM_samples//640)−5，所有六条件共用同一L。N与T各自原生时钟，不跨臂换音轨、无DTW/重采样/时长匹配。guard20统计要求双臂L>40；validpair和guard0同时报告全部74eval。

'''
    report+=f"仅时长排除主guard20的ID：{', '.join(gate['excluded_short'])}；与既有71支持一致：{gate['same_old_support']}。\n\n"
    report+='''
固定image3，生成框[138,90,357,387]，评分框[33,18,462,447]→224；25fps、FFV1、官方Wav2Lip mel分块，batch32。SyncNet视觉保留FFmpeg JPEG→OpenCV前端。waveform为FLOAT32，MFCC输入float64×32768，保留历史PCM整数振幅单位，未再量化；MFCC默认参数与forward_aud不变。k固定旧静态cal的3，仅作anchor；标准C仍在31 lag ±15上计算B−D。bestlag=argmin−15，offset=−bestlag。

新identity视频均实际渲染。RAW只有mel逐元素相同且非首cal时才alias本轮新identity视频；首cal两臂额外独立RAW渲染检验桥接。历史视觉/音频特征只作bridge，未混入主baseline。

## 控制和独立复算

'''
    report+=f"工程控制通过：{control['engineering_passed']}；anchor迁移检查通过：{control['anchor_transfer_passed']}。所有条件cal N/T diagonal median lag：`{json.dumps(control['condition_arm_medianlags'],ensure_ascii=False)}`。固定k不因检查结果重估，分母不调整。\n\n"
    report+='|臂|重复像素一致|重复feature最大差|原PCM16/float MFCC差|原PCM16/float feature差|对应C差|RAW/identity C差|+5延迟最佳lag变化|\n|---|---|---:|---:|---:|---:|---:|---|\n'
    for a in ['N','T']:
        c=control['parity'][a];d=control['delay'][a]
        report+=f"|{a}|{c['repeat_pixel_equal']}|{c['repeat_feature_max']:.3g}|{c['old_float_mfcc_max']:.3g}|{c['old_float_audio_max']:.3g}|{c['old_float_score_C_difference']:.3g}|{c['raw_identity_score_C_difference']:.3g}|{d['base_bestlag']}→{d['delay_bestlag']}（预期{d['expected']}）|\n"
    report+=f"\n延迟测试为+3200样点/5帧、零头同长度；共同query25:L−25和lag重叠区保持同一支持。独立NumPy复算距离{ind['distance_entries']:,}项，最大差{ind['max_distance_error']:.3g}；统计最大差{ind['max_statistics_error']:.3g}，四格闭合最大误差{summary['maximum_algebra_closure_error']:.3g}。4项契约测试通过。\n"
    report+='\n## 固定声学操作和原始差异\n\n512/128 periodic Hann，16k mono，−80dB相对floor；双向中心化log谱残差R，alpha=.8/1.2，复杂谱正实增益±6dB。identity为STFT往返。ENV控制匹配R波形的sample-level Hann512包络，固定20步半步幅度更新、±12dB中间gain界；每步恢复原臂RMS。两臂全部12波形统一headroom，不逐条peaknorm。\n\n'
    report+='|拆分|R有序臂比例|降低中位|增强中位|ENVlo匹配dB RMSE|ENVhi匹配dB RMSE|声学门通过|\n|---|---:|---:|---:|---:|---:|---|\n'
    for name,g in gate['splits'].items():report+=f"|{name}|{100*g['ordered_fraction']:.1f}%|{100*g['median_decrease']:.2f}%|{100*g['median_increase']:.2f}%|{g['ENV']['ENVlo']['median_db_rmse']:.4f}|{g['ENV']['ENVhi']['median_db_rmse']:.4f}|{g['passed']}|\n"
    acoustic=read(OUT/'acoustic_metrics.json');lookup={(r['id'],r['arm'],r['condition']):r for r in acoustic};desc=[]
    for split in ['calibration','evaluation']:
        ids=[r['id'] for r in support if r['split']==split]
        for field in ['residual_rms_db','global_rms','envelope_log_sd_db','spectrum_tilt_db_octave','f0_median_hz','voiced_fraction','floor_fraction']:
            values=[lookup[sid,'T','raw'][field]-lookup[sid,'N','raw'][field] for sid in ids if lookup[sid,'T','raw'][field] is not None and lookup[sid,'N','raw'][field] is not None]
            desc.append({'split':split,'metric':field,'n':len(values),'paired_mean':float(np.mean(values)),'paired_median':float(np.median(values))})
    (OUT/'raw_acoustic_differences.json').write_text(json.dumps(desc,indent=2)+'\n')
    report+='\n|RAW T−N描述量|26cal配对均值/中位|74eval配对均值/中位|\n|---|---|---|\n'
    for field in ['residual_rms_db','global_rms','envelope_log_sd_db','spectrum_tilt_db_octave','f0_median_hz','voiced_fraction','floor_fraction']:
        a=next(r for r in desc if r['metric']==field and r['split']=='calibration');b=next(r for r in desc if r['metric']==field and r['split']=='evaluation');report+=f"|{field}|{a['paired_mean']:+.6f} / {a['paired_median']:+.6f}|{b['paired_mean']:+.6f} / {b['paired_median']:+.6f}|\n"
    report+='\n尤其旧26cal R RMS的原始T−N仅均值+0.110dB、中位+0.022dB；即使操纵R引起评分显著改变，也不证明它解释原生TTS优势。核心读数是预设的T−N G/E/I/total效应差及R−ENV对照。声学表不作特征筛选、回归或中介百分比。\n'
    report+='\n## 表征桥接及敏感性\n\nσA、σV、σM和covAV在各四格相同guard20 query、固定k3上计算，M=(A[i+3]+V[i])/2，各轨迹减自身时间均值。逐cell在scores/*.json:bridge；另存representation_bridge.json。该描述不把共同轨迹称生理真值。unit几何先对每行原特征L2归一化再评分，没有干预后再归一化波形。\n'
    bridges=[]
    for r in scores:
        if r['split']!='evaluation' or not r['eligible']:continue
        for arm in ['N','T']:
            for key,value in r['cells'][arm]['raw'].items():
                if value.get('bridge'):bridges.append({'id':r['id'],'speaker':r['speaker'],'arm':arm,'cell':key,**value['bridge']})
    (OUT/'representation_bridge.json').write_text(json.dumps(bridges,indent=2)+'\n')
    bridge_lookup={(b['id'],b['arm'],b['cell']):b for b in bridges}
    bridge_changes=[]
    for arm in ['N','T']:
        ids=sorted({b['id'] for b in bridges if b['arm']==arm})
        for condition in ['identity','Rlo','Rhi','ENVlo','ENVhi']:
            for metric in ['sigmaA','sigmaV','sigmaM','covAV']:
                delta=[bridge_lookup[sid,arm,condition+'__'+condition][metric]-bridge_lookup[sid,arm,'identity__identity'][metric] for sid in ids]
                raw=[bridge_lookup[sid,arm,condition+'__'+condition][metric] for sid in ids]
                bridge_changes.append({'arm':arm,'condition':condition,'metric':metric,'mean_absolute':float(np.mean(raw)),'mean_change_vs_identity':float(np.mean(delta)),'median_change_vs_identity':float(np.median(delta))})
    (OUT/'representation_bridge_changes.json').write_text(json.dumps(bridge_changes,indent=2)+'\n')
    report+='\n表征原生及处理对角变化：`representation_bridge_changes.json`逐臂给出各条件σA/σV/σM/covAV均值及相对identity的均值/中位变化；这些为描述量，不作中介比例。\n'
    report+='\n|几何/支持|identity原配T−N|Rlo总效应T−N|Rhi总效应T−N|\n|---|---|---|---|\n'
    for view,block in summary['results'].items():
        b=block['C'];report+=f"|{view}|{fmt(b['baseline_T_minus_N'])}|{fmt(b['conditions']['Rlo']['total']['T_minus_N'])}|{fmt(b['conditions']['Rhi']['total']['T_minus_N'])}|\n"
    report+='\n完整四格C/B/D/C_anchor/D_anchor/bestlag、所有G/E/I/total与R−ENV的99CI保存在summary.json/effects.json/csv；score文件保留offset及完整31lag曲线。原始audio_encoder z只作捕获存档，未改变decoder/激活或用于选参数。\n\n## 复现产物\n\nprotocol.json/protocol.sha256、cpu_seal、score_lock与feature_seal分别封存协议/CPU支持/评分输入；audio/frontend/features/videos/scores保存波形、前端、特征和视频；cal_controls与独立验证包含数值控制。artifact_hashes.json给完整hash链。主脚本各stage独立，所有GPU操作用共享lease、绑定系统daemon例外并保持5GiB余量；无历史删除。\n'
    (OUT/'report.md').write_text(report)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,family,title in [(axes[0],main['conditions'],'T-minus-N response by operation'),(axes[1],main['R_minus_ENV'],'R-minus-ENV: T-minus-N response')]:
        labels=list(family);x=np.arange(len(labels))
        for j,effect in enumerate(['G','E','I','total']):
            d=[family[c][effect]['T_minus_N'] for c in labels];mean=np.array([t['mean'] for t in d]);lo=np.array([t['ci99'][0] for t in d]);hi=np.array([t['ci99'][1] for t in d]);ax.errorbar(x+(j-1.5)*.17,mean,yerr=np.stack([mean-lo,hi-mean]),fmt='o',capsize=3,label=effect)
        ax.axhline(0,color='gray',lw=1);ax.set_xticks(x,labels);ax.set_ylabel('Delta Sync-C, 99% CI');ax.set_title(title);ax.legend()
    fig.savefig(OUT/'generation_evaluation_cross.pdf');fig.savefig(OUT/'generation_evaluation_cross.png',dpi=160);plt.close(fig)
    hashes={str(f.relative_to(OUT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='artifact_hashes.json' and f.suffix!='.log'}
    (OUT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('report complete',len(hashes))

if __name__=='__main__':run()
