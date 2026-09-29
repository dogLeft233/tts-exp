"""Render the frozen LEVEL residual diagnostic without further scientific choices."""
from pathlib import Path
import json,hashlib,datetime,csv
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_level_residual_representation_20260927'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(r):return f"{r['mean']:+.3f} [{r['ci99'][0]:+.3f}, {r['ci99'][1]:+.3f}]"
def main():
 p=read(OUT/'protocol.json');s=read(OUT/'summary.json');b=read(OUT/'baseline_summary.json');c=read(OUT/'coefficient_summary.json');v=read(OUT/'independent_validation.json');ctrl=read(OUT/'controls.json');mc=read(OUT/'mc_precision.json')
 lines=['# LEVEL后的中点幅度与时间组织残余诊断','',f"状态：concluded；run `{OUT.name}`；全CPU、2线程，无新音视频或GPU。",'', '## 设计与可解释范围','',
 '以父LEVEL实际G/E diagonal的RAW与LEVEL A/V缓存为两背景。旧26cal仅检验k3迁移；共同71eval/15speaker为干预与主敏感性支持。74eval的valid/guard0原生附录保留全部样本；a1_013/053/100的T窗口L24/36/38没有guard20内区，因此旧M/whole操作不可定义，不改参数补救。',
 '固定k=3，J=[0,L−3)，I=[20,L−20)。A′=A[t+3]，M=(A′+V)/2，R=(A′−V)/2。M匹配仅缩放围绕I均值的动态幅度，T→N α=σN/σT，反向α取倒数，R逐query保持。RAW/LEVEL分别计算α；LEVEL×M是同一自适应规则的操作交互，并非固定α的独立因子效应。',
 'whole复用旧static_cloud预封存全部256成对排列，在左halo、I、右halo内分别置换V_t及A_(t+3)。所有背景、raw/unit、M共用索引。每次排列先计算B、minD与C，再平均256次指标。主guard20闭域保留正确配对距离多重集；valid/guard0保持J外数组不变并继承父有效mask/补零规则，仅作非守恒补充。',
 'unit在原向量上先归一化，再独立定义M/R/α，变换后不再归一化。raw/unit分开，不计算跨尺度解释比例。speaker等权20k PCG64 bootstrap，seed20260926，99CI；完整双方向/全端点，没有FWER声明。',
 '此为历史同输入表示诊断，M/R并非生理共享信号/嘴型误差，不是声学可实现干预或中介比例。CI跨0不证明完全解释；所有结论以闭域guard20和B/D/anchor分解共同判断。','', '## 原生基线与电平处理后残余','', '| 几何/支持 | 背景 | ΔC | ΔB | ΔDmin | ΔD_anchor | ΔC_anchor |','|---|---|---|---|---|---|---|']
 for geom in ['raw','unit']:
  for pol in ['guard20','valid','guard0']:
   for bg in ['RAW','LEVEL']:
    q=b[f'common71/{geom}/{pol}/{bg}/gap'];lines.append('| '+f'{geom}/{pol} | {bg} | '+' | '.join(fmt(q[x]) for x in ['C','B','D','D_anchor','C_anchor'])+' |')
 lines+=['','## 双方向匹配系数','', '| 几何 | 背景 | T→N α | N→T α | σN | σT |','|---|---|---|---|---|---|']
 for geom in ['raw','unit']:
  for bg in ['RAW','LEVEL']:
   q=c[f'{geom}/{bg}'];lines.append(f'| {geom} | {bg} | '+' | '.join(fmt(q[x]) for x in ['T_to_N','N_to_T','sigma_N','sigma_T'])+' |')
 lines+=['','## guard20干预后T−N与机制端点','', '| 几何 | 背景 | 方向 | 操作 | ΔC | ΔB | ΔDmin | ΔD_anchor | ΔC_anchor |','|---|---|---|---|---|---|---|---|---|']
 for geom in ['raw','unit']:
  for bg in ['RAW','LEVEL']:
   for direction in ['T_to_N','N_to_T']:
    for cell in ['native','M','whole','M_whole']:
     q=s[f'{geom}/guard20/{bg}/{direction}/{cell}/gap'];lines.append(f'| {geom} | {bg} | {direction} | {cell} | '+' | '.join(fmt(q[x]) for x in ['C','B','D','D_anchor','C_anchor'])+' |')
 lines+=['','## guard20条件效应、交互和LEVEL−RAW操作差','', '| 几何 | 背景或差 | 方向 | 对比 | ΔC | ΔB | ΔDmin | ΔD_anchor | ΔC_anchor |','|---|---|---|---|---|---|---|---|---|']
 for geom in ['raw','unit']:
  for bg in ['RAW','LEVEL','LEVEL_minus_RAW']:
   for direction in ['T_to_N','N_to_T']:
    for effect in ['M_effect','whole_effect','M_after_whole','whole_after_M','interaction']:
     q=s[f'{geom}/guard20/{bg}/{direction}/{effect}/gap'];lines.append(f'| {geom} | {bg} | {direction} | {effect} | '+' | '.join(fmt(q[x]) for x in ['C','B','D','D_anchor','C_anchor'])+' |')
 lines+=['','## LEVEL−RAW在各cell的处理效应','', '| 几何 | 方向 | cell | ΔC | ΔB | ΔDmin | ΔD_anchor | ΔC_anchor |','|---|---|---|---|---|---|---|---|']
 for geom in ['raw','unit']:
  for direction in ['T_to_N','N_to_T']:
   for cell in ['native','M','whole','M_whole']:
    q=s[f'{geom}/guard20/LEVEL_minus_RAW/{direction}/{cell}/gap'];lines.append(f'| {geom} | {direction} | {cell} | '+' | '.join(fmt(q[x]) for x in ['C','B','D','D_anchor','C_anchor'])+' |')
 lines+=['','## 全敏感性与逐臂结果','', '完整N/T/T−N、raw/unit、guard20/valid/guard0、双方向、背景及背景差的六端点（含bestlag）见`all_contrasts.csv`和`summary.json`。全部74的原生valid/guard0见`baseline_summary.json`与`baseline_all74.csv`。未按显著性选择行。下表集中列出补充支持的ΔC/ΔC_anchor，J外未变且边缘query不拥有闭区守恒。','', '| 几何/支持 | 背景 | 方向 | 操作 | ΔC | ΔC_anchor |','|---|---|---|---|---|---|']
 for geom in ['raw','unit']:
  for pol in ['valid','guard0']:
   for bg in ['RAW','LEVEL']:
    for direction in ['T_to_N','N_to_T']:
     for cell in ['native','M','whole','M_whole']:
      q=s[f'{geom}/{pol}/{bg}/{direction}/{cell}/gap'];lines.append(f'| {geom}/{pol} | {bg} | {direction} | {cell} | {fmt(q["C"])} | {fmt(q["C_anchor"])} |')
 lines+=['','## 校验与资源','',f"- 原生2388有效cell的指标与完整曲线重现误差均0；raw/id A/V数组误差0。cal两背景N/T bestlag中位数均3。全部baseline门通过后才评分。",f"- 独立直接距离{v['direct_distances']:,}个；{v['permutation_metric_rows']:,}排列指标行。指标误差{v['metric_max']:.3g}、曲线误差{v['curve_max']:.3g}；统计复核{v['statistics']:,}项误差{v['statistics_max']:.3g}；C/B/D和anchor闭合误差{v['closure_max']:.3g}。",f"- M-only anchor向量float64最大误差{max(x['anchor64'] for x in ctrl):.3g}，float32舍入最大{max(x['anchor32'] for x in ctrl):.3g}；逆变换{max(x['inverse'] for x in ctrl):.3g}；常量M平移全实数距离{max(x['constant_M_shift'] for x in ctrl):.3g}。whole内区anchor多重集逐次精确一致，能量误差{max(x['energy'] for x in ctrl):.3g}。",f"- 256固定，不按结果调整；64/128相对256的各组效应/交互最大绝对差详见mc_precision.json，总最大{max(mc.values()):.6f}。接近0的CI须结合MC有限精度阅读。",'- 不复制源A/V或完整Gram矩阵，保留guard20曲线及所有支持的逐排列指标；250MiB上限和5GiB磁盘保留门持续检查。','', '## 产物与封存','',f"- protocol.json SHA256 `{sha(OUT/'protocol.json')}`。",'- protocol_design.json保留首次设计及时间；protocol.json保留理论端批准后的解释约束、全部输入/旧π/代码绑定。','- baseline_validation.json、calibration_transfer.json、score_lock.json、permutation_controls.json、coefficients.json、controls.json、independent_validation.json及artifact_hashes.json。']
 conclusions=['## 结果与结论','',
 '1. **电平处理后仍有主端点残余，raw读数更明确地支持错位背景差。** LEVEL raw guard20 T−N C=+0.266 [0.075,0.443]，B=+0.303 [0.102,0.501]；Dmin=+0.037 [−0.086,0.162]、D_anchor=+0.076 [−0.073,0.226]，没有确认T的正确配对距离更小。C_anchor仍+0.227 [0.014,0.423]。这支持残余C与背景分离有关，不能从统计分解声称唯一根因或中介比例。',
 '2. **主闭域下，LEVEL后的自适应M匹配、whole及二者交互均未确认进一步缩小差距。** raw T→N M对gap作用+0.062 [−0.190,0.350]，反向−0.022 [−0.223,0.199]；whole −0.147 [−0.317,0.014]，对应C_anchor −0.109 [−0.275,0.033]。这些区间仍容许实质下降，不能据此断言时间组织作用不存在。RAW whole的标准C效应−0.171 [−0.333,−0.020]，RAW→LEVEL whole效应差+0.024 [−0.001,0.051]；相应anchor差+0.004 [−0.015,0.022]，未确认时间响应随LEVEL消失。',
 '3. **联合残余跨0不证明全解释，且unit残余仍正。** LEVEL raw M+whole双方向残余+0.185 [−0.018,0.440]、+0.114 [−0.033,0.280]；unit对应+0.037 [0.012,0.073]、+0.027 [0.011,0.047]。几何内的负结果和残余并列报告，不能跨raw/unit计算解释份额。',
 '4. **LEVEL改变了自适应M匹配规则的操作结果。** raw M的gap效应LEVEL−RAW为T→N +0.215 [0.084,0.355]、反向+0.205 [0.079,0.339]；但T→N平均α从RAW 0.985 [0.955,1.020]变为LEVEL 1.007 [0.981,1.036]。处理目标与方向一同变化，不能说固定M扰动的网络响应减弱。主raw三重交互T→N标准C +0.008 [0.002,0.014]而anchor +0.003 [−0.004,0.010]；它包含minD变化，非确认纯背景交互。',
 '5. **边界/支持会改变结论，补充不替代主闭域。** 同71 raw-valid LEVEL原生gap+0.105 [−0.045,0.238]，whole作用−0.122 [−0.252,−0.007]；raw-guard0原生gap−0.115 [−0.272,0.034]。unit-valid LEVEL联合残余仍正，unit-guard0则跨0。valid/guard0未保持整个评分域的anchor多重集，不把其显著性升级为闭域根因证据。','']
 lines[4:4]=conclusions
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 with (OUT/'all_contrasts.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['contrast','metric','mean','ci99_lo','ci99_hi','pairs','speakers'])
  for k,q in s.items():
   for metric,r in q.items():w.writerow([k,metric,r['mean'],*r['ci99'],r['n'],r['speakers']])
 with (OUT/'baseline_all74.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['contrast','metric','mean','ci99_lo','ci99_hi'])
  for k,q in b.items():
   if k.startswith('all74'):
    for metric,r in q.items():w.writerow([k,metric,r['mean'],*r['ci99']])
 (OUT/'report_code_seal.json').write_text(json.dumps({str(Path(__file__)):sha(Path(__file__))},indent=2)+'\n')
 print('REPORT',sha(OUT/'report.md'))
if __name__=='__main__':main()
