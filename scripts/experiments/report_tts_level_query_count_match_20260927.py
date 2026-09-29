"""Reporting fixed query-count matching and operator-order diagnostics."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_level_query_count_match_20260927';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def fmt(r):return f"{r['mean']:+.3f} [{r['ci99'][0]:+.3f},{r['ci99'][1]:+.3f}]"
def main():
 p=read(OUT/'protocol.json');s=read(OUT/'summary.json');q=read(OUT/'support_summary.json');v=read(OUT/'independent_validation.json');controls=read(OUT/'controls.json');mc=read(OUT/'mc_precision.json');lines=['# 评分查询数量匹配的电平残余诊断','', '状态：concluded。2 CPU线程，固定1024子集，无GPU/音视频生成/模型推理。','', '## 冻结设计与范围','',
 '共同71eval/15speaker，RAW/LEVEL实际native diagonal、raw/unit分别，k3、原±15lag、I=[20,L−20)。每对q=min(|I_N|,|I_T|)，每臂在自身原I均匀无放回取q；固定1024个子集，索引跨RAW/LEVEL/raw/unit共用。子集按原绝对query顺序保存；源A/V/lag与时间标签不变，未嵌套独立null。',
 '每clip/arm/b使用SHA256(20260927|query_count|id|arm|b)前16bytes派生PCG64。全部索引、输入、科学代码先封存，全query基线门通过后才进行子集评分。q=|I|仍保留1024次全集；q短不筛除。',
 '每子集先平均选定的原query距离行成31lag曲线，再算C/B/D/D_anchor/C_anchor/bestlag/search_uplift；主结果为mean(metric(subset_curve))。预设描述同时报告metric(mean(subset_curve))及两种算子顺序之差。B的median与D的min均非线性，不能将两种C差全归给min。',
 'speaker等权20k PCG64 bootstrap，seed20260926，99CI，无FWER；不跨raw/unit算解释比例。1024固定，256/512对1024仅MC精度描述。此操作只改变评分估计查询样本数，不能称真实语速/内容/时长匹配，也不把匹配后分数当更好真值。','', '## 查询支持','',f"q范围{q['q_min']}–{q['q_max']}，speaker等权q {fmt(q['q'])}。q≤5的clip：{', '.join(q['q_le5'])}；全部保留。",'', '| 臂 | 原I数量 | 减少数量 | 减少比例 | 原query全集未变clip数 |','|---|---|---|---|---|']
 for a in ['N','T']:d=q['arms'][a];lines.append(f"| {a} | {fmt(d['I'])} | {fmt(d['removed'])} | {fmt(d['removed_fraction'])} | {d['unchanged_clips']} |")
 lines+=['','每clip原N/T query数、q、减少数量/比例在protocol.json support中；1024次完整索引在indices/中。','', '## 全query与matched-query：T−N及LEVEL差','', '| 几何 | 背景或差 | 条件/作用 | ΔC | ΔB | ΔDmin | ΔD_anchor | ΔC_anchor | Δsearch uplift |','|---|---|---|---|---|---|---|---|---|']
 fields=['C','B','D','D_anchor','C_anchor','search_uplift']
 for g in ['raw','unit']:
  for bg in ['RAW','LEVEL','LEVEL_minus_RAW']:
   for term in ['full','matched','matched_minus_full']:
    d=s[f'{g}/{bg}/gap/{term}'];lines.append(f'| {g} | {bg} | {term} | '+' | '.join(fmt(d[x]) for x in fields)+' |')
 lines+=['','## 逐臂完整评分与处理效应','', '| 几何 | 背景或差 | 臂 | 条件/作用 | C | C_anchor | search uplift |','|---|---|---|---|---|---|---|']
 for g in ['raw','unit']:
  for bg in ['RAW','LEVEL','LEVEL_minus_RAW']:
   for a in ['N','T']:
    for term in ['full','matched','matched_minus_full']:
     d=s[f'{g}/{bg}/{a}/{term}'];lines.append(f'| {g} | {bg} | {a} | {term} | '+' | '.join(fmt(d[x]) for x in ['C','C_anchor','search_uplift'])+' |')
 lines+=['','## 两种算子顺序：必须区分','', '| 几何 | 背景或差 | 臂/差 | 条件/作用 | C | B | Dmin | C_anchor | search uplift |','|---|---|---|---|---|---|---|---|---|']
 for g in ['raw','unit']:
  for bg in ['RAW','LEVEL','LEVEL_minus_RAW']:
   for a in ['N','T','gap']:
    for term in ['matched','metric_of_mean_curve','mean_metric_minus_metric_mean','mean_curve_minus_full']:
     d=s[f'{g}/{bg}/{a}/{term}'];lines.append(f'| {g} | {bg} | {a} | {term} | '+' | '.join(fmt(d[x]) for x in ['C','B','D','C_anchor','search_uplift'])+' |')
 lines+=['','## 校验与精度','',f"- 全query568 cell基线最大误差{read(OUT/'baseline_validation.json')['max']:.3g}，全部通过后才评分。",f"- 独立原数组距离{v['direct_distances']:,}；float64直接距离对生产torch float32最大误差{v['distance_float64_vs_torch_float32_max']:.3g}，由原数组重建的子集距离端点最大误差{v['independent_source_metric_max']:.3g}。近并列bestlag浮点差异{v['near_tie_bestlag_rounding_cases']}次，若有均要求竞争lag距离差≤两倍数值误差；不据此选择lag或删除子集。",f"- {v['subset_indices_regenerated']:,}个子集重生成逐元素相同，q、互异性及I成员校验通过。经独立直接距离核验后的封存矩阵再重算全部子集：曲线误差{v['curve_reconstruction_max']:.3g}，指标误差{v['metric_max']:.3g}。",f"- {v['statistics']:,}统计项独立复算误差{v['statistics_max']:.3g}；C=C_anchor+search uplift闭合误差{v['closure_max']:.3g}。",f"- q=I的全集子集曲线最大误差{max(r['full_query_identity_max'] for r in controls if r['full_query_identity_max'] is not None):.3g}；MC平均曲线与原full曲线最大差{max(r['mean_curve_vs_full_MC_max'] for r in controls):.6f}（非筛选门）。",f"- 256/512相对1024的组均值六距离端点最大变化{max(abs(vv) for x in mc.values() for k,vv in x.items() if k!='best_lag'):.6f}；全部逐臂/背景差详见mc_precision.json。未改变固定1024次数。",'- 全108对比×7端点的完整99CI见all_contrasts.csv/summary.json。保存完整索引、每次指标、full/mean曲线和距离矩阵，可重建全部子集31曲线，未保存庞大的中间张量。','',f"protocol SHA256 `{sha(OUT/'protocol.json')}`。"]
 c=OUT/'conclusions.md'
 if c.exists():lines[4:4]=[c.read_text(),'']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 with (OUT/'all_contrasts.csv').open('w') as ff:
  w=csv.writer(ff);w.writerow(['contrast','metric','mean','ci99_lo','ci99_hi','pairs','speakers'])
  for key,val in s.items():
   for f,r in val.items():w.writerow([key,f,r['mean'],*r['ci99'],r['n'],r['speakers']])
 (OUT/'report_code_seal.json').write_text(json.dumps({str(Path(__file__)):sha(Path(__file__))},indent=2)+'\n');print('REPORT',sha(OUT/'report.md'))
if __name__=='__main__':main()
