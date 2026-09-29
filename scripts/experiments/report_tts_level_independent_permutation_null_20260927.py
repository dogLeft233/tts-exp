"""Fixed block-null result tables; no new scoring."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_level_independent_permutation_null_20260927';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def f(r):return f"{r['mean']:+.3f} [{r['ci99'][0]:+.3f},{r['ci99'][1]:+.3f}]"
def main():
 s=read(OUT/'summary.json');p=read(OUT/'protocol.json');v=read(OUT/'independent_validation.json');a=read(OUT/'accidental_summary.json');mc=read(OUT/'mc_precision.json');lines=['# 电平残余的分块独立置换与最优时滞搜索诊断','', '状态：concluded。全CPU2线程，无新音视频/模型推理/GPU。','', '## 协议与解释边界','',
 '固定共同71eval/15speaker，RAW和LEVEL实际diagonal背景，raw/unit分别；k3、guard20闭域、原±15lag。J=[0,L−3)，I=[20,L−20)，J分左halo20帧、内I、右halo17帧。π复用旧seed20260926全部256 whole排列。ρ按20260927/clip/arm/排列编号SHA256派生PCG64，在三个域内独立均匀排，RAW/LEVEL/raw/unit共用索引。',
 '仅native、paired whole(Vπ,Aπ)、independent whole(Vπ,Aρ)。每次排列先算C、B、minD、固定anchor与search uplift，再平均256；没有先平均曲线求min。所有ρ/输入/代码先封存，完整父曲线复现通过后才新评分。unit只在原向量上按父规范归一化，无M/额外归一化。',
 '三项：native−paired（时间排列响应）、paired−independent（随机时间背景下保留正确对应的作用）、independent（该分块null下剩余分数）；逐臂、T−N和背景差精确相加为native。此为表示空间操作恒等式，不是生理中介比例。',
 'independent仅保持各模态域内边际集合，故意改变R并破坏时间自相关，但仍保留粗域标签。短I有较高偶然对应率，不删短样本、不拒绝固定点、不强制derangement。其剩余可能包含粗域组成和有限样本搜索，不能全叫搜索噪声，也不是所有异步视频的通用零分。',
 'search_uplift=C−C_anchor=D_anchor−Dmin；逐排列计算再平均。报告固定anchor和搜索上浮，避免仅凭标准C判断对应质量。speaker等权20k PCG64 bootstrap，seed20260926、99CI；无FWER，无跨几何份额。256固定，64/128稳定性仅描述。','', '## 原生、成对与独立null：T−N','', '| 几何 | 背景 | 条件 | ΔC | ΔB | ΔDmin | ΔD_anchor | ΔC_anchor | Δsearch uplift |','|---|---|---|---|---|---|---|---|---|']
 fields=['C','B','D','D_anchor','C_anchor','search_uplift']
 for geom in ['raw','unit']:
  for bg in ['RAW','LEVEL','LEVEL_minus_RAW']:
   for term in ['native','paired_whole','independent_whole']:
    q=s[f'{geom}/{bg}/gap/{term}'];lines.append(f'| {geom} | {bg} | {term} | '+' | '.join(f(q[x]) for x in fields)+' |')
 lines+=['','## 三项闭合：各臂与T−N','', '| 几何 | 背景 | 臂/差 | 分量 | C | C_anchor | search uplift |','|---|---|---|---|---|---|---|']
 for geom in ['raw','unit']:
  for bg in ['RAW','LEVEL','LEVEL_minus_RAW']:
   for arm in ['N','T','gap']:
    for term in ['native_minus_paired','paired_minus_independent','independent_residual']:
     q=s[f'{geom}/{bg}/{arm}/{term}'];lines.append(f'| {geom} | {bg} | {arm} | {term} | '+' | '.join(f(q[x]) for x in ['C','C_anchor','search_uplift'])+' |')
 lines+=['','## 每臂各条件','', '| 几何 | 背景 | 臂 | 条件 | C | C_anchor | search uplift |','|---|---|---|---|---|---|---|']
 for geom in ['raw','unit']:
  for bg in ['RAW','LEVEL','LEVEL_minus_RAW']:
   for arm in ['N','T']:
    for term in ['native','paired_whole','independent_whole']:
     q=s[f'{geom}/{bg}/{arm}/{term}'];lines.append(f'| {geom} | {bg} | {arm} | {term} | '+' | '.join(f(q[x]) for x in ['C','C_anchor','search_uplift'])+' |')
 lines+=['','## 偶然对应与粗域保留','', '| 臂 | I长度范围 | I内ρ==π观察率（speaker等权） | 条件期望1/I（speaker等权） |','|---|---|---|---|']
 for arm in ['N','T']:q=a[arm];lines.append(f"| {arm} | {q['I_min']}–{q['I_max']} | {f(q['observed'])} | {f(q['expected'])} |")
 lines+=['','每clip三域长度、全部256次观察比例及理论期望保存在`accidental_correspondence.json`，两背景/几何同一索引。`accidental_summary.json`完整列出全部71 clip，无阈值筛选。','', '## 校验与完整数据','',f"- 父native/paired 568 cell曲线最大复现误差{read(OUT/'baseline_validation.json')['parent_curves_max']:.3g}；原始source-native曲线误差{read(OUT/'baseline_validation.json')['source_native_max']:.3g}（原父完整baseline复现误差0）。",f"- 独立直接距离{v['direct_distances']:,}，曲线误差{v['curve_max']:.3g}、指标{v['metric_max']:.3g}、父曲线{v['parent_curves_max']:.3g}；ρ重新生成逐元素一致，域多重集检查{v['marginal_multiset_checks']:,}次通过，偶然对应比例误差0。",f"- {v['statistics']:,}统计项独立复算误差{v['statistics_max']:.3g}；三项逐clip/臂/背景差闭合误差{v['closure_max']:.3g}。",f"- 固定256；64/128相对256组均值/对比的六距离端点最大绝对变化{max(mc.values()):.6f}，详见mc_precision.json，接近0CI保留MC精度边界。",'- 完整108条件/对比×7端点（含bestlag）、N/T/T−N和LEVEL−RAW见all_contrasts.csv及summary.json。所有跨零结果保留。','- 新增≤100MiB并持续保留5GiB磁盘，不复制父特征或native/paired曲线。','',f"protocol SHA256 `{sha(OUT/'protocol.json')}`。"]
 derived=read(OUT/'native_minus_independent.json')
 lines+=['','## 理论端追加的既有分量合计','', '在冻结结果出现后，理论端要求报告native−independent，即前两项之和。这里只汇总既有逐clip分数，没有新距离或新排列；独立cluster复核误差'+str(derived['independent_cluster_error'])+'。','', '| 几何 | 背景或差 | T−N C | T−N C_anchor | T−N search uplift |','|---|---|---|---|---|']
 for geom in ['raw','unit']:
  for bg in ['RAW','LEVEL','LEVEL_minus_RAW']:
   q=derived['results'][f'{geom}/{bg}/gap/native_minus_independent'];lines.append(f'| {geom} | {bg} | '+' | '.join(f(q[x]) for x in ['C','C_anchor','search_uplift'])+' |')
 lines+=['','全部逐臂/B/D/anchor及CI见native_minus_independent.json。完整64/128对256逐对比差见mc_contrast_differences.json；主结论固定256不变。']
 conclusion=OUT/'conclusions.md'
 if conclusion.exists():lines[4:4]=[conclusion.read_text(),'']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 with (OUT/'all_contrasts.csv').open('w') as ff:
  w=csv.writer(ff);w.writerow(['contrast','metric','mean','ci99_lo','ci99_hi','pairs','speakers'])
  for key,q in s.items():
   for met,r in q.items():w.writerow([key,met,r['mean'],*r['ci99'],r['n'],r['speakers']])
 (OUT/'report_code_seal.json').write_text(json.dumps({str(Path(__file__)):sha(Path(__file__))},indent=2)+'\n');print('REPORT',sha(OUT/'report.md'))
if __name__=='__main__':main()
