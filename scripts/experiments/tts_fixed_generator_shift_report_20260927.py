"""Read-only rendering of all prespecified shift-factorial results."""
import gzip,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/tts_fixed_generator_shift_cross_20260927'
def read(p):
 p=Path(p);return json.loads(gzip.open(p,'rt').read() if p.suffix=='.gz' else p.read_text())
def cell(x):return f"{x['mean']:+.3f} [{x['ci99'][0]:+.3f}, {x['ci99'][1]:+.3f}]"
def main():
 assert not (RUN/'report.md').exists();assert read(RUN/'independent_validation.json')['status']=='PASS'
 s=read(RUN/'summary.json.gz');e=read(RUN/'event_summary.json.gz');a=read(RUN/'analysis.json');p=read(RUN/'protocol.json');dose=read(RUN/'dose_descriptions.json.gz')
 lines=['# FIXED生成器：原型残差收缩 × 带ReLU投影来源位移','',
 '2026-09-27。状态 concluded。独立后续协议在父592混合处理效果揭示前冻结并先验PASS；原模板、λ=.5、支持与端点未依父效果调整。','',
 '## 主结论与操作','',
 '主端点：共同71 clips/15 speakers、raw guard20官方C，自然phone `S=q01−q00` 的99CI下界>0才称“固定自然音轨下，生成路径的Sync-C收益”。固定k3 anchor同时报告；只有标准C改善不能称原时钟同步改善。',
 f"主自然phone S：{cell(s['raw/guard20/common71/C/phone/source_S/N'])}；冻结判据={a['natural_phone_source_shift_gain']}。",'',
 f"预定global次要条件的自然S：C {cell(s['raw/guard20/common71/C/global/source_S/N'])}；anchor {cell(s['raw/guard20/common71/C_anchor/global/source_S/N'])}。这不能替代失败的phone主端点；比较未作FWER校正。",
 '严格正/负判据使用未四舍五入的JSON；表中CI界限显示+0.000不代表精确零。', '',
 'global的raw valid共同71视图C及anchor区间跨0，全部74 valid的C也跨0；旧32事件自然margin未确认改善。完整敏感性如下，不能只保留主guard20的小幅阳性。', '',
 '原00=baseline；10=自己的source模板半权混合；01=新P(z+.5(μother−μown))；11=另一source模板半权混合。P为ReLU。N受体的other是T，T受体的other是N，故不同于统一给两臂同一T模板。旧00/10/11按原字节引用，仅新增phone/global×N/T×74=296视频。评价A始终原FIXED。','',
 'B=q10−q00为相对校准原型的残差收缩，含说话人/句子偏移和phone内部动态；S=q01−q00为带投影来源位移；I=q11−q10−q01+q00；total=q11−q00。B+S+I=total只是操作闭合，不是生理中介比例。I的CI跨0不能称独立。这里component_B不等于评分背景距离Bmedian。','',
 '所有表均speaker等权均值[99%CI]，20,000 paired bootstrap/seed20260926，每比较无FWER。raw/unit各沿旧定义，不能跨量纲算比例。','']
 def table(title,data,prefix,rows,metrics):
  lines.extend(['### '+title,'','| 对比 | '+' | '.join(metrics)+' |','|:--|' + ':--|'*len(metrics)])
  for label,key in rows:lines.append('| '+label+' | '+' | '.join(cell(data[prefix+m+'/'+key]) for m in metrics)+' |')
  lines.append('')
 lines+=['## 1. 主raw guard20共同71','']
 for kind in ['phone','global']:
  rows=[(kind+'/'+q+'/'+arm,kind+'/cell/'+q+'/'+arm) for q in ['q00','q10','q01','q11'] for arm in 'NT']
  table(kind+'四cell',s,'raw/guard20/common71/',rows,['C','B','D','C_anchor','D_anchor','search_uplift'])
  rows=[(kind+'/'+comp+'/'+arm,kind+'/'+comp+'/'+arm) for comp in ['component_B','source_S','interaction_I','total','S_after_B','B_after_S'] for arm in ['N','T','TminusN']]
  table(kind+'响应与交互',s,'raw/guard20/common71/',rows,['C','B','D','C_anchor','D_anchor','search_uplift'])
  table(kind+'各cell原配T−N差',s,'raw/guard20/common71/',[(q,kind+'/gap/'+q) for q in ['q00','q10','q01','q11']],['C','B','D','C_anchor','search_uplift'])
 table('phone−global同分量',s,'raw/guard20/common71/',[(c+'/'+arm,'phone_minus_global/'+c+'/'+arm) for c in ['component_B','source_S','interaction_I','total'] for arm in ['N','T','TminusN']],['C','C_anchor','search_uplift'])
 lines+=['## 2. 全10视图敏感性','', '全部端点/比较见summary.json.gz；下表预定并列自然S的C/anchor，不据视图选择结论。','',
 '| 几何/支持 | phone S C | phone S anchor | global S C | global S anchor |','|:--|:--|:--|:--|:--|']
 for geom in ['raw','unit']:
  for policy,support in [('guard20','common71'),('valid','common71'),('valid','all74'),('guard0','common71'),('guard0','all74')]:
   pre=f'{geom}/{policy}/{support}/';values=[cell(s[pre+m+'/'+kind+'/source_S/N']) for kind in ['phone','global'] for m in ['C','C_anchor']]
   lines.append('| '+pre.rstrip('/')+' | '+' | '.join(values)+' |')
 lines+=['','## 3. 旧32/13/1411事件衔接','', '原query/donor/k3/image3完整保持，不套guard20，不与71官方C直接分解或换算。','']
 for geom in ['raw','unit']:
  rows=[(kind+'/'+c+'/'+arm,kind+'/'+c+'/'+arm) for kind in ['phone','global'] for c in ['component_B','source_S','interaction_I','total'] for arm in ['N','T','TminusN']]
  table(geom+'事件分量',e,geom+'/',rows,p['event_metrics'])
 lines+=['## 4. 输入剂量、投影与舍入（描述）','',
 '同λ不保证N/T特征剂量相同；标量RMS接近也不证明操作剂量等价。以下仅描述全部共同71，未据此筛样/调参/拟合。B描述是父效果后由root要求的输入量补充；S/projection在新评分前预定。','',
 '| 受体/模板类 | B speech坐标RMS | S投影后RMS | S投影前halfδ RMS | 投影改变量RMS | pre<0比例clip中位 |','|:--|:--|:--|:--|:--|:--|']
 for arm in 'NT':
  for kind in ['phone','global']:
   d=dose['summaries']['common71/'+arm+'/'+kind]
   vals=[d[k]['speaker_equal_mean_RMS'] for k in ['B_own_prototype_change','S_applied_projected_change','S_preprojection_half_delta','projection']]
   lines.append('| '+arm+'/'+kind+' | '+' | '.join(f'{v:.6f}' for v in vals)+f" | {d['projection_fraction']['clip_median']:.3%} |")
 lines.append('')
 desc=read(RUN/'operation_descriptions.json.gz')
 for kind in ['rounding_pre','rounding_post']:
  maximum=max(v[kind]['max_abs'] for v in desc.values());lines.append(f'- `(own+t32)−old other` {kind} 最大绝对浮点余差：{maximum:.17g}。旧other不为追闭合重生成。')
 lines+=['','有投影处不能称保持原动态；即使无投影，phone条件场仍随标签/边界变化。大量零激活可使投影比例高而数值改变量小，不能只凭比例或RMS认定根因。','',
 '## 5. 工程与复核','',
 '- 原15LOSO fit不变，无新fit/波形/MFA/音频forward。全部600旧own/other输入hash exact、旧官方曲线重放max0；1411旧事件字节引用。',
 '- 新01固定subtract32→multiply32(.5)→add32→maximum32，非speech直接copy；实时native z逐批与父cache exact。',
 '- 首cal8个现场AVI经独立FFmpeg/PTS/JPEG/重复审计；4组新替换/旧alias exact。描述性RMS/L2独立差8.88e−16，不能把描述也写成逐位exact。',
 '- 全296创建时像素/PTS/JPEG门由producer执行；外部最终只核封存数组/metadata，不冒称已删除eval视频的现场独立复核。',
 '- 最终独立核300组替换/输入描述最大差2.66e−15；2,019,092距离项、180,608事件值、7020统计最大差0，旧三cell/事件字节exact。',
 f"- 所有分量闭合最大误差：{a['factorial_closure_max']:.17g}；完整独立数值误差/输入hash见independent_validation.json绑定的receipts。",'',
 '## 6. 边界与预定条件后续','',
 '模板来自TTS/自然音频经Wav2Lip audio_encoder的z，不是TTS生成视频的成分；潜在收益只说明此音频表征操作可利用，不能直接确认/反驳视频成分迁移。历史同一模型/句子/speakers上的机制探索、固定cal推断；不是物理嘴型质量，不是跨模型通则。',
 '事前条件路线：若自然phone或global的raw71 C与anchor两者99CI均正，才考虑80自然句的开发集外正/反迁移；触发仍两family全做。该80曾用于其他分析且缺z/未绑定phone时间轴，仍需独立工程协议，不因当前结果直接跳过门。','']
 for kind in ['phone','global']:
  trigger=all(s['raw/guard20/common71/'+m+'/'+kind+'/source_S/N']['ci99'][0]>0 for m in ['C','C_anchor'])
  lines.append(f'- 预定{kind} C+anchor双正条件：{trigger}。')
 lines+=['','## 7. 资产与资源','',
 'protocol v2、parent_binding、fit原引用、feature_seal_evaluation、scores/*.json.gz、全部query/clip数组与summary保存；artifact_hashes逐项绑定。独立160MiB（main144+audit16），free扣余承诺≥4.05GiB；父320MiB/4.25GiB阶段已先结项，不回写其门。GPU累计≤60min含live；worker计时与外层每秒监督并行，临时96MiB。最终释放/精确字节见resource_closure.json。','']
 (RUN/'report.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
