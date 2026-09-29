"""Report-only rendering of frozen natural80 statistics; never creates effect estimates."""
from pathlib import Path
import csv,io,json,math,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments import tts_fixed_generator_shift_holdout80_20260927 as core
O=core.OUT;read=core.read

def put(path,text):
 data=text.encode();core.limits(math.ceil(len(data)/4096)*4096);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(data)
 core.limits()
def run():
 p,rows=core.locked();s=read(O/'summary.json.gz');a=read(O/'analysis.json');assert a['statistic_count']==len(s)==896
 def f(key):
  x=s[key];return f"{x['mean']:+.6f} [{x['ci99'][0]:+.6f}, {x['ci99'][1]:+.6f}]"
 def q(g,pol,m,fam,c):return f(g+'/'+pol+'/'+m+'/'+fam+'/'+c)
 out=io.StringIO();w=csv.writer(out,delimiter='\t');w.writerow(['endpoint','mean','ci95_lo','ci95_hi','ci99_lo','ci99_hi','clips','speakers'])
 for key,v in s.items():w.writerow([key,v['mean'],*v['ci95'],*v['ci99'],v['n'],v['speakers']])
 put(O/'all_endpoints.tsv',out.getvalue())
 dose={}
 for c in core.CONDS:
  mm=[read(O/'metadata'/r['id']/'N'/(c+'.json'))['mix'] for r in rows]
  dose[c]={k:{'median':float(np.median(v)),'min':float(min(v)),'max':float(max(v))} for k,v in {'applied_RMS':[x['applied']['RMS'] for x in mm],'pre_relu_negative_fraction':[x['pre_relu_negative_fraction'] for x in mm],'projection_RMS':[x['projection']['RMS'] for x in mm]}.items()}
 core.write(O/'report_dose_descriptions.json',dose)
 cg=read(O/'mfa_cal_gate.json');eg=read(O/'mfa_evaluation_gate.json');exit=read(O/'gpu_runtime/supervised_exit_evaluation.json');ind=read(O/'independent_binding.json') if (O/'independent_binding.json').exists() else None
 lines=['# 80句自然语音：固定自然音轨下生成路径的来源方向迁移','',('状态：concluded，独立复核 PASS。' if ind else '状态：全部评分分析完成，待独立复核。'),'','## 问题与冻结判据','',
 '开发集 global 来源位移属于预设次要分支，其 C 与固定 k3 anchor 阳性触发此迁移；原 phone 主分支未成功。这里将同一20cal pooled自然/TTS音频编码原型方向，施加到40个不同音频speaker的80句自然语音，固定原FIXED评价A，不生成新TTS。80句以前分析过其他端点，称新处理的开发集外迁移与反向控制，不称完全未见确认或模型预训练独立。','',
 '方向来源是Wav2Lip音频编码z，不是TTS视频信息。global+/−及phone+/−各80，共320个真实新生成，图3/ROI/25fps/原时钟与支持固定。δ来自已有S0912全20cal pooled模板；post-ReLU z上subtract32、multiply32(.5)、add32、ReLU，非speech原样。未用80分数拟合/调剂量/选支持。','',
 '主raw guard20须global+−baseline及global+−global−，各在C和原k3 C_anchor的99CI下界均>0，四门合取。40speaker先各2句配对效应均值，再等权20k PCG64 seed20260926 bootstrap，95/99CI均完整保存；各CI及合取不声明FWER。','',
 '## 主四门与反向控制','', '|对比|ΔC [99CI]|ΔC_anchor [99CI]|','|---|---|---|']
 for c,label in [('plus_minus_baseline','global+ − baseline'),('plus_minus_minus','global+ − global−'),('minus_minus_baseline','global− − baseline')]:lines.append('|'+label+'|'+q('raw','guard20','C','global',c)+'|'+q('raw','guard20','C_anchor','global',c)+'|')
 lines += ['',f"四门：`{a['four_primary_tests']}`；合取 **{a['global_direction_transfer_supported']}**。",'',
 '主结果支持这一小幅TTS来源音频编码方向在固定自然评价音轨下带来生成路径Sync-C收益，并优于反号；anchor同向改善，不能仅归为最佳lag搜索上浮。收益数值很小，不等同可见嘴型改善，也不能按比例解释原TTS−自然整体优势。','',
 '## 主端点的距离组成','', '|对比|ΔB [99CI]|ΔDmin [99CI]|ΔD_anchor [99CI]|Δsearch uplift [99CI]|','|---|---|---|---|---|']
 for c,label in [('plus_minus_baseline','global+ − baseline'),('minus_minus_baseline','global− − baseline'),('plus_minus_minus','global+ − global−')]:lines.append('|'+label+'|'+'|'.join(q('raw','guard20',m,'global',c) for m in ['B','D','D_anchor','search_uplift'])+'|')
 lines += ['','global正向的B变化未确认；Dmin与D_anchor均降低。search uplift变化CI跨0，不能称等效为0。此为同输入方向干预结果，不是原生T−N分差的中介分解。','',
 '## 固定次要phone结果','', '|对比|ΔC [99CI]|ΔC_anchor [99CI]|','|---|---|---|']
 for c,label in [('plus_minus_baseline','phone+ − baseline'),('minus_minus_baseline','phone− − baseline'),('plus_minus_minus','phone+ − phone−')]:lines.append('|'+label+'|'+q('raw','guard20','C','phone',c)+'|'+q('raw','guard20','C_anchor','phone',c)+'|')
 lines += ['','phone+相对baseline的C和anchor均未确认改善；phone−损伤明确。主guard20 phone正向−反向的C跨0、anchor为正，完整保留，不能将phone分支称成功。phone模板含邻接上下文/相位组成，并非孤立音素身份。','',
 '## 全8视图：两family与两方向','', '下表全部同80支持；raw/unit分开，不跨尺度算比例。每项为mean [99CI]；完整B/D/anchor/uplift/bestlag/offset及绝对cell、phone−global对比的95/99CI见all_endpoints.tsv与summary.json.gz。','',
 '|geometry/policy|family/contrast|ΔC [99CI]|ΔC_anchor [99CI]|','|---|---|---|---|']
 for g in ['raw','unit']:
  for pol in ['guard20','valid','guard0','guard15']:
   for fam in ['global','phone']:
    for c,label in [('plus_minus_baseline','+ − baseline'),('minus_minus_baseline','− − baseline'),('plus_minus_minus','+ − −')]:lines.append(f'|{g}/{pol}|{fam} {label}|'+q(g,pol,'C',fam,c)+'|'+q(g,pol,'C_anchor',fam,c)+'|')
 lines += ['','阴性敏感性保留：global+−baseline的raw guard15 C 99CI跨0；unit valid的C和anchor均跨0。global正向−反向在全部8视图C/anchor均为正；并不因此抹去相对baseline的这些不确定结果。所有条件guard20的median bestlag均为3，仅描述、不重选k。','',
 '## 输入、处理强度与支持','', f"MFA3.4.1原26cal输入检查通过：{cg['matched']}/{cg['active']}活跃中心被原82模板标签覆盖（{cg['coverage']:.4%}）；80输入 {eg['matched']}/{eg['active']}（{eg['coverage']:.4%}），未覆盖speech按source global回退，silence/unknown/gap identity，未删任何clip。cal20模板仍仅原723双臂occurrence，当前MFA只定义新输入mask，不重fit。",'',
 '|条件|实际speech位移RMS clip中位 [min,max]|pre-ReLU负坐标比例clip中位|投影RMS clip中位|','|---|---|---|---|']
 for c,d in dose.items():
  v=d['applied_RMS'];lines.append(f"|{c}|{v['median']:.8f} [{v['min']:.8f},{v['max']:.8f}]|{d['pre_relu_negative_fraction']['median']:.6f}|{d['projection_RMS']['median']:.8f}|")
 lines += ['','以上只是已封输入metadata的描述，不按剂量/投影筛选；正反号及phone/global经ReLU后实际位移可不同，相同λ不保证有效剂量相同。投影、mask边界与模型非线性均包含在操作内。','',
 '## 工程与独立复核','',
 '旧runtime（Torch2.5.1+cu124）精确重放旧FIXED80基线：CPU全部8view曲线/指标及原99CI/speaker均值max0；GPU全80原像素创建hash/A/V exact，首次2条none/noop/cached/full-stream控制exact。原78条旧视频未保留，不能声称独立重解码它们。新80 native z实提并保存；处理时每batch native hook exact。首处理4条件各repeat、full-stream、像素/PTS/JPEG工程门通过。','',
 '外部独立现场审核限首identity与首处理现场；其余320视频producer在创建时另FFmpeg核像素/帧/PTS并封存hash，随后释放自己临时视频；最终独立只可复核保存数组/来源绑定/距离/统计，不能伪称重解码已删除的320媒体。', '',
 ('独立结果receipt：`'+ind['receipt']+'`，SHA `'+ind['receipt_sha256']+'`。' if ind else '独立结果receipt待绑定；本报告结果暂不用于放行条件后续。'), '',
 f"CPU预定896统计全保留，配对闭合max {a['paired_contrast_closure_max']:.3g}。GPU累计含live {exit['cumulative_wall_seconds']/60:.2f}min<90，exit0、compute空、lease可用、tmp0。监督器maxrss只代表监督进程，不能当完整worker峰值RAM。main176MiB/audit16MiB、floor4GiB门始终保留；精确最终占用见resource_closure.json。",'',
 '## 条件后续状态','',
 '此run四门独立通过可触发已预先准备的分量机制路线，但它不是自动GPU授权。后续仍需本run完整结项、资源闭合、新完整实现/输入审阅与root裁决。v4 strict_orth准备的cal数值检查不改变本run任何科学端点或输入；同80后续不称第二份独立确认。']
 put(O/'report.md','\n'.join(lines)+'\n');print('REPORT_WRITTEN',len(s))
if __name__=='__main__':run()
