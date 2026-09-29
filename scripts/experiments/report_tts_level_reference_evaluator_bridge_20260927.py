"""Compact reporting of the frozen CPU reference-condition E-only bridge."""
from pathlib import Path
import hashlib,json,time,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_level_reference_evaluator_bridge_20260927'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(r):return f"{r['mean']:+.3f} [{r['ci99'][0]:+.3f}, {r['ci99'][1]:+.3f}]"
def main():
    p=read(OUT/'protocol.json');s=read(OUT/'summary.json');v=read(OUT/'independent_validation.json');assert v['passed'];b=read(OUT/'input_validation.json');raw=read(OUT/'raw_baseline_audit.json');assert b['passed'] and raw['passed'];cal=read(OUT/'calibration_transfer.json')
    lines=['# 电平评价音频作用的参考条件缓存桥接','',
      '仅CPU缓存距离评分；没有生成新音频、视频或模型特征。协议在LEVEL主结果揭示前冻结，待父run特征自然封存后执行，不读取其主scores选择设计。', '',
      '**估计量仅为E-only**：每个固定的原生成视频V比较C(V,A_LEVEL)−C(V,A_RAW)。视频仍由原未处理音频生成，下面的“LEVEL评价差距”不是LEVEL新生成视频的原配总效应。所有D0/D66/S0/S66条件保留；主分析继承源48eval/13speaker及各臂官方L，固定k3。', '',
      '## 主分析：generated、raw guard20、原48eval/13speaker', '',
      '| 参考条件 | N的E响应 | T的E响应 | T−N的E响应差 | 原始评价T−N | LEVEL评价T−N（固定原视频） |','|---|---:|---:|---:|---:|---:|']
    for c,block in s['evaluation48/generated/raw/guard20'].items():
        d=block['C'];lines.append('| '+c+' | '+' | '.join(f(d[k]) for k in ['N','T','T_minus_N','raw_gap','LEVEL_evaluation_gap'])+' |')
    lines+=['','四种参考条件的T−N E-only响应差点估计均为负；D0、D66和S66的99%CI不含0，S0为−0.033 [−0.076,+0.001]。电平评价音频作用在多个参考条件下保留，但不能称四条件均确认，也未检验其大小相等。只替换评价音频后，四条件TTS−自然差距仍为正。']
    lines+=['','speaker等权20000次bootstrap，PCG64 seed20260926，逐比较99%CI，不声称FWER。跨零不等于等效。', '', '## 同48支持的敏感性', '', '| 几何/支持 | 参考条件 | C的E响应差(T−N) | 固定k3 C_anchor的E响应差 |','|---|---|---:|---:|']
    for geom in ['raw','unit']:
      for policy in ['guard20','valid','guard0']:
       for c,d in s[f'evaluation48/generated/{geom}/{policy}'].items():lines.append(f"| {geom}/{policy} | {c} | {f(d['C']['T_minus_N'])} | {f(d['C_anchor']['T_minus_N'])} |")
    lines+=['','## 预设source-only附录（原48、raw guard20）','', 'source-only为同源参考序列的现成视觉特征，不是嘴型真值。', '', '| 条件 | N的E | T的E | T−N的E响应差 |','|---|---:|---:|---:|']
    for c,d in s['evaluation48/source_only/raw/guard20'].items():lines.append('| '+c+' | '+' | '.join(f(d['C'][k]) for k in ['N','T','T_minus_N'])+' |')
    lines+=['','## 原74总样本附录（含26cal，不替代48eval主分析）','', '| 视图 | 条件 | C的E响应差(T−N) | C_anchor的E响应差 |','|---|---|---:|---:|']
    for geom in ['raw','unit']:
      for policy in ['guard20','valid','guard0']:
       for c,d in s[f'all74_appendix/generated/{geom}/{policy}'].items():lines.append(f"| {geom}/{policy} | {c} | {f(d['C']['T_minus_N'])} | {f(d['C_anchor']['T_minus_N'])} |")
    lines+=['','## 复用与独立核查','',
      f"原PCM input hash逐一相同、16k与样点/官方联合L不变、新音频特征覆盖全部原L；RAW共同前缀embedding最大差{b['errors']['raw_embedding']:.3g}，实际MFCC窗口最大差{b['errors']['raw_mfcc_window']:.3g}。模型hash及共同前端代码一致；source与LEVEL运行环境版本差异记录在input_validation.json，数值一致门全部通过。",f"新RAW重算源各条件native C的最大差{raw['max_C_error']:.3g}，门限1e−5。此门对全部74、四条件、generated/source-only、raw/unit、三种支持先完成，之后才算任何LEVEL距离。",f"父LEVEL全局gain操纵/独立波形门和原source/LEVEL的首cal N/T重复及+5帧控制均核验继承，未新增模型任务。LEVEL各cal条件median lag={cal['LEVEL_medians']}；固定k3迁移充分标记={cal['transfer_flag_sufficient']}，不改k、不删样。",f"独立核查{v['distance_entries']}项距离：最大距离误差{v['max_distance_error']:.3g}、指标误差{v['max_metric_error']:.3g}、E效应及全部汇总/bootstrap误差{v['max_statistics_error']:.3g}；主48/附录74成员完整一致。",'',
      '完整C/B/D/C_anchor/D_anchor、各臂响应和source-only敏感性见summary.json，逐样本见effects.json/scores。所有输入、支持与代码绑定见protocol.json、input_validation.json、score_lock.json，产物见artifact_hashes.json。', '',
      '## 限制','',
      '这是同输入在不同参考几何/运动条件下的探索，非新speaker确认；参考条件同时改变嘴部、姿态等，不能称单一嘴型因素。各条件共用原source ROI；与static image3的ROI不同，因此不将跨ROI分数相减当作生成效应。根因判读仍以独立LEVEL主四格为主，本桥只检验评价音频响应的参考依赖。没有因结果选择某个D/S条件。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    cp=OUT/'code_snapshot'/Path(__file__).name;shutil.copyfile(__file__,cp)
    final={'status':'concluded','validation_passed':True,'finished_epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'report_sha256':sha(OUT/'report.md'),'cpu_only':True,'new_model_inference':False,'n_primary':48,'speakers_primary':13,'source_only_appendix':True}
    (OUT/'final.json').write_text(json.dumps(final,indent=2)+'\n');artifacts={str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['artifact_hashes.json','report.log']};(OUT/'artifact_hashes.json').write_text(json.dumps(artifacts,indent=2)+'\n');print('artifacts',len(artifacts))
if __name__=='__main__':main()
