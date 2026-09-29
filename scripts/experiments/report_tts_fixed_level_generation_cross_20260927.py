"""All frozen FIXED endpoints and streamed retention disclosure."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_fixed_level_generation_cross_20260927'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v):return f"{v['mean']:+.3f} [{v['ci99'][0]:+.3f}, {v['ci99'][1]:+.3f}]"
s=read(OUT/'summary.json');p=read(OUT/'protocol.json');v=s['results']['raw/guard20/common71']['C'];checker=read(OUT/'independent_validation_all.json');hashes=read(OUT/'independent_hashes.json');assert checker['status']=='passed' and hashes['status']=='PASS'
lines=['# Cal-only 固定绝对 RMS 生成／评价四格','', '## 问题与冻结设计','',f"目标 RMS = **{p['fixed_target']['target_RMS']:.16g}**，仅由52个cal原声臂的log RMS中位得到。FIXED 每条输入独立以 target/rms(x) 缩放；没有测试时配对另一臂信息，没有eval估计目标或额外headroom。所有200臂声学与独立标量门通过，峰值≤0.98；保留全部100对，26cal/74eval。",'', 'Wav2Lip static image3，生成框[138,90,357,387]，25fps/FFV1/batch32；SyncNet固定ROI[33,18,462,447]→224/JPEG。FLOAT32波形、MFCC以float64×32768；同臂所有cell joint L=min(F,PCM//640)−5，共同query，C±15lag，anchor固定k=3。', '', '主分析raw/guard20/common71，15 speakers等权，20k bootstrap seed20260926/99CI，无FWER。G=q10−q00，E=q01−q00，I=q11−q10−q01+q00，total=q11−q00。','', '## 主要结果','',f"原native T−N：**{fmt(v['baseline_T_minus_N'])}**。",'', '| 操作 | G 的 T−N | E 的 T−N | I 的 T−N | total 的 T−N | 处理后 native T−N |','|---|---:|---:|---:|---:|---:|']
for c in ['LEVEL','FIXED']:
 d=v['conditions'][c];lines.append('| '+c+' | '+' | '.join(fmt(d[e]['T_minus_N']) for e in ['G','E','I','total'])+' | '+fmt(d['processed_T_minus_N'])+' |')
lines+=['','| FIXED−LEVEL 操作差 | N | T | T−N |','|---|---:|---:|---:|']
for e,d in v['contrasts']['FIXED_minus_LEVEL'].items():lines.append('| '+e+' | '+' | '.join(fmt(d[a]) for a in ['N','T','T_minus_N'])+' |')
lines+=['','| FIXED 两臂作用 | N | T |','|---|---:|---:|']
for e in ['G','E','I','total']:lines.append('| '+e+' | '+fmt(v['conditions']['FIXED'][e]['N'])+' | '+fmt(v['conditions']['FIXED'][e]['T'])+' |')
lines+=['','## 全部冻结敏感性','', '| view | FIXED total T−N | FIXED native T−N | FIXED−LEVEL total T−N |','|---|---:|---:|---:|']
for view,allmetrics in s['results'].items():
 d=allmetrics['C'];lines.append('| '+view+' | '+fmt(d['conditions']['FIXED']['total']['T_minus_N'])+' | '+fmt(d['conditions']['FIXED']['processed_T_minus_N'])+' | '+fmt(d['contrasts']['FIXED_minus_LEVEL']['total']['T_minus_N'])+' |')
lines+=['','全部C/B/D、anchor C/D、bestlag、各臂各效应及曲线见 summary.json、effects.json、scores/*；sigmaA/sigmaV/sigmaM/covAV仅描述，z由observer保存，不拟合或计算中介比例。','', '## 工程控制、严格缓存与复核','',f"600个RAW/identity/LEVEL臂条件从已concluded LEVEL run严格复用。原PCM与实际mel/MFCC输入逐元素相同，图像、模型、前端、batch规则、frames与query一致且来源hash可核验。首cal N/T identity新重复pixel/features精确，原PCM/FLOAT等幅桥接与+5delay控制见cal_controls.json。cal engineering={s['cal_controls']['engineering_passed']}，anchor transfer={s['cal_controls']['anchor_transfer_passed']}；k未重估。",'', '首cal N/T FIXED分别完整保留与临时流式重复生成，pixel/MFCC/A/V/z/query逐元素相同；持久化重新打开逐元素相同才放行eval，见controls/*/streaming.json。', '', f"独立距离复核 {checker['distance_entries']} 个值，最大误差 {checker['max_distance_error']:.3g}；统计误差 {checker['max_statistics_error']:.3g}；四格闭合 {checker['max_fourcell_closure']:.3g}。",'', '## 产物保留与资源限制','', '**所有52个cal FIXED视频及identity控制重复保留。148个eval FIXED视频未长期保留。** 每个eval视频仅在本run临时目录产生，通过原版JPEG前端提取A/V和z，保存音频、frontend、features、曲线/逐样本scores、file/pixel hash、源与模型参数及RNG状态；feature/metadata原子提交并重新打开验证后移除该新临时视频。retention_commits记录创建时hash。这些记录不等于不存在视频的最终文件hash复核。封存输入、代码、参数、模型与RNG可重建。未改动旧数据、未写CloudFS。', '', '新增独占磁盘≤0.75GiB，持续保留5GiB。600缓存特征为旧inode硬链接，不计为新分配视频。资源快照与阶段lease见produce_*_resource_gate.json、gpu_release_*.json和completion_resource.json。', '', '## 解释边界','', 'FIXED验证cal-only固定目标下的同一批数据响应，不是独立新数据确认。RMS均衡不是LUFS或听感响度完全均衡。FIXED−配对LEVEL改变两臂共同operating level，不能换算额外解释百分比。固定video换audio是评价路径，固定audio新video是生成路径的Sync-C效应，不能自动称真实嘴型改善。目标保留时间与谱形，但Wav2Lip mel及SyncNet MFCC对幅度可有前端依赖。', '', '## 封存指针','', f"protocol SHA256 `{sha(OUT/'protocol.json')}`；support `{sha(OUT/'support.json')}`；独立声学门、哈希与全部统计分别见 independent_acoustic.json、independent_hashes.json、independent_validation_all.json。"]
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
fig,ax=plt.subplots(figsize=(8,4.8));labels=['G','E','I','total'];colors={'LEVEL':'#2f6e9c','FIXED':'#c96f32'}
for j,c in enumerate(['LEVEL','FIXED']):
 vals=[v['conditions'][c][k]['T_minus_N'] for k in labels];x=np.arange(4)+(j-.5)*.17;y=np.array([q['mean'] for q in vals]);lo=np.array([q['ci99'][0] for q in vals]);hi=np.array([q['ci99'][1] for q in vals]);ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o',capsize=4,label=c,color=colors[c])
ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(4),labels);ax.set_ylabel('Change in T − N Sync-C');ax.set_title('Fixed target versus pair midpoint RMS\nraw / guard20 / common71; speaker equal 99% CI');ax.legend();fig.tight_layout();fig.savefig(OUT/'fixed_gxe_effects.png',dpi=180);fig.savefig(OUT/'fixed_gxe_effects.pdf');plt.close(fig)
final={'status':'concluded','validation':'PASS','finished_epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'report_sha256':sha(OUT/'report.md'),'summary_sha256':sha(OUT/'summary.json'),'streamed_eval_videos_not_retained':148,'all_cal_FIXED_videos_retained':52}
(OUT/'final.json').write_text(json.dumps(final,indent=2)+'\n');files={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='artifact_hashes.json' and f.suffix!='.log'};(OUT/'artifact_hashes.json').write_text(json.dumps(files,indent=2)+'\n');print(final);print('artifacts',len(files))
