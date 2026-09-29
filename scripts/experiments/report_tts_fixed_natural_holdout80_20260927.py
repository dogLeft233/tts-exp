"""Final scientific report and figures for the predeclared natural80 transfer."""
from pathlib import Path
import hashlib,json,time,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_fixed_natural_holdout80_20260927'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v):return f"{v['mean']:+.3f} [{v['ci99'][0]:+.3f}, {v['ci99'][1]:+.3f}]"
p=read(OUT/'protocol.json');s=read(OUT/'summary.json');v=s['results']['raw/guard20']['C'];check=read(OUT/'independent_scores.json');hs=read(OUT/'independent_hashes.json');assert check['status']==hs['status']=='PASS';assert read(OUT/'independent_acoustic.json')['status']=='PASS';assert read(OUT/'engineering_gate.json')['passed']
g=v['effects']['G'];interpretation='主G的99%区间完全高于0，支持在这80句上自然生成路径的正向Sync-C响应。' if g['ci99'][0]>0 else '主G的99%区间完全低于0，表明这80句上自然生成路径的Sync-C响应为负。' if g['ci99'][1]<0 else '主G的99%区间包含0，本迁移未确认方向；这不等于证明无效或等效。'
lines=['# 固定绝对RMS在80句自然音频上的生成路径迁移','', '## 结果','',f"**主要G-only = {fmt(g)} Sync-C。** {interpretation}",'', '| raw/guard20，80句/40人 | speaker等权均值 [99% CI] |','|---|---:|']
for e in ['G','E','I','total']:lines.append('| '+e+' | '+fmt(v['effects'][e])+' |')
lines+=['| RAW native C | '+fmt(v['baseline'])+' |','| FIXED native C | '+fmt(v['FIXED_native'])+' |','','G固定原自然评价音轨，只换FIXED驱动生成的视频；E固定RAW视频，只换FIXED评价音频；I为四格非加性交互。这里没有TTS臂，没有T−N端点。','', '## 冻结设计与独立范围','',f"目标RMS严格 **{p['target_RMS']:.16g}**，来自此前cloud100旧26cal的52臂log RMS中位。每条自然输入独立gain=target/rms(x)，无eval或配对T信息、无cap/额外headroom、无新TTS。全部80原输入保留，每speaker2句，L范围68–190，所有guard20可用；时间轴和各cell query完全一致。",'', '这40个canonical音频speaker和80个PCM与当前cloud100/旧26cal开发输入零交集，来源身份/hash已由feasibility和先前只读审查绑定。80句曾用于其它分析，不称整个数据集未看过；不主张全项目绝对真人身份或模型预训练独立。root说明分支选择发生在其看到cloud100 FIXED新分数之前，记录为决策说明，不冒称外部时间戳证明。', '', '原static图像3，生成框[138,90,357,387]，25fps/FFV1/batch32，SyncNet ROI[33,18,462,447]→224/JPEG。旧holdout环境~/.venvs/syncnet（torch2.5.1+cu124/numpy2.2.6等六项记录版本完全一致）。新FIXED为FLOAT32 waveform，MFCC float64×32768；L=min(F,PCM//640)−5，C±15lag，anchor固定k=3。', '', '主端点raw/guard20 C的G-only；G/E/I/total与C/B/D/anchor C/D/bestlag全部报告。两句先speaker内平均，再40speaker等权，20k bootstrap seed20260926，99% CI，无FWER。没有按得分筛选、重估lag/目标或改变分母。', '', '## 全部边界和几何敏感性','', '| view | G | E | I | total |','|---|---:|---:|---:|---:|']
for view,d in s['results'].items():lines.append('| '+view+' | '+' | '.join(fmt(d['C']['effects'][e]) for e in ['G','E','I','total'])+' |')
lines+=['','完整逐句四格、所有指标和31lag曲线见scores/、effects.json、summary.json。各cell的sigmaA/sigmaV/sigmaM/covAV仅描述，不拟合或计算中介比例。', '', '## 门控和复核','', '原80音频、旧RAW视频/A/V缓存与模型/前端hash均核验。旧RAW/identity共享严格原幅度前端；首预定两句BAC009S0002W0122、BAC009S0002W0123在新进程实际重渲染，解码pixel和完整A/V逐元素一致后才渲染任何FIXED视频。PCM/FLOAT MFCC与音频feature桥接精确。新PCM+5delay使用固定共同query25:L−25，记录原/延迟lag和重叠曲线。', '', '全部80保存FLOAT先通过峰值≤0.98、长度、RMS、float64/float32 scalar inverse/zero/cosine/谱形/R不变量及独立int16/FFT复核。首两句FIXED完整/流式重复pixel/MFCC/A/V/query逐元素一致，所有视频创建时独立解码pixel hash等于render像素流。', '', f"80句RAW/FIXED diagonal median lag：{s['lag_transfer']['medians']}；raw迁移门={s['lag_transfer']['raw_passed']}。无论迁移门状态，k和支持均未修改。", '', f"独立距离 {check['distance_entries']} 项，最大误差 {check['max_distance_error']:.3g}；统计误差 {check['max_statistics_error']:.3g}；四格闭合 {check['max_fourcell_closure']:.3g}；{check['curves_and_argmin_exact']} 条曲线及argmin精确；{check['bridge_identities']} 个尺度协方差恒等式最大误差 {check['max_bridge_identity']:.3g}。", '', f"现存文件hash复核 {hs['existing_hashes_verified']} 项，80旧RAW视频和2新FIXED保留视频已核验；78新FIXED仅核验创建封存记录，不声称不存在视频的最终文件hash被再次验证。", '', '## 保留与资源','', '**仅预定前2条新FIXED完整视频长期保留；另78条及控制重复视频不长期保留。** 新临时视频经原版JPEG前端提取A/V，保存波形、frontend、A/V、file/pixel hash、源模型/代码/参数/RNG与元数据，原子提交并重读后删除。旧80 RAW视频一律不动。可从封存材料重建；最终保留hash与创建时hash严格区分。没有保存z，以遵守预算。', '', '预算新增≤0.20GiB、磁盘≥5GiB，每32帧渲染batch与持久化cell检查；GPU共享lease且等待visual最终释放后运行，结束释放。无旧数据删除、无CloudFS写入、无API调用。最终用量见completion_resource.json。', '', '## 解释边界','', '该结果仅检验自然语音生成路径的Sync-C响应跨当前开发集canonical音频speaker迁移。它不确认云TTS−自然gap，不等于真实嘴型识别/质量改善，不测试新脸，不构成全项目或模型预训练的绝对独立确认。RMS不是LUFS或听感响度完全均衡；单一全局增益仍会经Wav2Lip mel和SyncNet MFCC前端进入模型。', '', '## 封存','',f"protocol SHA256 `{sha(OUT/'protocol.json')}`；input/support `{sha(OUT/'support.json')}`。源码快照、CPU seal、工程gate、feature seal、score lock、逐条retention commit、独立复核及artifact_hashes.json共同定义本run。"]
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
fig,axs=plt.subplots(1,2,figsize=(10,4.5));effects=['G','E','I','total']
for ax,geom in zip(axs,['raw','unit']):
 d=s['results'][geom+'/guard20']['C']['effects'];m=np.array([d[e]['mean'] for e in effects]);lo=np.array([d[e]['ci99'][0] for e in effects]);hi=np.array([d[e]['ci99'][1] for e in effects]);ax.errorbar(range(4),m,yerr=[m-lo,hi-m],fmt='o',capsize=4,color='#336f95');ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(4),effects);ax.set_ylabel('Change in Sync-C');ax.set_title(geom+' / guard20')
fig.suptitle('FIXED RMS: natural80 / 40 audio speakers\nSpeaker equal effects and 99% CI');fig.tight_layout();fig.savefig(OUT/'natural80_fixed_effects.png',dpi=180);fig.savefig(OUT/'natural80_fixed_effects.pdf');plt.close(fig)
seen=set();total=0
for f in OUT.rglob('*'):
 if f.is_file():
  z=f.stat();key=z.st_dev,z.st_ino
  if key not in seen and z.st_nlink==1:total+=z.st_blocks*512
  seen.add(key)
free=shutil.disk_usage(OUT).free;assert total<=.20*2**30 and free>=5<<30
(OUT/'completion_resource.json').write_text(json.dumps({'time':time.time(),'new_allocated_bytes':total,'new_allocated_GiB':total/2**30,'free_GiB':free/2**30,'budget_passed':True,'reserve_passed':True,'reused_hardlinks_excluded':True},indent=2)+'\n')
final={'status':'concluded','validation':'PASS','finished_epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'report_sha256':sha(OUT/'report.md'),'summary_sha256':sha(OUT/'summary.json'),'primary_G':g,'new_FIXED_retained_videos':2,'new_FIXED_streamed_videos':78};(OUT/'final.json').write_text(json.dumps(final,indent=2)+'\n');artifacts={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='artifact_hashes.json' and f.suffix!='.log'};(OUT/'artifact_hashes.json').write_text(json.dumps(artifacts,indent=2)+'\n');print(final);print('artifacts',len(artifacts))
