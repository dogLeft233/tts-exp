"""Score already sealed audio-only DTW maps; cannot alter their construction."""
import argparse
import json
import shutil
import numpy as np
from scripts.experiments.tts_instance_audio_dtw import ROOT,SOURCE,OUT,ORACLE,read,write,sha
from scripts.experiments.tts_pcm_residual import cluster

def run():
    seal=read(OUT/'query_mapping_seal.json');assert sha(OUT/'audio_mapping_seal.json')==seal['audio_mapping_seal_sha256']
    for group in ('support_assets','mapping_assets'):
        for f,h in seal[group].items():assert sha(f)==h
    write(OUT/'scoring_start.json',{'query_mapping_seal_sha256':sha(OUT/'query_mapping_seal.json'),'audio_mapping_seal_sha256':sha(OUT/'audio_mapping_seal.json'),'note':'both seals verified before reading embeddings or oracle score arrays'})
    inputs={}
    for label in ('primary','mfa_only'):
        for k in (3,2,4):
            mappings=read(OUT/'indices'/f'{label}_k{k}.json');oracle_file=ORACLE/f'{label}_k{k}'/'queries.json';inputs[str(oracle_file)]=sha(oracle_file)
            refs={(r['id'],r['query'],r['geometry'],r['mode']):r for r in read(oracle_file) if r['mode'] in ('r0','r80')}
            records=[];cells=[];cache={}
            for r in mappings:
                sid=r['id']
                if sid not in cache:
                    f=SOURCE/'audio_features'/(sid+'.npz');inputs[str(f)]=sha(f);zz=np.load(f);aa={a:zz[a].astype(float) for a in ('Q1','Q2')};vv={}
                    for a in aa:
                        f=SOURCE/'scores/3'/sid/(a+'.npz');inputs[str(f)]=sha(f);vv[a]=np.load(f)['visual'].astype(float)
                    cache[sid]=(aa,vv)
                original_a,original_v=cache[sid]
                for geo in ('raw','unit'):
                    aa,vv=original_a,original_v
                    if geo=='unit':aa={a:x/np.linalg.norm(x,axis=1)[:,None] for a,x in aa.items()};vv={a:x/np.linalg.norm(x,axis=1)[:,None] for a,x in vv.items()}
                    eff={}
                    for a,b in (('Q1','Q2'),('Q2','Q1')):
                        name=a+b;d=r['directions'][name];v=vv[a][d['visual_i']]
                        selfd=float(np.linalg.norm(v-aa[a][d['self_j']]+1e-6));mfa=float(np.linalg.norm(v-aa[b][d['original_j']]+1e-6));dt=float(np.linalg.norm(v-aa[b][d['chosen_j']]+1e-6));oracle=refs[sid,r['query'],geo,'r80']['directions'][name]['cross']
                        assert abs(mfa-refs[sid,r['query'],geo,'r0']['directions'][name]['cross'])<1e-10 and dt>=oracle-1e-10
                        eff.update({name+'_self':selfd,name+'_mfa_gap':mfa-selfd,name+'_dtw_gap':dt-selfd,name+'_oracle_gap':oracle-selfd,name+'_dtw_minus_mfa':dt-mfa,name+'_win':float(dt<=selfd),name+'_abs_shift_ms':abs(d['shift_ms']),name+'_shift_ms':d['shift_ms'],name+'_changed':float(d['chosen_j']!=d['original_j']),name+'_fallback':float(d['fallback']),name+'_endpoint_clamped':float(d['endpoint_clamped']),name+'_target_outside':float(d['target_outside_candidate_range'])})
                    for metric in ('self','mfa_gap','dtw_gap','oracle_gap','dtw_minus_mfa','abs_shift_ms','shift_ms','changed','fallback','endpoint_clamped','target_outside'):
                        eff[metric]=.5*(eff['Q1Q2_'+metric]+eff['Q2Q1_'+metric])
                    eff['win']=float(eff['dtw_gap']<=0);records.append({**{key:r[key] for key in ('id','speaker','query')},'geometry':geo,'effects':eff})
            for sid in sorted(cache):
                for geo in ('raw','unit'):
                    rr=[r for r in records if r['id']==sid and r['geometry']==geo];cells.append({'id':sid,'speaker':rr[0]['speaker'],'geometry':geo,'queries':len(rr),'effects':{m:float(np.mean([r['effects'][m] for r in rr])) for m in rr[0]['effects']}})
            analysis={geo:{m:cluster([r['effects'][m] for r in cells if r['geometry']==geo],[r['speaker'] for r in cells if r['geometry']==geo]) for m in cells[0]['effects']} for geo in ('raw','unit')}
            dest=OUT/f'{label}_k{k}';write(dest/'queries.json',records);write(dest/'cells.json',cells);write(dest/'analysis.json',analysis);print(label,k,flush=True)
    write(OUT/'scoring_inputs.json',inputs)

def report():
    def fmt(z):return f"{z['speaker_mean']:+.3f} [{z['speaker_ci99'][0]:+.3f}, {z['speaker_ci99'][1]:+.3f}]"
    lines=['# 纯音频音素内DTW实例诊断','','先由每音频CMVN后的Wav2Lip 80维mel构建同phone DTW；共同路径双向映射，全部路径和离散query索引封存后才读取视觉/SyncNet评分。未用评分选择参数、路径或超参；无GPU、新TFG或波形迁移主实验。','','## 原支持固定正距离惩罚','','speaker等权均值与99%CI；负的DTW−MFA表示改善。这里只分析正距离，不是q margin或官方Sync-C。','','| 支持/k/geometry | MFA cross−self | DTW cross−self | oracle80 cross−self | DTW−MFA |','|---|---|---|---|---|']
    for label in ('primary','mfa_only'):
        for k in (3,2,4):
            z=read(OUT/f'{label}_k{k}/analysis.json')
            for geo in ('raw','unit'):
                lines.append(f'| {label}/{k}/{geo} | '+' | '.join(fmt(z[geo][m]) for m in ('mfa_gap','dtw_gap','oracle_gap','dtw_minus_mfa'))+' |')
    lines.extend(['','## 主支持k3双向','','| geometry/direction | DTW cross−self | DTW−MFA | 改变行比例 | 平均绝对位移ms | 回退比例 | 目标超出候选区间比例 |','|---|---|---|---:|---:|---:|---:|'])
    z=read(OUT/'primary_k3/analysis.json')
    for geo in ('raw','unit'):
        for d in ('Q1Q2','Q2Q1'):
            r=z[geo];lines.append(f"| {geo}/{d} | {fmt(r[d+'_dtw_gap'])} | {fmt(r[d+'_dtw_minus_mfa'])} | {r[d+'_changed']['speaker_mean']:.3f} | {r[d+'_abs_shift_ms']['speaker_mean']:.1f} | {r[d+'_fallback']['speaker_mean']:.3f} | {r[d+'_target_outside']['speaker_mean']:.3f} |")
    diagnostics=read(OUT/'mapping_diagnostics.json')['primary_k3']
    lines.extend(['',f"主685query/1370方向：{diagnostics['changed']}方向改变音频行，{diagnostics['fallback']}回退，{diagnostics['endpoint_clamped']}次路径端点钳制；{diagnostics['target_outside_candidate_range']}个目标超出离散有效候选范围，其中仅{diagnostics['target_outside_80ms_original_center']}个超出原MFA中心±80ms。其余可来自同phone/guard与离散网格范围，不能全称80ms限幅。逐项计数见mapping_diagnostics.json。",'',
                  '100对音频独立建图：87对Q1/Q2音素序列兼容，13对序列不兼容保留记录；主/敏感性既定query没有因此改变。'])
    lines.extend(['','## 纯音频操纵检查','','固定cal a1_001/Q1；每phone>=80ms前半时间映至前75%、后半映至后25%，waveform线性取样，时长/边界不变。这会改变局部频谱，不是音色保持合成。对两方向原25/50/75%相位报告已知映射误差；短phone可能小于40ms评分格点分辨率。','','| 支持 | 相位×方向数 | MFA时间MAE(ms) | DTW时间MAE(ms) | identity MAE(ms) | MFA索引MAE(frame) | DTW索引MAE(frame) |','|---|---:|---:|---:|---:|---:|---:|'])
    for name,r in read(OUT/'controls.json')['summary'].items():lines.append(f"| {name} | {r['n']} | "+' | '.join(f"{r[m]['mean']:.3f}" for m in ('mfa_error_ms','dtw_error_ms','identity_error_ms','mfa_index_error_frames','dtw_index_error_frames'))+' |')
    lines.extend(['','完整控制逐phone、方向、原/真值/恢复时间及索引误差在controls.json。所有控制/主结果均未用于调参。identity索引正确，但已知变形的总体恢复只略好，>=160ms仅2个phone/12个相位方向，误差反而变大；不能据此声称该DTW已准确恢复真实语音时序。','','## 本轮判断','','主k3合并raw/unit及MFA-only都存在独立于SyncNet选时的小幅改善；两方向各自99%CI未同时确认改善，k4没有保持改善。DTW后仍有正残余，但控制没有建立可靠时间恢复，且更有利的局部oracle可使残余区间跨0。因此时序归因仍未识别，残余不能直接叫非时序实例表征效应，也不能给native优势贡献比例。','','## 解释限制','','复用TFG输入mel，但映射不读取SyncNet/视觉；不是完全模型无关，也不是声学或物理事件真值。结果只诊断固定视频的正距离，不是native优势分解；oracle是刻意优待交叉的乐观参照。MFA原query集合不变，±80ms范围、同phone/guard限制、回退和端点处理全部预定。','','## 验证','','独立复核：`'+json.dumps(read(OUT/'validation.json'),ensure_ascii=False)+'`。','','protocol.json、audio_mapping_seal.json、query_mapping_seal.json、scoring_start.json记录先封存后评分；audio-only阶段Python审计钩子禁止读取评分数组和oracle结果。路径/音频mel在maps/、mel/，映射索引在indices/，各支持评分在primary或mfa_only_k2/3/4。源码快照和所有产物哈希见code/、provenance.json。'])
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z=read(OUT/'primary_k3/analysis.json')['raw'];names=('mfa_gap','dtw_gap','oracle_gap');v=[z[m] for m in names];means=np.array([r['speaker_mean'] for r in v]);ci=np.array([r['speaker_ci99'] for r in v]).T
    fig,ax=plt.subplots(figsize=(5.5,3.3),layout='constrained');ax.errorbar(range(3),means,yerr=np.vstack((means-ci[0],ci[1]-means)),fmt='o',capsize=5);ax.set_xticks(range(3),('MFA','Audio-only DTW','Optimistic oracle'));ax.axhline(0,color='gray',lw=1);ax.set_ylabel('Cross minus self distance, 99% CI');fig.savefig(OUT/'summary.png',dpi=180);fig.savefig(OUT/'summary.svg');plt.close(fig)
    (OUT/'code').mkdir(exist_ok=True)
    for f in (ROOT/'scripts/experiments').glob('*tts_instance_audio_dtw*.py'):shutil.copyfile(f,OUT/'code'/f.name)
    write(OUT/'provenance.json',{str(f):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='provenance.json'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('run','report'));globals()[p.parse_args().stage]()
