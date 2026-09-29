"""Report every frozen endpoint; no result-driven selection or sample changes."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.tts_native_boundary_audit import stats
OUT=ROOT/'runs/tts_native_holdout_confirmation_20260926'
def read(p):return json.loads(Path(p).read_text())
def fmt(s):
    if s is None:return '不可用'
    ci=s.get('ci99');return f"{s['mean']:+.3f}"+(f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else '')
def main():
    p=read(OUT/'protocol.json');native=read(OUT/'native_summary.json');mech=read(OUT/'mechanism_summary.json');gate=read(OUT/'calibration_transfer_gate.json');generation=read(OUT/'generation_distribution.json');valid=read(OUT/'independent_validation.json');support=read(OUT/'support.json');assert valid['passed']
    eos=set(generation['all_Q_EOS_ids']);speaker={r['id']:r['speaker'] for r in p['rows']};sensitivity={}
    for geom in ('raw','unit'):
        ids=[r['id'] for r in support if r['eligible'] and r['id'] in eos];vv={}
        for sid in ids:
            q=[]
            for arm in ('Q1','Q2'):
                z=read(OUT/'scores/pairs'/geom/sid/(arm+'.json'))['k_results']['3'];s=z['scores'];t=z['permuted'];q.append({'original':s['T_base']['C']-s['N_base']['C'],'T2N_effect':s['T_M']['C']-s['T_base']['C'],'T2N_residual':s['T_M']['C']-s['N_base']['C'],'N2T_effect':s['N_M']['C']-s['N_base']['C'],'N2T_residual':s['T_base']['C']-s['N_M']['C'],'whole_gap':t['T_base']['C']-t['N_base']['C'],'T2N_joint':t['T_M']['C']-t['N_base']['C'],'N2T_joint':t['T_base']['C']-t['N_M']['C']})
            vv[sid]={name:np.mean([x[name] for x in q]) for name in q[0]}
        sensitivity[geom]={'ids':ids,'contrasts':{key:stats([x[key] for x in vv.values()],[speaker[s] for s in vv]) for key in next(iter(vv.values()))}}
    (OUT/'mechanism_eos_sensitivity.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    raw=mech['raw/k3']['contrasts'];mainci=raw['original']['C']['ci99'];direction='明确为正' if mainci[0]>0 else '明确为负' if mainci[1]<0 else '99%区间跨零'
    lines=['# 独立于本轮完整机制开发/校准集的前瞻固定协议复现','',f"本批固定40个AISHELL-1 canonical speaker、每人2句，原配本地strict Q1/Q2均值−N的guard20 ΔC为 **{fmt(raw['original']['C'])}**，{direction}。以下完整报告预设的原生、幅度与时间组织端点；不把表示干预解释为实际嘴型改善。",'', '## 来源独立性与生成条件','', '评分前只读审计A PASS：80条/40 speaker与本轮dynamic74/static100/旧cal26的canonical speaker不重叠，PCM hash零碰撞。B（全项目历史）仍UNRESOLVED：旧数值77/90及deleted/remote覆盖缺口保留；A不覆盖B，不称全项目绝对未见或模型预训练未见。source image3固定，因此不是新脸确认。v1/v2审计、范围裁决与释放核验均在评分前封存。','', '本地strict faster_qwen3 0.6B ICL，actual bf16；每条自然参考与同一人工语料text同时用于target/ref_text；与历史ASR text和云提供者条件不同，不冒称历史云端复现。先显式完成lazy CUDA graph预热，再逐call固定Q1=42/Q2=43，max_new_tokens2048；不使用ASR/MFA、换seed、短token救样或按输出筛选。25fps Wav2Lip、原固定框、FFV1无损视频、own canonical16k PCM与旧SyncNet JPEG前端；联合L=min(F,floor(S/640))−5，在重建lag padding前截取两模态。','', '## 完成、分母与校准迁移','']
    complete=sum(r['eligible'] for r in support);sp=len({r['speaker'] for r in support if r['eligible']});lines += [f"160个Q生成全部完成，159 EOS、1 context_capacity。容量停止为BAC009S0040W0121/Q2，157.68秒，完整保留且进入主分母，没有静默截短。固定首条seed42 replay的音频/codec hash一致。guard20共同支持为 **{complete}/80条、{sp}/40 speaker**；全部逐条状态见support.json和generation_distribution.json。",'',f"旧26cal固定k0=3；新臂raw guard20最佳lag中位数为 {gate['arm_best_lag_medians']}，迁移门 **{'PASS' if gate['passed'] else 'FAIL：校准迁移不足'}**。门要求臂中位数差≤1且距3≤1；不根据新结果改k或筛样。k2/k4仅诊断。",'', 'Q1/Q2先分别取每条分数，再条内平均，再speaker内句子平均，speaker等权；20,000次PCG64(20260926) speaker bootstrap，99%CI。主分母保留生成异常；EOS子集仅为预设敏感性。','', '## 原生官方距离端点','', '|支持|策略|n/speaker|ΔC|ΔB|ΔD|','|---|---|---:|---|---|---|']
    for s in ('common_guard20','maximum','common_guard20_all_Q_EOS'):
        for policy,r in native[s].items():lines.append(f"|{s}|{policy}|{r['n']}/{r['speakers']}|{fmt(r['contrasts']['C'])}|{fmt(r['contrasts']['B'])}|{fmt(r['contrasts']['D'])}|")
    lines+=['','B为31个lag平均距离的中位数，D为最小平均距离，C=B−D。valid删除每lag越界pair并独立归一化；guard15/20保留所有lag共同query。敏感性改变评分支持，不代表真实无声区/音素组成得到控制。','', '## 双向中点动态幅度匹配','', 'M=(V+A对齐)/2，R=(A对齐−V)/2；在I=20:L−20估计每clip均值和动态RMS尺度。T→N将T的M幅度乘sigmaN/sigmaT，反向对N乘其倒数，R保持；保持原音频索引与±15搜索。N/Q配对尺度匹配是预设test-time表示干预，不是训练模型样本外预测或可部署归一化。unit先逐行L2归一化原特征，再独立估计尺度；处理后不再归一化，其单位不等于官方Sync-C。','', '|几何/k|alpha T→N|T→N的T ΔC|处理后Q−N|N→T的N ΔC|处理后Q−N|','|---|---|---|---|---|---|']
    for key,r in mech.items():
        c=r['contrasts'];lines.append(f"|{key}|{fmt(r['coefficients']['alpha_T2N'])}|{fmt(c['T2N_effect']['C'])}|{fmt(c['T2N_residual']['C'])}|{fmt(c['N2T_effect']['C'])}|{fmt(c['N2T_residual']['C'])}|")
    lines+=['','主k3的完整C/B/D/D_anchor/C_anchor、各臂sigma/rho和系数见mechanism_summary.json。anchor向量差理论不变，而跨lag最小D和best lag仍可能变化；不能把ΔD(min)当作真实嘴型误差。','', '## Whole paired permutation × M-only 2×2','', '固定256重复，J左halo/I/右halo各区独立置换，两个模态按同π保持anchor配对。两臂都做成对置换；raw/unit/M前后共用π，同N的π在Q1/Q2配对中复用。所有索引在第一份距离矩阵之前封存。每重复先计C，再重复平均、Q平均和speaker平均，不对平均曲线冒充平均C。','', '|几何|对比|C（99%CI）|C_anchor（99%CI）|B|D|','|---|---|---|---|---|---|']
    for geom in ('raw','unit'):
        c=mech[geom+'/k3']['contrasts']
        for label in ('original','whole_gap','whole_effect','T2N_residual','T2N_joint','T2N_interaction','N2T_residual','N2T_joint','N2T_interaction','N_whole_effect','T_whole_effect'):
            r=c[label];lines.append(f"|{geom}|{label}|{fmt(r['C'])}|{fmt(r['C_anchor'])}|{fmt(r['B'])}|{fmt(r['D'])}|")
    lines+=['','C_anchor=B−D(k3)。interaction=joint_gap−M_gap−permuted_gap+original_gap，不解释为生理贡献比例。置换衡量固定表示上的时间组织依赖，不是实际重排音视频质量；本批不做within-phone、MFA或新source图。','', '## 客观停止类型敏感性','', '|几何|仅双Q EOS n|原生ΔC|T→N效应|T→N残余|N→T效应|N→T残余|whole gap|','|---|---:|---|---|---|---|---|---|']
    for geom,r in sensitivity.items():
        c=r['contrasts'];lines.append(f"|{geom}|{len(r['ids'])}|{fmt(c['original'])}|{fmt(c['T2N_effect'])}|{fmt(c['T2N_residual'])}|{fmt(c['N2T_effect'])}|{fmt(c['N2T_residual'])}|{fmt(c['whole_gap'])}|")
    lines+=['','此表按生成器客观EOS状态划分，不使用ASR、内容判断或分数；主分母与主结论不据此删除容量停止记录。','', '## 独立复核与产物','',f"独立NumPy向量距离/变换/统计复算通过：{valid['native_cells']} native cells、{valid['pairs']} pair×geometry，{valid['independent_direct_distance_count']:,}个M相关直接距离；所有256置换区间/anchor支持核对，并完整复算固定第0/255次曲线。最大误差：`{valid['errors']}`。3项synthetic契约测试通过；模型特征重复对照见feature_repeat_controls.json。",'', '协议与主数据：protocol.json、原inventory manifest；评分前seal：score_lock.json、permutation_seal.json、support.json；来源与范围：leakage_audit_v1/v2.json、scope_decision_receipt.json、score_release_review.json；逐cell/pair：features/、scores/；汇总：native_summary.json、mechanism_summary.json、mechanism_eos_sensitivity.json；验证：independent_validation.json；GPU进程/租约释放记录：gpu_release.json。','', '工程记录：第一次使用wav2lip环境因缺python_speech_features在任何特征/评分前退出，改用已有syncnet完整环境，未安装修改环境；相同模型/固定前端及实际版本hash均留档。生成使用与旧显式预热恢复实验相同的运行时。','']
    (OUT/'report.md').write_text('\n'.join(lines))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['original','T2N_residual','N2T_residual','whole_gap','T2N_joint','N2T_joint']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,geom in zip(axes,('raw','unit')):
        rows=[mech[geom+'/k3']['contrasts'][name]['C'] for name in names];mean=np.array([r['mean'] for r in rows]);ci=np.array([r['ci99'] for r in rows]);y=np.arange(len(rows));ax.errorbar(mean,y,xerr=np.stack([mean-ci[:,0],ci[:,1]-mean]),fmt='o',capsize=3);ax.axvline(0,color='gray',lw=1);ax.set_yticks(y,labels=names);ax.invert_yaxis();ax.set_title(geom+' geometry: Q mean − N');ax.set_xlabel('Speaker-equal gap (99% bootstrap CI)')
    fig.tight_layout();fig.savefig(OUT/'confirmation_effects.png',dpi=180);fig.savefig(OUT/'confirmation_effects.pdf');plt.close(fig)
    print('report written',OUT/'report.md')
if __name__=='__main__':main()
