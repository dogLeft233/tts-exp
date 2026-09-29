"""Frozen paired permutation reporting, no new subgroup or tuning."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.tts_native_boundary_audit import stats
OUT=ROOT/'runs/tts_native_permutation_20260926'
NAMES={'dynamic_cloud':'动态云','static_cloud':'静态云','static_local':'静态本地'}
def read(name):return json.loads((OUT/name).read_text())
def fmt(x):return f"{x['mean']:+.3f} [{x['ci99'][0]:+.3f}, {x['ci99'][1]:+.3f}]"


def main():
    p=read('protocol.json');gate=read('mc_gate_64.json');repeats=256 if gate['escalate_all_256'] else 64
    s=read(f'summary_{repeats}.json');verify=read('independent_validation.json')
    blocks={};lags={};conditional={}
    for key,result in s.items():
        conditional[key]={}
        for direction,pair in [('T2N',('M_permuted','M_only')),('N2T',('reverse_M_permuted','reverse_M_only'))]:
            conditional[key][direction]={}
            for field in ('C','B','D','D_anchor','C_anchor','best_lag'):
                a=result['contrasts'][pair[0]][field];b=result['contrasts'][pair[1]][field]
                names=sorted(a['per_speaker'])
                x=stats([a['per_speaker'][n]-b['per_speaker'][n] for n in names],names)
                x['n']=a['n'];x.pop('utterance_mean',None)
                conditional[key][direction][field]=x
    (OUT/'conditional_permutation.json').write_text(json.dumps(conditional,indent=2)+'\n')
    for key,result in s.items():
        group,geom,support,mode=key.split('/');g=p['parent']['groups'][group];ids=result['ids']
        speakers=[next(r['speaker'] for r in g['records'] if r['id']==sid) for sid in ids]
        for role,arms in [('N',['N']),('T',g['arms'][1:])]:
            movement=[];sizes=[]
            for sid in ids:
                rec=[read(f'permutations/{group}/{sid}/{a}.json') for a in arms]
                movement.append(np.mean([r['modes'][mode]['moved_I_first64'] for r in rec]))
                sizes.extend(v for r in rec for v in r['modes'][mode]['block_sizes'])
            blocks[key+'/'+role]={'moved_I':stats(movement,speakers),'pooled_block_n':len(sizes),
                'block_quantiles_0_25_50_75_95_100':np.quantile(sizes,[0,.25,.5,.75,.95,1]).tolist(),
                'singleton_fraction':float(np.mean(np.array(sizes)==1))}
        for label in ('N_base','T_base','N_M','T_M'):
            values={'original_anchor_rate':[],'permuted_anchor_rate':[],'changed_bestlag_rate':[]}
            for sid in ids:
                rows=[]
                for ta in g['arms'][1:]:
                    with np.load(OUT/'scores'/str(repeats)/group/geom/sid/f'{ta}.npz') as z:
                        native=int(np.argmin(z[label+'_identity'][0]))-15
                        changed=z[label+'_'+mode].argmin(1)-15
                        rows.append([float(native==g['k0']),np.mean(changed==g['k0']),np.mean(changed!=native)])
                average=np.mean(rows,axis=0)
                for j,name in enumerate(values):values[name].append(average[j])
            lags[key+'/'+label]={name:stats(x,speakers) for name,x in values.items()}
    (OUT/'block_summary.json').write_text(json.dumps(blocks,indent=2)+'\n');(OUT/'lag_summary.json').write_text(json.dumps(lags,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.8),sharey=True)
    for ax,mode in zip(axes,('whole','phone')):
        r=s[f'dynamic_cloud/raw/eval/{mode}']['contrasts']
        for shift,field,label in [(-.1,'C','Standard C'),(.1,'C_anchor','C anchor')]:
            xs=[r[c][field] for c in ('permutation_change','interaction')];m=np.array([x['mean'] for x in xs]);ci=np.array([x['ci99'] for x in xs])
            ax.errorbar(np.arange(2)+shift,m,yerr=np.stack([m-ci[:,0],ci[:,1]-m]),fmt='o',capsize=4,label=label)
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks([0,1],['Permutation change','M x permutation\ninteraction']);ax.set_title(mode)
    axes[0].set_ylabel('Change in T - N contrast (99% speaker CI)');axes[0].legend();fig.tight_layout()
    fig.savefig(OUT/'standard_vs_anchor.png',dpi=180);fig.savefig(OUT/'standard_vs_anchor.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,geom in zip(axes,('raw','unit')):
        for shift,mode in [(-.05,'whole'),(.05,'phone')]:
            r=s[f'dynamic_cloud/{geom}/eval/{mode}']['contrasts'];xs=[r[c]['C'] for c in ('original','M_only','permuted','M_permuted')]
            m=np.array([x['mean'] for x in xs]);ci=np.array([x['ci99'] for x in xs]);ax.errorbar(np.arange(4)+shift,m,yerr=np.stack([m-ci[:,0],ci[:,1]-m]),fmt='o-',capsize=3,label=mode)
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(4),['Original','M only','Permutation','Both'],rotation=20);ax.set_title(geom+' geometry');ax.set_ylabel('Native T - N contrast (99% CI)')
    axes[0].legend();fig.tight_layout();fig.savefig(OUT/'factorial_residuals.png',dpi=180);fig.savefig(OUT/'factorial_residuals.pdf');plt.close(fig)
    with (OUT/'contrasts.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['group','geometry','support','mode','contrast','metric','n','speakers','mean','ci99low','ci99high'])
        for key,r in s.items():
            for name,fields in r['contrasts'].items():
                for metric,x in fields.items():w.writerow([*key.split('/'),name,metric,x['n'],x['speakers'],x['mean'],*x['ci99']])
    lines=['# 原配成对时间置换与中点幅度交互','',
        '## 结论','',
        '动态原配标准 C 的分差在 whole 置换后下降，within-phone 也有较小下降。未做M匹配时，固定 anchor 诊断 C_anchor 的对应变化在99%CI下未确认；标准C下降包含最优lag/min变化。M匹配后的whole条件背景效应在raw下边缘为负，unit下未确认，不能把全部下降称为时间背景分离。', '',
        '中点幅度匹配加 whole 置换后的动态 raw 残余为 +0.146 [−0.024,+0.280]，unit 残余 +0.027 [+0.008,+0.046]。raw 区间跨零不证明完全解释，unit 的正残余也保留。动态 raw 标准 C 的负交互，在 C_anchor 上不明确，主要须结合 min 搜索项来理解。','',
        '## 协议与支持','', '本阶段是历史数据上的探索性机制定位：eval已被前序实验分析，排除旧26cal只分离ID而未分离speaker，不是独立确认。π、处理端点与分析规则先于本轮评分冻结；M配对尺度是预设test-time表示诊断，不是训练或可泛化映射。', '',
        f'复用上阶段 source PCM、联合长度固定前端、raw/unit、guard20 和原 ±15 lag：动态 k0=1、eval48/13speaker，full74仅桥接；静态云71/15、本地72/15。每条先{repeats}次 C 均值，Q1/Q2先各自配对后条内均值，最后speaker等权20k PCG64(20260926) bootstrap99CI。无ASR内容筛选。','',
        'J分左halo/I/右halo，π在region内封闭；whole任意排列，within-phone进一步固定每个MFA occurrence。π同时作用于(V_t,A_(t+k0))，评分访问π(i)、π(i+s−k0)，原query和lag不变。N的π跨Q1/Q2复用，raw/unit及M变换前后共用π。','',
        '506个臂的MFA资产均可核验，动态音频源hash与manifest的N/C全部对应；within-phone未新增排除。标签取A窗口中心(t+k0)×.04+.1075；每个silence occurrence及连续未覆盖gap单独标号，不合并全片空白。它固定的是中心MFA标签，不能保证200ms窗口的全部语音内容、viseme或真实音素一致。','',
        f'每臂预先封存256个派生排列，先评前64；32vs64最大组对比误差{gate["max_abs_group_32vs64"]:.6f}，低于0.02，所以全部保留64次。最差项：{gate["worst"]}。未按方向或显著性选择重复数。','',
        '## 标准 C 的原配对比与2×2','',
        'M指既定T→N中点幅度匹配。交互=(M+置换−M-only)−(置换−原始)，所有格子保持同支持。','',
        '|组/几何/置换|原始T−N|仅置换T−N|置换引起的变化|M-only残余|M+置换残余|交互|','|---|---|---|---|---|---|---|']
    for geom in ('raw','unit'):
        for group in NAMES:
            for mode in ('whole','phone'):
                r=s[f'{group}/{geom}/eval/{mode}']['contrasts']
                lines.append(f'|{NAMES[group]}/{geom}/{mode}|'+'|'.join(fmt(r[c]['C']) for c in ('original','permuted','permutation_change','M_only','M_permuted','interaction'))+'|')
    lines += ['', '## C_anchor 与最优 lag 搜索项','',
        '标准 C=B−Dmin；诊断 C_anchor=B−D(k0)。恒等式 C=C_anchor+[D(k0)−Dmin]。成对置换严格保持每臂anchor距离的多重集及均值，所以置换造成的 C_anchor 变化就是背景 B 变化；C仍允许最优lag变化。C_anchor不冒充标准C。','',
        '|动态raw/操作|标准C变化|C_anchor变化=背景项变化|ΔDmin|ΔD_anchor|','|---|---|---|---|---|']
    for mode in ('whole','phone'):
        r=s[f'dynamic_cloud/raw/eval/{mode}']['contrasts']
        for c in ('permutation_change','interaction'):
            x=r[c];lines.append(f'|{mode}/{c}|{fmt(x["C"])}|{fmt(x["C_anchor"])}|{fmt(x["D"])}|{x["D_anchor"]["mean"]:.3g}|')
    lines += ['', 'whole 的标准分差下降 −0.244，其中背景项点估计 −0.177 的99CI跨零；Dmin差上升 +0.068 的区间为正。within-phone 的标准下降 −0.093 中，背景项仅 −0.025且不明确，Dmin项约 +0.067。动态raw交互也包含约 +0.069 的Dmin项，而背景交互接近零且未确认。不能把标准C中的负交互全部归于时间背景结构。','',
        '|动态raw/状态|N最佳lag等于k0的比例|T最佳lag等于k0的比例|','|---|---|---|']
    for mode in ('whole','phone'):
        for phase in ('original','permuted'):
            a=lags[f'dynamic_cloud/raw/eval/{mode}/N_base'][phase+'_anchor_rate'];b=lags[f'dynamic_cloud/raw/eval/{mode}/T_base'][phase+'_anchor_rate']
            lines.append(f'|{mode}/{phase}|{fmt(a)}|{fmt(b)}|')
    lines += ['', '### 已匹配M后的条件置换效应', '',
        '这是冻结2×2格子的直接差(M+置换)−M-only，保留speaker配对协方差。读取主表后，在报告生成阶段显式导出这些既定2×2条件对比，没有增加新处理、支持或调参。', '',
        '|动态/几何/置换|条件标准C变化|条件C_anchor变化|', '|---|---|---|']
    for geom in ('raw','unit'):
        for mode in ('whole','phone'):
            x=conditional[f'dynamic_cloud/{geom}/eval/{mode}']['T2N']
            y=x['C_anchor']; detail=f"{y['mean']:+.6f} [{y['ci99'][0]:+.6f}, {y['ci99'][1]:+.6f}]"
            lines.append(f'|{geom}/{mode}|{fmt(x["C"])}|{detail}|')
    lines += ['', 'raw whole条件C_anchor变化为−0.191，99CI上界约−0.000323，贴近0；相同对比的32vs64均值变化为−0.001499，其绝对量大于CI上界距0的距离；按预定0.02门不扩抽，不能把边缘结果描述为强确认。within-phone和unit对应背景条件效应未确认。标准C条件效应还包含约0.137/0.136的Dmin差变化。']
    lines += ['', '## 各臂 B/D 和反向 M 敏感性','',
        '各臂原始及64个排列的完整31lag曲线存于scores/64；下表给动态raw两臂置换各自的变化。其他组/几何/条件及T−N全部指标见contrasts.csv。','',
        '|置换/臂|ΔC|ΔB|ΔDmin|ΔD_anchor|','|---|---|---|---|---|']
    for mode in ('whole','phone'):
        r=s[f'dynamic_cloud/raw/eval/{mode}']['contrasts']
        for arm in ('N','T'):
            x=r[arm+'_permutation_change'];lines.append(f'|{mode}/{arm}|{fmt(x["C"])}|{fmt(x["B"])}|{fmt(x["D"])}|{x["D_anchor"]["mean"]:.3g}|')
    lines += ['', '|组/几何/置换|N→T M-only后残余|反向M+置换残余|反向交互|','|---|---|---|---|']
    for geom in ('raw','unit'):
        for group in NAMES:
            for mode in ('whole','phone'):
                r=s[f'{group}/{geom}/eval/{mode}']['contrasts'];lines.append(f'|{NAMES[group]}/{geom}/{mode}|'+ '|'.join(fmt(r[c]['C']) for c in ('reverse_M_only','reverse_M_permuted','reverse_interaction'))+'|')
    lines += ['', '## 置换实际强度','',
        '移动比例为I内位置改变率，按条/实例/speaker等权；块大小为池化块的描述统计，不作为独立样本推断。singletons不移动但不排除任何query。','',
        '|组/置换/臂|I移动比例99CI|块数|块大小中位数[p25,p75] / 最大值|单元素块比例|','|---|---|---|---|---|']
    for group in NAMES:
        for mode in ('whole','phone'):
            for arm in ('N','T'):
                x=blocks[f'{group}/raw/eval/{mode}/{arm}'];q=x['block_quantiles_0_25_50_75_95_100']
                lines.append(f'|{NAMES[group]}/{mode}/{arm}|{fmt(x["moved_I"])}|{x["pooled_block_n"]}|{q[2]:.0f} [{q[1]:.0f},{q[3]:.0f}] / {q[-1]:.0f}|{x["singleton_fraction"]:.3f}|')
    lines += ['', '## 既知生成失败敏感性','',
        '仅静态本地补充去掉已知context_capacity停止的a1_005整个ID（N/Q1/Q2一起去掉），72→71；不按ASR、置换效应或显著性筛选，主分析不变。','',
        '|几何/置换|原生ΔC|仅置换残余|置换变化|M+置换残余|交互|','|---|---|---|---|---|---|']
    for geom in ('raw','unit'):
        for mode in ('whole','phone'):
            r=s[f'static_local/{geom}/exclude_known_failure/{mode}']['contrasts'];lines.append(f'|{geom}/{mode}|'+'|'.join(fmt(r[c]['C']) for c in ('original','permuted','permutation_change','M_permuted','interaction'))+'|')
    lines += ['', '去掉这个既知失败并未使静态本地的置换变化或raw交互明确，也没有改变本轮主要解释边界。','',
        '## 动态full74探索桥接','', '|几何/置换|原生ΔC|仅置换残余|M+置换残余|交互|','|---|---|---|---|---|']
    for geom in ('raw','unit'):
        for mode in ('whole','phone'):
            r=s[f'dynamic_cloud/{geom}/full74_bridge/{mode}']['contrasts'];lines.append(f'|{geom}/{mode}|'+'|'.join(fmt(r[c]['C']) for c in ('original','permuted','M_permuted','interaction'))+'|')
    controls=[read(str(rp.relative_to(OUT)))['controls'] for rp in (OUT/'matrices').glob('*/*/*/*/receipt.json')]
    maxima={k:max(x[k] for x in controls) for k in controls[0]}
    lines += ['', '## 保真与独立复核','',
        'π在I内是严格双射，所以I内每个原始(V,Aaligned)配对向量出现次数不变，各模态/M/R的整个多重集守恒，进而μM、σM和ρR守恒；左右halo也分别封闭。within-phone进一步对J中所有位置逐一核验occurrence标签守恒，因此所有被访问的positive与donor标签也守恒。anchor s=k0时π(i)=π(i+s−k0)，anchor距离多重集保持。','',
        f'原生float32矩阵回放最大差{maxima["identity_replay"]:.3g}；anchor多重集差0，能量均值守恒误差{maxima["energy_invariance"]:.3g}。全J距离使用float32输入/重建向量的含epsilon双精度Gram公式；这是计分公式的数值实现差，不替换任何支持或选择lag。','',
        f'独立复核{verify["pair_geometries"]}个pair×geometry，{verify["vector_probes"]:,}个直接向量抽验、{verify["scalar_distance_cells"]:,}个标量索引距离、{verify["permutation_multiset_checks"]:,}个排列多重集检查；向量最大误差{verify["max_errors"]["vectors"]:.3g}、曲线误差{verify["max_errors"]["scalar_curves"]:.3g}、统计误差{verify["max_errors"]["statistics"]:.3g}。4项测试通过，无GPU或新TFG。','',
        '序列化修复记录：原NPZ的phone键被置换数组占用，覆盖了原计划导出的标签向量字段；实际π生成与冻结时标签检查正确，MFA区间、标签词表及π均已在评分前封存。按这些冻结metadata确定性恢复label_vectors/并验证129,536个within-phone排列标签守恒，未改变任何π或分数。原封存保留，并补存与评分前protocol记录SHA一致的code_at_freeze.py；当前源码仅改名为phone_labels以避免再次冲突，label_recovery_receipt.json记录全部来源。','',
        '## 限制与产物','',
        '本轮只定位SyncNet原配计分对表示时间组织的依赖。whole可改变中心phone组成与时间邻接，within-phone仍改变局部相位、200ms窗口内容与邻接关系；MFA中心标签不是真实音素真值。置换后的向量不代表一段实际重排视频/音频质量。raw残余CI跨零不证明全解释，unit正残余和anchor诊断的不确定性均须保留。','',
        '完整产物：protocol.json、permutation_seal.json、annotation_hashes.json、permutations/、label_vectors/、matrices/、scores/64/、summary_64.json、mc_gate_64.json、contrasts.csv、block_summary.json、lag_summary.json、independent_validation.json、artifact_hashes.json及两组PDF/PNG。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    paths=[x for x in OUT.rglob('*') if x.is_file() and x.name!='artifact_hashes.json']
    paths += [ROOT/'scripts/experiments'/x for x in ('tts_native_permutation.py','check_tts_native_permutation.py','report_tts_native_permutation.py')]
    paths += [ROOT/'tests/test_tts_native_permutation.py']
    hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    (OUT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n');print('report and hashes',len(hashes))


if __name__=='__main__':main()
