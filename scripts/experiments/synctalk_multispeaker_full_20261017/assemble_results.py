"""Equal-identity27 tables and fixed validation curves; no clip pseudoreplication."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
IDS=['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers'];SEEDS=[42,2026];GAINS=[-6,0,6]
METRICS=['sync_c','sync_d','distance_d0','background_median']

def aggregate(rows):
    result={k:float(np.mean([row[k] for row in rows])) for k in METRICS}
    offsets=[v for row in rows for v in row['offsets']];result.update(offsets=offsets,offset_mean=float(np.mean(offsets)))
    return result

def table(cells,support):
    lines=['| 训练 dB | 驱动 dB | 评价 dB | Sync-C | Dmin | D0 | B | offset均值 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cell in cells:
        values=cell[support];gains=[cell[k] for k in ['training_gain_db','driver_gain_db','evaluation_gain_db']]
        lines.append('| '+' | '.join([f'{g:+d}' for g in gains]+[f"{values[k]:.3f}" for k in METRICS]+[f"{values['offset_mean']:.3f}"] )+' |')
    return '\n'.join(lines)

def run():
    output=ROOT/'delivery';output.mkdir(exist_ok=False);per_seed={};per_identity={};curves=[]
    for identity in IDS:
        for seed in SEEDS:
            folder=ROOT/'scores'/identity/f'seed{seed}'
            qualification=json.loads((folder/'independent_score_qualification.json').read_text());assert qualification['passed'] and qualification['clip_matrices']==162
            raw=json.loads((folder/'scores.json').read_text());assert raw['passed'] and len(raw['cells'])==27
            per_seed[identity,seed]=[{**{k:c[k] for k in ['training_gain_db','driver_gain_db','evaluation_gain_db']},
                **{s:aggregate([c['mean_'+s]]) for s in ['guard20','full']}} for c in raw['cells']]
            for phase,epochs in [('coarse',range(1,11)),('fine',range(11,18))]:
                assert json.loads((ROOT/'blocks'/identity/f'seed{seed}'/f'{phase}_qualification.json').read_text())['passed']
                for epoch in epochs:
                    for gain in GAINS:
                        path=ROOT/'training'/identity/f'seed{seed}'/f'gain_{gain:+d}'/phase/f'ep{epoch:04d}/attempt01/epoch_receipt.json'
                        receipt=json.loads(path.read_text());assert receipt['passed']
                        curves.append(dict(identity=identity,seed=seed,training_gain_db=gain,epoch=epoch,phase=phase,
                            training_original_total_loss=receipt['post_validation_stats']['loss'][-1],
                            validation_by_driver_gain=receipt['validation']['rows'],amp_skips=receipt['epoch_amp_skips'],
                            successful_optimizer_updates=receipt['epoch_successful_updates']))
        per_identity[identity]=[]
        for index in range(27):
            base=per_seed[identity,SEEDS[0]][index]
            assert all([per_seed[identity,s][index][k] for k in ['training_gain_db','driver_gain_db','evaluation_gain_db']]==
                [base[k] for k in ['training_gain_db','driver_gain_db','evaluation_gain_db']] for s in SEEDS)
            per_identity[identity].append({**{k:base[k] for k in ['training_gain_db','driver_gain_db','evaluation_gain_db']},
                **{support:aggregate([per_seed[identity,s][index][support] for s in SEEDS]) for support in ['guard20','full']}})
    combined=[]
    for index in range(27):
        base=per_identity[IDS[0]][index]
        combined.append({**{k:base[k] for k in ['training_gain_db','driver_gain_db','evaluation_gain_db']},
            **{support:aggregate([per_identity[i][index][support] for i in IDS]) for support in ['guard20','full']}})
    result=dict(passed=True,identities=IDS,seeds=SEEDS,scientific_unit='identity n=4 exploratory descriptive',
        hierarchy='six fixed clips -> seed -> two seeds within identity -> four equal identities',
        equal_identity=combined,per_identity=per_identity,per_identity_seed={f'{i}/seed{s}':v for (i,s),v in per_seed.items()})
    (output/'tables27.json').write_text(json.dumps(result,indent=2));(output/'learning_curves.json').write_text(json.dumps(curves,indent=2))
    report=['# SyncTalk 四新身份完整训练三轴电平实验',
        '四身份各两个seed，24个官方head+lip模型。固定最终EMA、60000+42000 attempted steps，完整65536 rays/原FP16；每个配对块实际optimizer更新预算已独立验收。',
        '每clip先计算，六clip等权、两个seed等权、四身份等权。n=4探索性结果，不把片段/帧当新增独立样本，不将Sync-C变化称为感知质量改善。offset均值为六clip离散lag的描述性汇总，全部原offset保存在JSON。',
        '数据为可追溯HDTF自然处理镜像；跨录制几何采用ADR039冻结共享训练参考点适配。Patrick测试源482平方上采样512；各录制原始满幅/静音/时钟指标按实际采用的数据版本manifest保留；上游预训练暴露未知。',
        '原配对几何资格失败与ADR040/041唯一候选尝试全部保留，实际采用的数据版本见准入及dataset_revisions合同。该数据适配发生在TFG更新前；Cathy候选405秒原料，实际监督与其他身份相同。结论限于通过冻结几何门的配对，不代表随机录制总体。',
        '评价0为原始PCM16；±6为float64原PCM数值单位增益，区别于报告51的PCM16舍入。source-owned ROI、SyncNet FP32、guard20主/full次；完整距离矩阵及独立QA随附。',
        '## 四身份等权 guard20',table(combined,'guard20'),'## 四身份等权 full',table(combined,'full')]
    for identity in IDS:
        report+=['## '+identity+' 两seed平均',table(per_identity[identity],'guard20')]
        for seed in SEEDS:report+=['### '+identity+f' seed{seed}',table(per_seed[identity,seed],'guard20')]
    (output/'report.md').write_text('\n\n'.join(report)+'\n')
    plot(curves,output)
    files=[dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(output.iterdir()) if p.is_file()]
    (output/'analysis_receipt.json').write_text(json.dumps(dict(passed=True,identity_seed_tables=8,identity_tables=4,combined_cells=27,files=files),indent=2))
    print(json.dumps(dict(passed=True,identity_seed_tables=8,identity_tables=4,combined_cells=27)))

def plot(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in ['training','validation']:
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for identity,ax in zip(IDS,axes.flat):
            for gain,color in zip(GAINS,['#4477aa','#228833','#cc6677']):
                for seed,style in zip(SEEDS,['-','--']):
                    selected=sorted([r for r in rows if r['identity']==identity and r['seed']==seed and r['training_gain_db']==gain],key=lambda r:r['epoch'])
                    assert len(selected)==17
                    ys=[r['training_original_total_loss'] if kind=='training' else next(v['mean_mse'] for v in r['validation_by_driver_gain'] if v['gain_db']==0) for r in selected]
                    ax.plot(range(1,18),ys,color=color,linestyle=style,label=f'{gain:+d} dB seed{seed}')
            ax.axvline(10.5,color='gray',linestyle=':');ax.set_title(identity);ax.set_xlabel('Full epoch');ax.set_yscale('log')
            ax.set_ylabel('Original total training loss' if kind=='training' else 'Fixed 0 dB driver validation MSE');ax.legend(fontsize=7)
        fig.savefig(output/f'{kind}_curves.png',dpi=180);fig.savefig(output/f'{kind}_curves.pdf');plt.close(fig)
if __name__=='__main__':run()
