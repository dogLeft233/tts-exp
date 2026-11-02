"""Predeclared descriptive identity contrasts, exact scorer decomposition and finite-budget screen."""
import hashlib,itertools,json,math
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
IDS=['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers'];SEEDS=[42,2026]
METRICS=['sync_c','sync_d','distance_d0','background_median']
def summary(values):
    assert len(values)==4 and all(math.isfinite(x) for x in values)
    return dict(mean=math.fsum(values)/4,min=min(values),max=max(values),positive=sum(x>0 for x in values),negative=sum(x<0 for x in values),zero=sum(x==0 for x in values),identities=4)
def index(cells):
    result={(c['training_gain_db'],c['driver_gain_db'],c['evaluation_gain_db']):c for c in cells}
    assert len(cells)==len(result)==27 and set(result)==set(itertools.product([-6,0,6],repeat=3));return result
def paired_contrast(per_seed,a,b,support):
    rows=[]
    for identity in IDS:
        seeds=[]
        for seed in SEEDS:
            cells=per_seed[identity,seed]
            seeds.append(dict(seed=seed,values={m:cells[a][support][m]-cells[b][support][m] for m in METRICS}))
        rows.append(dict(identity=identity,seeds=seeds,values={m:math.fsum(s['values'][m] for s in seeds)/2 for m in METRICS}))
    return dict(a=a,b=b,support=support,per_identity=rows,equal_identity={m:summary([r['values'][m] for r in rows]) for m in METRICS})
def decomposition(cells,training,gain,support,metric):
    c=lambda d,e:cells[training,d,e][support][metric]
    total=c(gain,gain)-c(0,0)
    driven_video_scored_with_original_audio=c(gain,0)-c(0,0)
    scorer_on_driven_video=c(gain,gain)-c(gain,0)
    scorer_on_baseline_video=c(0,gain)-c(0,0)
    assert abs(total-driven_video_scored_with_original_audio-scorer_on_driven_video)<1e-10
    return dict(total=total,driver_at_original_score=driven_video_scored_with_original_audio,
        scorer_on_driven_video=scorer_on_driven_video,scorer_on_baseline_video=scorer_on_baseline_video,
        interaction=scorer_on_driven_video-scorer_on_baseline_video)
def plateau(previous,tail,tolerance=.02):
    assert len(previous)==len(tail)==3 and all(math.isfinite(v) and v>=0 for v in previous+tail)
    a=math.fsum(previous)/3;b=math.fsum(tail)/3
    if a==0:return dict(previous_mean=a,tail_mean=b,relative_change=None,plateau_screen_passed=b==0)
    relative=(b-a)/a
    return dict(previous_mean=a,tail_mean=b,relative_change=relative,plateau_screen_passed=abs(relative)<=tolerance)
def run():
    output=ROOT/'delivery';qa=json.loads((output/'independent_aggregate_qualification.json').read_text());assert qa['passed']
    contract_path=ROOT/'analysis_contract_v1.json';contract=json.loads(contract_path.read_text());assert contract['identity_count']==4
    tables=json.loads((output/'tables27.json').read_text());assert tables['passed']
    per_seed={(i,s):index(tables['per_identity_seed'][f'{i}/seed{s}']) for i,s in itertools.product(IDS,SEEDS)}
    contrasts=[]
    for support in ['guard20','full']:
        for axis in ['training','driver','scorer']:
            for gain in [-6,6]:
                a=(gain,0,0) if axis=='training' else ((0,gain,0) if axis=='driver' else (0,0,gain))
                contrasts.append(dict(axis=axis,gain_db=gain,**paired_contrast(per_seed,a,(0,0,0),support)))
    matched=[]
    for support,training,gain in itertools.product(['guard20','full'],[-6,0,6],[-6,6]):
        rows=[]
        for identity in IDS:
            seed_values=[dict(seed=s,metrics={m:decomposition(per_seed[identity,s],training,gain,support,m) for m in METRICS}) for s in SEEDS]
            values={m:{k:math.fsum(v['metrics'][m][k] for v in seed_values)/2 for k in seed_values[0]['metrics'][m]} for m in METRICS}
            rows.append(dict(identity=identity,seeds=seed_values,metrics=values))
        combined={m:{k:summary([r['metrics'][m][k] for r in rows]) for k in rows[0]['metrics'][m]} for m in METRICS}
        matched.append(dict(training_gain_db=training,gain_db=gain,support=support,per_identity=rows,equal_identity=combined))
    curves=json.loads((output/'learning_curves.json').read_text());assert len(curves)==408
    lookup={(r['identity'],r['seed'],r['training_gain_db'],r['epoch']):r for r in curves};screens=[]
    for identity,seed,gain,phase in itertools.product(IDS,SEEDS,[-6,0,6],['coarse','fine']):
        rule=contract['plateau_check'];before=rule[phase+'_previous_epochs'];tail=rule[phase+'_tail_epochs']
        def mse(epoch):return next(v['mean_mse'] for v in lookup[identity,seed,gain,epoch]['validation_by_driver_gain'] if v['gain_db']==0)
        screens.append(dict(identity=identity,seed=seed,training_gain_db=gain,phase=phase,previous_epochs=before,tail_epochs=tail,
            **plateau([mse(e) for e in before],[mse(e) for e in tail],rule['relative_absolute_change_max'])))
    result=dict(passed=True,scientific_unit='identity n=4 exploratory',seed_repeats=2,contrasts=contrasts,matched_decompositions=matched,
        plateau_screens=screens,plateau_pass_count=sum(r['plateau_screen_passed'] for r in screens),
        plateau_total=48,interpretation_limits=contract['interpretation_limits'],
        analysis_contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target=output/'factorial_interpretation.json';assert not target.exists();target.write_text(json.dumps(result,indent=2))
    lines=['# 训练、驱动与评分器的描述性对照','主支持为guard20，full为敏感性；以下单位为四个身份，两个seed先在身份内平均。保留逐身份/seed差值，不以clip或seed扩大n，也不作总体显著性主张。',
        '| 轴 | 增益 dB | ΔSync-C均值 | 身份范围 | 正/负/零 |','|---|---:|---:|---:|---:|']
    for row in contrasts:
        if row['support']!='guard20':continue
        c=row['equal_identity']['sync_c'];lines.append(f"| {row['axis']} | {row['gain_db']:+d} | {c['mean']:+.3f} | [{c['min']:+.3f}, {c['max']:+.3f}] | {c['positive']}/{c['negative']}/{c['zero']} |")
    lines+=['','评分器分解为精确加法恒等式：同档驱动与评分的总变化 = 原音频评分下的驱动变化 + 已改变视频上的评分器变化。另列固定基线视频的评分器项和两者交互，完整数值见JSON；不计算统一百分比。',
        f"固定0dB验证MSE尾部三epoch相对前一组三epoch的绝对变化≤2%：{result['plateau_pass_count']}/48个模型阶段通过描述性平台筛查。完整逐seed/增益结果及所有学习曲线保留；这不证明已经最优收敛或数据量充分。",'',*contract['interpretation_limits']]
    (output/'factorial_interpretation.md').write_text('\n\n'.join(lines)+'\n')
    files=[]
    for path in [target,output/'factorial_interpretation.md']:
        files.append(dict(path=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (output/'interpretation_receipt.json').write_text(json.dumps(dict(passed=True,files=files,analysis_contract_sha256=result['analysis_contract_sha256'],
        independent_aggregate_qa_sha256=hashlib.sha256((output/'independent_aggregate_qualification.json').read_bytes()).hexdigest(),
        original_generated_results_only=True,scope='descriptive analysis, not population or perceptual claims'),indent=2))
    print(json.dumps(dict(passed=True,contrasts=len(contrasts),matched_decompositions=len(matched),plateau_screens=48)))
if __name__=='__main__':run()
