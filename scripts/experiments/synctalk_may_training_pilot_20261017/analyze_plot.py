"""Descriptive one-identity pilot analysis; no population inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT,label,save_json,sha

GAINS=[-6,0,6]
COLORS={-6:'#2471a3',0:'#555555',6:'#ca6f1e'}


def plateau(curve,gain):
    rows=[r for r in curve if r['test_gain_db']==gain]
    early=[r['MSE'] for r in rows if r['step'] in [40000,42000,44000,46000,48000]]
    late=[r['MSE'] for r in rows if r['step'] in [52000,54000,56000,58000,60000]]
    assert len(early)==len(late)==5
    first=float(np.median(early));last=float(np.median(late));change=(last-first)/first
    status='plateau_screen' if abs(change)<=.02 else ('worsening' if change>.02 else 'still_improving')
    return {'early_median_MSE':first,'late_median_MSE':last,'relative_change':change,'screen':status}


def main():
    out=ROOT/'analysis';assert not out.exists();out.mkdir()
    score=json.loads((ROOT/'scores/scores.json').read_text())
    qa=json.loads((ROOT/'independent_results_qa.json').read_text());assert qa['passed']
    curves={g:json.loads((ROOT/'training'/label(g)/'validation_curve.json').read_text()) for g in GAINS}
    result={'scope':'one exposed English identity, one seed42, descriptive controlled first-stage pilot',
            'protocol_sha256':sha(ROOT/'protocol.json'),'validation_screen':[],
            'cells':[],'contrasts':{},'population_p_values':None,'data_sufficiency':'UNVERIFIED'}
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for g in GAINS:
        loss=[json.loads(s) for s in (ROOT/'training'/label(g)/'train_curve.jsonl').read_text().splitlines()]
        axes[0].plot([r['step'] for r in loss],[r['window_mean_loss'] for r in loss],color=COLORS[g],label=f'train {g:+d} dB',lw=1)
        for metric,axis in [('MSE',axes[1]),('mouth_bbox_MSE',axes[2])]:
            steps=sorted({r['step'] for r in curves[g]})
            values=[np.mean([r[metric] for r in curves[g] if r['step']==s]) for s in steps]
            axis.plot(steps,values,color=COLORS[g],label=f'train {g:+d} dB')
    for ax,title in zip(axes,['Official training loss (100-step windows)','Validation MSE, mean over 3 gains','Validation mouth-bbox MSE, mean over 3 gains']):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Successful optimizer updates');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for j,g in enumerate(GAINS):
        for h in GAINS:
            rows=[r for r in curves[g] if r['test_gain_db']==h]
            for metric,ax in [('MSE',axes[0,j]),('mouth_bbox_MSE',axes[1,j])]:
                ax.plot([r['step'] for r in rows],[r[metric] for r in rows],color=COLORS[h],label=f'input {h:+d} dB')
                ax.set_yscale('log');ax.grid(alpha=.2);ax.set_xlabel('Updates');ax.legend(fontsize=8)
            result['validation_screen'].append({'training_gain_db':g,'input_gain_db':h,**plateau(curves[g],h)})
        axes[0,j].set_title(f'Train {g:+d} dB: validation MSE');axes[1,j].set_title(f'Train {g:+d} dB: mouth-bbox MSE')
    fig.savefig(out/'validation_by_input_gain.png',dpi=180);plt.close(fig)
    for g in GAINS:
        for h in GAINS:
            cell=next(c for c in score['cells'] if c['training_gain_db']==g and c['test_gain_db']==h)
            path=ROOT/'renders'/('train_'+label(g))/('test_'+label(h))
            render=json.loads((path/'render_receipt.json').read_text());val=json.loads((path/'full_validation.json').read_text())
            result['cells'].append({'training_gain_db':g,'test_gain_db':h,'guard20':cell['mean_guard20'],
                                    'full':cell['mean_full'],'test_reconstruction':render['summary'],
                                    'validation_reconstruction':val['summary'],
                                    'clip_offsets':[r['guard20']['av_offset'] for r in cell['clips']]})
    fields=['sync_c','sync_d','distance_d0','background_median']
    grids={k:np.array([[next(c for c in result['cells'] if c['training_gain_db']==g and c['test_gain_db']==h)['guard20'][k] for h in GAINS] for g in GAINS]) for k in fields}
    fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
    for ax,key,title in zip(axes,fields,['Sync-C (higher)','Sync-Dmin (lower)','Distance at 0 lag (lower)','Median background distance']):
        im=ax.imshow(grids[key],cmap='viridis');ax.set_xticks(range(3),['-6','0','+6']);ax.set_yticks(range(3),['-6','0','+6'])
        ax.set_xlabel('Test gain (dB)');ax.set_ylabel('Training gain (dB)');ax.set_title(title,fontsize=10)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{grids[key][i,j]:.3f}',ha='center',va='center',color='white',bbox={'facecolor':'black','alpha':.35,'edgecolor':'none'})
        fig.colorbar(im,ax=ax,shrink=.65)
    fig.savefig(out/'test_matrix.png',dpi=180);plt.close(fig)
    for key,x in grids.items():
        result['contrasts'][key]={
            'training_gain_at_test0_vs_train0':{'-6':float(x[0,1]-x[1,1]),'+6':float(x[2,1]-x[1,1])},
            'test_gain_within_each_model_vs_test0':{str(g):{'-6':float(x[i,0]-x[i,1]),'+6':float(x[i,2]-x[i,1])} for i,g in enumerate(GAINS)},
            'interaction_vs_train0':{str(g):{str(h):float((x[i,j]-x[i,1])-(x[1,j]-x[1,1])) for j,h in enumerate(GAINS) if h!=0} for i,g in enumerate(GAINS) if g!=0}}
    save_json(out/'analysis.json',result)
    lines=['# May gain training pilot','',
        'One exposed English identity, seed42, 4150 training frames (166 s), 550 validation frames, 1100 test frames.',
        'Each model: same initial tensors and actual rays; 60000 successful FP32 updates, 4096 rays/update; fixed final EMA. Original natural PCM is used for every score. No test-based checkpoint selection.',
        '', '## Learning-curve screen','',
        'Frozen heuristic: median MSE at 40/42/44/46/48k versus 52/54/56/58/60k. Absolute relative change <=2% is a plateau screen, not proof of convergence or sufficient data.','',
        '| Training dB | Validation input dB | Relative MSE change | Screen |','|---:|---:|---:|---|']
    for r in result['validation_screen']:lines.append(f"| {r['training_gain_db']:+d} | {r['input_gain_db']:+d} | {r['relative_change']:+.2%} | {r['screen']} |")
    lines+=['','## Fixed test grid','',
        'Four fixed 10-second clips; equal clip mean; guard20 primary. Sync-C = B - Dmin. Shared ground-truth ROI.','',
        '| Train dB | Test dB | Sync-C | Dmin | D0 | B | Full-val mouth MSE |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['cells']:
        m=r['guard20'];v=r['validation_reconstruction']
        lines.append(f"| {r['training_gain_db']:+d} | {r['test_gain_db']:+d} | {m['sync_c']:.3f} | {m['sync_d']:.3f} | {m['distance_d0']:.3f} | {m['background_median']:.3f} | {v['mouth_bbox_MSE']:.6f} |")
    lines+=['','The controls and raw matrices passed independent verification. This pilot estimates conditional responses for May only. Four clips and many frames do not create independent speakers or seeds. No population p-values or best-gain claim is made. Data-amount sufficiency remains UNVERIFIED; SyncNet improvement alone does not establish perceptual or geometric synchronization. This is a first-stage pilot with a reduced ray budget, not a complete official two-stage reproduction.','',
        '![Learning curves](learning_curves.png)','![Validation by input](validation_by_input_gain.png)','![Test matrix](test_matrix.png)','']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'analysis':str(out),'validation_screens':result['validation_screen']}),flush=True)


if __name__=='__main__':main()
