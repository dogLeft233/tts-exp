"""CPU-only tables and scientific figures for the frozen TTS extension."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen_full_tts_extension_20261022_attempt01'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=json.loads((RUN/'analysis.json').read_text())
    t=json.loads((RUN/'tts/tts_manifest.json').read_text())
    out=RUN/'report_assets';out.mkdir(exist_ok=False)
    roles=['N','R','T1','T2','T3'];records=a['records']
    x=np.array([[r['policies']['common'][k]['C'] for k in roles] for r in records])
    three=np.column_stack((x[:,:2],x[:,2:].mean(1)))
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),constrained_layout=True)
    for row in three:axes[0].plot(range(3),row,color='#9ca3af',alpha=.65,lw=1,marker='o',ms=3)
    axes[0].plot(range(3),three.mean(0),color='#1d4ed8',lw=2.5,marker='o',label='Mean of 10 speakers')
    axes[0].set_xticks(range(3),['Natural N','Codec R','Full TTS T\n(mean of 3 seeds)'])
    axes[0].set_ylabel('Sync-C (common valid support)');axes[0].legend(fontsize=8)
    for i,key in enumerate(['T-N','T-R']):
        st=a['summary']['common']['C']['contrasts'][key]
        axes[1].scatter(np.full(10,i)+np.linspace(-.08,.08,10),st['values'],s=24,color='#64748b')
        lo,hi=st['CI97_5'];mean=st['mean']
        axes[1].errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color='#1d4ed8',capsize=6,lw=2.5)
    axes[1].axhline(0,color='black',lw=.8,ls='--');axes[1].set_xticks([0,1],['T - N','T - R'])
    axes[1].set_ylabel('Paired Sync-C difference')
    axes[1].set_title('Dots: speakers; blue: mean and 97.5% bootstrap CI',fontsize=9)
    fig.suptitle('Full self-ICL TTS extension: 10 exposed development speakers',fontsize=11)
    fig.savefig(out/'paired_sync_c.png',dpi=180);plt.close(fig)
    with (out/'all_policies_endpoints.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['sample_id','speaker_id','support_policy','condition','C','B','Dmin','D0','offset','query_count','lag_boundary'])
        for r in records:
            for policy,cells in r['policies'].items():
                for role,m in cells.items():writer.writerow([r['sample_id'],r['speaker_id'],policy,role,*[m[k] for k in ['C','B','Dmin','D0','offset','query_count','best_lag_at_boundary']]])
    diagnostics=[]
    for r in t['records']:
        diagnostics.append(dict(sample_id=r['sample_id'],seed_index=r['seed_index'],duration_s=r['duration_s'],
                                new_steps=r['new_steps'],eos=r['eos_in_talker_sequences'],
                                cut_rounding_difference=r['cut_rounding_difference_samples'],
                                **r['diagnostics']['official_vs_newcodes_only']))
    meta=dict(common_query_counts=[r['common_query_count'] for r in records],
              low_common_query_samples=[r['sample_id'] for r in records if r['low_common_query_warning']],
              eos_missing=[(r['sample_id'],r['seed_index']) for r in t['records'] if r['eos_in_talker_sequences'] is not True],
              tts_duration_range_s=[min(r['duration_s'] for r in t['records']),max(r['duration_s'] for r in t['records'])],
              context_R_exact_count=sum(r['context_vs_parent_R']['array_exact'] for r in t['context_controls']),
              context_R_mae_range=[min(r['context_vs_parent_R']['overlap_mae'] for r in t['context_controls']),max(r['context_vs_parent_R']['overlap_mae'] for r in t['context_controls'])],
              official_vs_new_only_exact_count=sum(d['array_exact'] for d in diagnostics),
              diagnostics=diagnostics)
    (out/'diagnostic_summary.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k!='diagnostics'},indent=2))


if __name__=='__main__':main()
