"""Post-hoc S0002 final-character exclusion; diagnostic, not a causal control."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/wjj/tts-audio/tts-exp/runs')
SRC=ROOT/'qwen_phrase_local_20261024_attempt02'
OUT=ROOT/'qwen_phrase_visual_review_20261024_attempt02'
a={x['id']:x for x in json.loads((SRC/'alignment.json').read_text())['audio']}
m=json.loads((SRC/'local_metrics.json').read_text());rows=[];bridge=[]
for x in m:
    if x['sample_id']!='BAC009S0002W0122' or x['phrase_index']!=1 or x['variant']!='nominal':
        continue
    D=np.load(SRC/('distance_'+x['id']+'_track0.npz'))['distance']
    k=x['fixed_best_column']
    full=D[x['valid_rows']].mean(axis=0)
    error=abs(float(np.median(full)-full[k])-x['contrast'])
    assert error<1e-10
    bridge.append(error)
    boundary=a[x['id']]['alignment'][0]['timestamp'][14][0]/1000
    # The prior parent audit proved all selected tracks are contiguous and start at frame zero.
    for shift in [-.04,0,.04]:
        cut=boundary+shift
        ix=[i for i in x['valid_rows'] if (i+5)*.04<=cut+1e-9
            and (i+k-15)*.04+.215-.0690625<=cut+1e-9]
        assert len(ix)>=5
        L=D[ix].mean(axis=0)
        rows.append({'id':x['id'],'cut_s':cut,'shift':shift,'n':len(ix),'rows':ix,
                     'contrast':float(np.median(L)-L[k]),'matched':float(L[k]),
                     'background':float(np.median(L))})
gaps={}
for sh in [-.04,0,.04]:
    by={x['id']:x for x in rows if x['shift']==sh}
    gaps[str(sh)]=[by['BAC009S0002W0122_L17_'+str(d)]['contrast']-
                  by['BAC009S0002W0122_L06_'+str(d)]['contrast'] for d in [1,2,3]]
result={'posthoc':True,'rule':'retain existing nominal valid windows, matched native video and matched audio RF end before given-text FA final gou onset; +/-40ms boundary. Same fixed whole-track lag and 31 lag means. Not silence removal, not causal, not additive contribution.',
        'rows':rows,'L17_minus_L06':gaps,'full_metric_bridge_max_error':max(bridge),
        'minimum_support':min(x['n'] for x in rows)}
(OUT/'S0002_exclude_final_gou_sensitivity_replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'gaps':gaps,'bridge_error':max(bridge),'min_windows':result['minimum_support']}))
