"""Independent CPU reconstruction of speaker clusters, all resamples and Holm9."""
import json,argparse
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);root=p.parse_args().run
    summary=json.loads((root/'summary.json').read_text());rows=json.loads((root/'scoring_inputs.json').read_text())['rows']
    pairs=json.loads((root/'paired_scores.json').read_text());count=0;max_error=0
    for name,group in summary['subsets'].items():
        pvalues={}
        for metric,saved in group.items():
            planned={}
            for r in rows:
                if r['arm']!='N' or not r['paired_available']:continue
                if 'strict' in name and not r['strict_quality']:continue
                if 'new_only' in name and r['pilot_seen']:continue
                planned.setdefault(r['speaker'],[]).append(r['source'])
            actual={r['source']:r for r in pairs[metric]};speaker=[];pair_count=0
            for sp,sources in sorted(planned.items()):
                if any(s not in actual for s in sources):continue
                n=sum(actual[s]['N'] for s in sources)/len(sources);t=sum(actual[s]['T'] for s in sources)/len(sources)
                speaker.append((n,t));pair_count+=len(sources)
            assert len(speaker)==saved['n_speakers'] and pair_count==saved['n_pairs']
            if not speaker:pvalues[metric]=1.;continue
            n,t=np.array(speaker).T;d=t-n;rng=np.random.default_rng(20261031)
            ci=np.quantile(d[rng.integers(0,len(d),size=(10000,len(d)))].mean(axis=1),[.025,.975])
            exceed=0
            for _ in range(50):
                signs=rng.integers(0,2,size=(2000,len(d)))*2-1
                exceed+=np.count_nonzero(abs((signs*d).sum(1)/len(d))>=abs(d.mean())-1e-15)
            prob=(int(exceed)+1)/100001;pvalues[metric]=prob
            errors=[abs(float(n.mean())-saved['N_mean']),abs(float(t.mean())-saved['T_mean']),abs(float(d.mean())-saved['delta']),
                float(abs(ci-np.array(saved['ci95'])).max()),abs(prob-saved['signflip_p'])]
            assert max(errors)<1e-9,(name,metric,errors)
            max_error=max(max_error,max(errors));count+=1
        previous=0
        for rank,m in enumerate(sorted(pvalues,key=pvalues.get)):
            q=min(1.,max(previous,pvalues[m]*(len(pvalues)-rank)));previous=q
            assert abs(q-group[m]['holm9_p'])<1e-12
    result=dict(status='PASS',metric_strata=count,max_abs_error=max_error,
        scope='Independent source-to-speaker means, complete-speaker eligibility, pair counts,10k speaker bootstrap/100k signflip/Holm9')
    (root/'independent_statistics_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
