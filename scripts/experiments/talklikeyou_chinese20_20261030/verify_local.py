"""Recompute delivered source means, bootstrap CI, sign-flip and Holm on CPU."""
from pathlib import Path
import argparse, json
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output/'evidence';summary=json.loads((root/'summary.json').read_text());pairs=json.loads((root/'paired_scores.json').read_text())
    checked=0;max_error=0.
    for name,metrics in summary['subsets'].items():
        ps={}
        for metric,saved in metrics.items():
            group=[r for r in pairs[metric] if name=='all20' or r['strict_reference']==(name=='same_reference12')]
            assert len(group)==saved['n_sources']
            if not group:ps[metric]=1.;continue
            n=np.array([r['N'] for r in group]);t=np.array([r['T'] for r in group]);d=t-n
            rng=np.random.default_rng(20261030)
            boot=np.array([d[i].mean() for i in rng.integers(0,len(d),size=(10000,len(d)))])
            ci=np.quantile(boot,[.025,.975]);exceed=0
            for _ in range(50):
                signs=2*rng.integers(0,2,size=(2000,len(d)))-1
                exceed+=np.count_nonzero(abs(np.dot(signs,d)/len(d))>=abs(d.mean())-1e-15)
            pval=(int(exceed)+1)/100001;ps[metric]=pval
            errors=[abs(saved['N_mean']-float(n.mean())),abs(saved['T_mean']-float(t.mean())),abs(saved['delta']-float(d.mean())),float(abs(ci-np.array(saved['ci95'])).max()),abs(pval-saved['signflip_p'])]
            assert max(errors)<1e-9,(name,metric,errors)
            assert saved['T_higher']==int(np.count_nonzero(d>0)) and saved['N_higher']==int(np.count_nonzero(d<0))
            max_error=max(max_error,max(errors));checked+=1
        previous=0
        for rank,metric in enumerate(sorted(ps,key=ps.get)):
            adjusted=min(1.,max(previous,ps[metric]*(len(ps)-rank)));previous=adjusted
            assert abs(adjusted-metrics[metric]['holm9_p'])<1e-12
    result=dict(status='PASS',checked_nonempty_metric_subsets=checked,max_abs_error=max_error,
        independent_calculations='source means, T-N,10000 bootstrap intervals,100000 sign-flip sensitivities,Holm9 and win counts',numpy=np.__version__)
    (a.output/'local_statistics_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
