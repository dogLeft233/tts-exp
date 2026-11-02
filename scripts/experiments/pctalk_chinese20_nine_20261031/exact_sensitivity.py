"""Read-only exact sign-flip sensitivity for a borderline Monte Carlo decision."""
from pathlib import Path
import json
import sys
import numpy as np

def main():
    root=Path(sys.argv[1]);summary=json.loads((root/'summary.json').read_text())
    pairs=json.loads((root/'paired_scores.json').read_text());out={}
    for subset,metrics in summary['subsets'].items():
        values={}
        for metric,saved in metrics.items():
            ps=[p for p in pairs[metric] if subset=='all20' or p['strict_reference']==(subset=='same_reference12')]
            if not ps:
                values[metric]=dict(n=0,exact_p=1.);continue
            d=np.array([p['T']-p['N'] for p in ps]);n=len(d);hits=0
            threshold=abs(d.mean())-1e-15
            for start in range(0,1<<n,32768):
                index=np.arange(start,min(start+32768,1<<n),dtype=np.uint32)
                signs=((index[:,None]>>np.arange(n,dtype=np.uint32))&1).astype(float)*2-1
                hits+=np.count_nonzero(abs((signs*d).mean(1))>=threshold)
            values[metric]=dict(n=n,exact_p=int(hits)/(1<<n),sign_patterns=1<<n)
        ordered=sorted(values,key=lambda m:values[m]['exact_p']);previous=0.
        for rank,metric in enumerate(ordered):
            q=min(1.,max(previous,values[metric]['exact_p']*(9-rank)));previous=q
            values[metric].update(exact_holm9_p=q,primary_mc_holm9_p=metrics[metric]['holm9_p'],
                decision_changed=(q<.05)!=(metrics[metric]['holm9_p']<.05))
        out[subset]=values
    result=dict(status='PASS',scope='Posthoc CPU sensitivity only; archived100k Monte Carlo primary unchanged. Exact enumeration of all sign patterns per source support, including nine hypotheses per stratum.',subsets=out)
    (root/'exact_signflip_sensitivity.json').write_text(json.dumps(result,indent=2))
    for subset,metrics in out.items():
        print(subset)
        for metric,s in metrics.items():print(metric,s['primary_mc_holm9_p'],s['exact_holm9_p'],'CHANGED' if s['decision_changed'] else '')

if __name__=='__main__':main()
