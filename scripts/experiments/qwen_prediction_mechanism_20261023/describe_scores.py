"""Two prespecified source-centered entropy-gap/score-gap descriptions only."""
from pathlib import Path
import argparse,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import read,save,sha

def centered_correlation(x,y):
    x=x-x.mean(axis=1,keepdims=True);y=y-y.mean(axis=1,keepdims=True)
    xx=(x*x).sum(1);yy=(y*y).sum(1);xy=(x*y).sum(1)
    def corr(ix):
        denom=np.sqrt(xx[ix].sum(axis=-1)*yy[ix].sum(axis=-1))
        return np.divide(xy[ix].sum(axis=-1),denom,out=np.full_like(denom,np.nan),where=denom>0)
    rng=np.random.default_rng(2026102307);b=corr(rng.integers(0,len(x),(20000,len(x))))
    scalar=lambda v:float(v) if np.isfinite(v) else None
    valid=b[np.isfinite(b)]
    return {'Pearson_r':scalar(corr(np.arange(len(x)))),'source_cluster_CI95':[float(v) for v in np.quantile(valid,[.025,.975])] if len(valid) else None,'undefined_bootstrap_replicates':int(np.sum(~np.isfinite(b))),'leave_one_source_out':[scalar(corr(np.delete(np.arange(len(x)),i))) for i in range(len(x))]}

def main(run):
    p=read(run/'B_protocol.json');diagnostics=read(run/'B_history_diagnostics.json')['statistics']
    metric={(r['reader'],r['id']):r for r in diagnostics if 'reader' in r}
    old=ROOT/'runs/qwen17_level_comparison10_20261023_attempt02/comparison_index.json';index=read(old)['records'];scores={};pins={str(old):sha(old)}
    for r in index:
        if r['level']=='norm' and r['model'] in ['L06','L17']:
            assert sha(r['score_path'])==r['score_sha256'];s=read(r['score_path']);assert s['status']=='COMPLETE'
            scores[(r['sample_id'],r['model'],r['draw_index'])]=s['sync_c'];pins[r['score_path']]=r['score_sha256']
    samples=sorted({r['sample_id'] for r in p['rows']});y=np.array([[scores[(sid,'L17',d)]-scores[(sid,'L06',d)] for d in range(1,4)] for sid in samples]);result=[]
    for name,cbs in [('coarse',[0]),('fine',list(range(1,16)))]:
        x=np.array([[np.mean([metric[('L17',f'{sid}_L17_{d}')]['codebooks'][cb]['effective_content_entropy'] for cb in cbs])-np.mean([metric[('L06',f'{sid}_L06_{d}')]['codebooks'][cb]['effective_content_entropy'] for cb in cbs]) for d in range(1,4)] for sid in samples])
        result.append({'candidate':name,'predictor':'native-own-history entropy(L17)-entropy(L06), centered within source across three matched predetermined draws','outcome':'norm self-score Sync-C(L17)-Sync-C(L06), centered within source','source_ids':samples,'entropy_gaps':x.tolist(),'score_gaps':y.tolist(),**centered_correlation(x,y)})
    save(run/'B_entropy_score_description.json',{'status':'DESCRIPTIVE_ONLY','candidates':result,'independent_unit':'10 sources, never 30 independent draws','all_draws_retained':True,'no_training':True,'no_gate_selection':True,'pins':pins})

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
