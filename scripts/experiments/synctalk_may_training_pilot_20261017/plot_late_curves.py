"""A read-only late-curve viewing aid; never selects a checkpoint."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,label


def main():
    gains=[-6,0,6];available={};steps=[]
    for g in gains:
        path=ROOT/'training'/label(g)/'validation_curve.json'
        rows=json.loads(path.read_text()) if path.exists() else []
        available[g]=rows;steps.append(str(max(r['step'] for r in rows)) if rows else 'absent')
    out=ROOT/'diagnostics';out.mkdir(exist_ok=True)
    destination=out/('late_val_'+'_'.join(steps)+'.png');assert not destination.exists()
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for j,g in enumerate(gains):
        for h,color in zip(gains,['#2471a3','#555555','#ca6f1e']):
            rows=[r for r in available[g] if r['test_gain_db']==h and r['step']>=20000]
            for i,key in enumerate(['MSE','mouth_bbox_MSE']):
                ax=axes[i,j]
                if rows:ax.plot([r['step'] for r in rows],[r[key] for r in rows],color=color,label=f'input {h:+d} dB',marker='.',ms=3)
                ax.set_xlim(20000,60000);ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
                ax.set_title(f'train {g:+d} dB: {key}',fontsize=10);ax.set_xlabel('Successful updates')
        if available[g]:axes[0,j].legend(fontsize=8)
    fig.suptitle('Late validation curves (linear scale); missing models/steps are not extrapolated',fontsize=11)
    fig.savefig(destination,dpi=180);plt.close(fig);print(destination,flush=True)


if __name__=='__main__':main()
