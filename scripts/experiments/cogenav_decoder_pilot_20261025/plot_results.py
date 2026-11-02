"""Plot comparable unweighted teacher error and fixed-monitor reconstruction error."""
import argparse
import json
from pathlib import Path


def read_curves(root):
    root=Path(root)
    trace=[json.loads(line) for line in (root/'train_metrics.jsonl').read_text().splitlines()]
    validation=[]
    baseline=json.loads((root/'untrained_monitor.json').read_text())
    validation.append(dict(global_step=0,stage='untrained',logmel=baseline['logmel']))
    for path in root.glob('validation_*_*.json'):
        _,stage,step=path.stem.split('_')
        value=json.loads(path.read_text())
        global_step=int(step)+(2000 if stage=='joint' else 0)
        validation.append(dict(global_step=global_step,stage=stage,logmel=value['logmel']))
    validation.sort(key=lambda r:r['global_step'])
    return trace,validation


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);trace,validation=read_curves(root)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),constrained_layout=True)
    for stage,color in [('warmup','#286BA6'),('joint','#D66B28')]:
        subset=[r for r in trace if r['stage']==stage]
        axes[0].plot([r['global_step'] for r in subset],[r['teacher_mse'] for r in subset],
                     color=color,lw=1.3,label=stage)
    axes[0].set(ylabel='Unweighted teacher-feature MSE',xlabel='Optimizer updates',title='Teacher reconstruction')
    axes[0].legend(frameon=False)
    axes[1].plot([r['global_step'] for r in validation],[r['logmel'] for r in validation],
                 color='#286BA6',marker='o',ms=3.5,lw=1.5)
    axes[1].set(ylabel='Mean log-Mel L1 (lower is better)',xlabel='Optimizer updates',title='Fixed 16-clip validation monitor')
    for ax in axes:
        ax.axvline(2000,ls='--',lw=1,color='#888888')
        ax.grid(alpha=.18)
    fig.suptitle('Frozen CoGenAV → adapted pretrained HiFi-GAN; no adversarial fine-tuning',fontsize=12)
    destination=root/'figures';destination.mkdir(exist_ok=True)
    fig.savefig(destination/'reconstruction_learning_curves.png',dpi=180)
    plt.close(fig)
    return destination/'reconstruction_learning_curves.png'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args();print(plot(args.root))
