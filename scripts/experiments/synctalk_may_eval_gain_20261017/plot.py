"""Static scientific summary of a completed scoring-only run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    analysis=json.loads((root/'analysis.json').read_text())
    gains=[-6,0,6]
    lookup={(r['training_gain_db'],r['driver_gain_db'],r['evaluation_gain_db']):r for r in analysis['cells']}
    matrices=[np.array([[lookup[t,d,e]['guard20']['sync_c'] for d in gains] for t in gains]) for e in gains]
    lo=min(x.min() for x in matrices);hi=max(x.max() for x in matrices)
    fig,axes=plt.subplots(1,3,figsize=(11.5,4),layout='constrained')
    for axis,e,matrix in zip(axes,gains,matrices):
        im=axis.imshow(matrix,vmin=lo,vmax=hi,cmap='viridis')
        for i in range(3):
            for j in range(3):axis.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]<(lo+hi)/2 else 'black')
        axis.set(xticks=range(3),xticklabels=['-6','0','+6'],yticks=range(3),yticklabels=['-6','0','+6'],
            xlabel='Driver gain (dB)',ylabel='Training gain (dB)',title=f'Evaluation audio {e:+d} dB')
    fig.colorbar(im,ax=axes,label='Sync-C (guard20)',shrink=.8)
    fig.suptitle('May fixed videos: 27 cells, same four test clips')
    fig.savefig(root/'score_27_cells.png',dpi=170);plt.close(fig)
    differences=[matrices[0]-matrices[1],matrices[2]-matrices[1]]
    limit=max(np.abs(x).max() for x in differences)
    fig,axes=plt.subplots(1,2,figsize=(8.2,4),layout='constrained')
    for axis,e,matrix in zip(axes,[-6,6],differences):
        im=axis.imshow(matrix,vmin=-limit,vmax=limit,cmap='RdBu_r')
        for i in range(3):
            for j in range(3):axis.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center')
        axis.set(xticks=range(3),xticklabels=['-6','0','+6'],yticks=range(3),yticklabels=['-6','0','+6'],
            xlabel='Driver gain (dB)',ylabel='Training gain (dB)',title=f'Evaluation {e:+d} dB minus 0 dB')
    fig.colorbar(im,ax=axes,label='Fixed-video delta Sync-C',shrink=.8)
    fig.suptitle('Scorer-side response: visual embeddings held fixed')
    fig.savefig(root/'scorer_only_delta.png',dpi=170);plt.close(fig)
    rows=analysis['matched_decomposition'];x=np.arange(len(rows));width=.25
    fig,axis=plt.subplots(figsize=(10,4.7),layout='constrained')
    for i,(key,label,color) in enumerate([
        ('driver_at_original_eval','Driver change, evaluation kept at 0 dB','#377eb8'),
        ('scorer_at_changed_driver','Evaluation change, video held fixed','#ff7f00'),
        ('total','Both driver and evaluation changed','#4daf4a')]):
        y=[r['guard20']['sync_c'][key] for r in rows]
        bars=axis.bar(x+(i-1)*width,y,width,label=label,color=color)
        axis.bar_label(bars,labels=[f'{v:+.3f}' for v in y],fontsize=8,padding=3)
    axis.axhline(0,color='black',linewidth=.7)
    axis.set_xticks(x,[f'Train {r["training_gain_db"]:+d}\nDrive/eval {r["matched_gain_db"]:+d}' for r in rows])
    axis.set(ylabel='Delta Sync-C vs driver 0 / evaluation 0',title='Exact path decomposition within each trained model')
    axis.legend(fontsize=8,loc='best')
    axis.margins(y=.25)
    fig.savefig(root/'matched_decomposition.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
