"""Scientific nine-panel paired effect figure; percentage points for K5/P0."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import RUN
p=json.loads((RUN/'summary.json').read_text())['subsets']['all20']
fig,axes=plt.subplots(3,3,figsize=(11,8))
for ax,(name,r) in zip(axes.flat,p.items()):
    scale=100 if name in ['MTDVocaLiST','VocaLiST','Synchformer','SparseSync'] else 1
    ax.axvline(0,color='gray',lw=.8)
    if r['n_sources']:
        d=r['delta']*scale;lo,hi=[x*scale for x in r['ci95']]
        ax.plot([lo,hi],[0,0],color='#245a81',lw=2);ax.scatter([d],[0],color='#245a81')
        ax.set_title(f"{name} (n={r['n_sources']})\nHolm9 p={r['holm9_p']:.4g}",fontsize=10)
    else:ax.set_title(name+' unmeasurable')
    ax.set_yticks([]);ax.set_ylim(-1,1);ax.set_xlabel('TTS - natural'+(' (percentage points)' if scale==100 else ''))
fig.suptitle('TalkLikeYou HDTF20: speaker-averaged paired effects\nMarginal 95% bootstrap CI; one TTS draw / segment, two segments / speaker',fontsize=12)
fig.tight_layout(rect=(0,0,1,.93))
fig.savefig(RUN/'paired_differences.png',dpi=180);fig.savefig(RUN/'paired_differences.pdf');plt.close(fig)
