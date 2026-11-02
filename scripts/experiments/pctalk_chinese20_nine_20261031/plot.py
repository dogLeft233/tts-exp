"""Export source-level mean differences and marginal bootstrap intervals."""
from pathlib import Path
import argparse, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    data=json.loads((a.run/'summary.json').read_text(encoding='utf8'))
    names=list(data['subsets']['all20']);fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
    for ax,name in zip(axes.flat,names):
        ax.axvline(0,color='#9298a0',linestyle='--',linewidth=1,zorder=0)
        labels=[]
        for y,subset,color,label in [(1,'all20','#156a9b','All eligible'),(0,'same_reference12','#c57c17','Same reference')]:
            row=data['subsets'][subset][name];n=row['n_sources'];labels.append((y,f'{label} (n={n})'))
            if n:
                mean=row['delta'];lo,hi=row['ci95']
                ax.plot([lo,hi],[y,y],color=color,linewidth=2.5)
                ax.scatter([mean],[y],color=color,s=38,zorder=3)
        ax.set_yticks([x[0] for x in labels],[x[1] for x in labels],fontsize=8)
        ax.set_ylim(-.55,1.55);ax.set_title(name,fontsize=12,loc='left',fontweight='bold')
        ax.set_xlabel('TTS - natural',fontsize=9);ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3));ax.tick_params(axis='x',labelsize=8)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('PC-Talk: source-paired TTS vs natural audio\nRMS -24 dBFS | fixed face and seed | mean difference with marginal 95% bootstrap CI',fontsize=14)
    fig.savefig(a.run/'paired_differences.png',dpi=180);fig.savefig(a.run/'paired_differences.pdf')
    plt.close(fig)

if __name__=='__main__':main()
