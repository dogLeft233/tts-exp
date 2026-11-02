"""Presentation-only precision correction; preserves the original B plot."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(run):
    p=json.loads((run/'B_gate_and_calibration.json').read_text());a=np.array(p['source_codebook_means']['L06']);b=np.array(p['source_codebook_means']['L17'])
    fig,axes=plt.subplots(1,2,figsize=(8,4))
    fig.subplots_adjust(left=.09,right=.985,bottom=.14,top=.76,wspace=.28)
    for ax,name,cols in zip(axes,['coarse','fine'],[[0],list(range(1,16))]):
        x=a[:,cols].mean(1);y=b[:,cols].mean(1);g=p['gates'][name]
        for u,v in zip(x,y):ax.plot([0,1],[u,v],color='#9ba9bb',alpha=.7,linewidth=1,marker='o',markersize=3)
        ax.plot([0,1],[x.mean(),y.mean()],color='#173d72',linewidth=3,marker='o',markersize=7,label='10-source mean')
        ax.set_title(f'{name.capitalize()} content entropy\nL17-L06 {g["mean_L17_minus_L06_nat"]:+.6f} nat',fontsize=11)
        ax.set_xticks([0,1],['L06','L17']);ax.set_ylabel('Effective entropy (nat)');ax.set_xlim(-.2,1.2);ax.grid(axis='y',alpha=.2);ax.legend(loc='upper right',fontsize=7)
        label=f'97.5% CI [{g["CI97_5"][0]:+.6f}, {g["CI97_5"][1]:+.6f}]\nEntropy gate: {"PASS" if g["pass"] else "STOP"}'
        if not g['pass']:label+=' (|mean| < 0.100000 nat)'
        ax.text(.02,.02,label,transform=ax.transAxes,fontsize=7)
    fig.text(.5,.97,'Common complete histories; source-balanced, pre-existing eligibility',ha='center',va='top',fontsize=11)
    for suffix in ['png','svg']:
        out=run/('B_source_entropy_precise_v2.'+suffix)
        if out.exists():raise FileExistsError(out)
        fig.savefig(out,dpi=180)
    plt.close(fig)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
