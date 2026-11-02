"""Static paired source-level entropy figure for the frozen B gate."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(run):
    p=json.loads((run/'B_gate_and_calibration.json').read_text());a=np.array(p['source_codebook_means']['L06']);b=np.array(p['source_codebook_means']['L17'])
    fig,axes=plt.subplots(1,2,figsize=(8,4),constrained_layout=True)
    for ax,name,cols in zip(axes,['coarse','fine'],[[0],list(range(1,16))]):
        x=a[:,cols].mean(1);y=b[:,cols].mean(1)
        for u,v in zip(x,y):ax.plot([0,1],[u,v],color='#9ba9bb',alpha=.7,linewidth=1,marker='o',markersize=3)
        ax.plot([0,1],[x.mean(),y.mean()],color='#173d72',linewidth=3,marker='o',markersize=7,label='10-source mean')
        g=p['gates'][name];ax.set_title(f'{name.capitalize()} content entropy\nL17-L06 {g["mean_L17_minus_L06_nat"]:+.3f} nat')
        ax.set_xticks([0,1],['L06','L17']);ax.set_ylabel('Effective entropy (nat)');ax.set_xlim(-.2,1.2);ax.grid(axis='y',alpha=.2)
        ax.legend(loc='upper right',fontsize=7)
        ax.text(.02,.02,f'97.5% CI [{g["CI97_5"][0]:+.3f}, {g["CI97_5"][1]:+.3f}]\nEntropy gate: {"PASS" if g["pass"] else "STOP"}',transform=ax.transAxes,fontsize=8)
    fig.suptitle('Common complete histories; source-balanced, pre-existing eligibility')
    fig.savefig(run/'B_source_entropy.png',dpi=180);fig.savefig(run/'B_source_entropy.svg');plt.close(fig)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
