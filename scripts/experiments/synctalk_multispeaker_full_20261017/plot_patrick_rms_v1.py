import sys,json,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,'E:/Documents/tts-exp/.codex/scratch/report25_plotdeps_attempt01')
P=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_v1/score_27_v1')
D=P/'delivery_v1'
summary=json.loads((P/'all27_preview.json').read_text())
indexed={(c['training_gain_db'],c['driver_gain_db'],c['evaluation_gain_db']):c for c in summary['cells']}
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(11,7.5),layout='constrained')
for row,metric in enumerate(['sync_c','sync_d']):
 values=[c['mean_guard20'][metric] for c in summary['cells']]
 for col,score in enumerate([-6,0,6]):
  mat=np.array([[indexed[(train,driver,score)]['mean_guard20'][metric] for driver in [-6,0,6]] for train in [-6,0,6]])
  ax=axes[row,col];im=ax.imshow(mat,vmin=min(values),vmax=max(values),cmap='viridis' if metric=='sync_c' else 'viridis_r')
  ax.set(xticks=range(3),xticklabels=['-6','0','+6'],yticks=range(3),yticklabels=['-6','0','+6'],xlabel='Driver gain (dB)',ylabel='Training gain (dB)',title=f"{'Sync-C (higher)' if row==0 else 'Sync-D (lower)'} | scorer {score:+d} dB")
  for i,j in itertools.product(range(3),repeat=2):ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',color='white' if sum(w*z for w,z in zip([.2126,.7152,.0722],im.cmap(im.norm(mat[i,j]))[:3]))<.5 else 'black',fontsize=12,bbox=dict(facecolor='gray',alpha=.25,edgecolor='none'))
 fig.colorbar(im,ax=axes[row,:],shrink=.75)
fig.suptitle('Patrick Leahy | RMS-AVE | seed 42 | coarse epoch 10 EMA\n6 fixed test clips, guard20 | descriptive preview',fontsize=13)
fig.savefig(D/'patrick_27_grid.png',dpi=180);plt.close(fig)
