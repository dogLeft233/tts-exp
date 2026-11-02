"""Static figures for model-specific common-support results; no model calls."""
from pathlib import Path
import sys,json,numpy as np
BASE=Path('F:/Datasets/tts-exp/encoder_survey_20261018')
sys.path.insert(0,str(BASE/'plotdeps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 j=json.loads((BASE/'analysis/results.json').read_text(encoding='utf8'))
 rows=[r for r in j['results'] if r['domain']=='full_phone']
 ids=j['completed_models'];colors=plt.cm.tab20(np.linspace(0,1,len(ids)))
 fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
 for c,mid in zip(colors,ids):
  for ax,ep in zip(axes[:2],['TTS_RAW','TTS_FIXED']):
   a=sorted([r for r in rows if r['model']==mid and r['endpoint']==ep],key=lambda r:r['layer'])
   ax.plot([r['layer_fraction'] for r in a],[r.get('pairwise_cos_mean',np.nan) for r in a],'.-',color=c,label=mid)
  a=sorted([r for r in rows if r['model']==mid and r['endpoint']=='GAIN_P3'],key=lambda r:r['layer'])
  axes[2].semilogy([r['layer_fraction'] for r in a],[r['mean_chord_rms_native'] for r in a],'.-',color=c,label=mid)
 for ax,title in zip(axes,['TTS - natural: original levels','TTS - natural: matched RMS','Pure +3 dB: direction change size']):
  ax.set_title(title);ax.set_xlabel('Relative encoder depth (0 = input representation)');ax.grid(alpha=.2)
 axes[0].set_ylabel('Across-speaker mean direction cosine');axes[1].set_ylabel('Across-speaker mean direction cosine')
 for ax in axes[:2]:ax.axhline(0,color='gray',lw=.8);ax.set_ylim(-.15,1)
 axes[2].set_ylabel('RMS unit-vector chord (native frame support)')
 axes[2].axhline(1e-5,color='black',ls='--',lw=1,label='direction resolution floor')
 axes[2].legend(fontsize=8,bbox_to_anchor=(1.04,1),loc='upper left')
 fig.suptitle('Same CAL20 / 10 speakers; selected-layer common matched-phone support; descriptive geometry',fontsize=12)
 for ext in ['png','svg','pdf']:fig.savefig(BASE/'analysis'/('encoder_commonality.'+ext),dpi=160)
 plt.close(fig)
 # Final layer paired contrasts, with no best-layer/model selection.
 last=[r for r in rows if r['layer_fraction']==1 and r['endpoint'] in ['TTS_RAW','TTS_FIXED']]
 fig,ax=plt.subplots(figsize=(11,5),layout='constrained');x=np.arange(len(ids))
 for ep,off,col in [('TTS_RAW',-.15,'#777777'),('TTS_FIXED',.15,'#2166ac')]:
  y=[next(r for r in last if r['model']==mid and r['endpoint']==ep).get('pairwise_cos_mean',np.nan) for mid in ids]
  ax.bar(x+off,y,.3,label=ep,color=col)
 ax.set_xticks(x,ids,rotation=35,ha='right');ax.set_ylabel('Across-speaker mean direction cosine')
 ax.set_title('Final layers, same matched-phone support; 10 speaker units');ax.axhline(0,color='black',lw=.8);ax.legend()
 for ext in ['png','svg','pdf']:fig.savefig(BASE/'analysis'/('final_layer_commonality.'+ext),dpi=160)
 plt.close(fig)
if __name__=='__main__':main()
