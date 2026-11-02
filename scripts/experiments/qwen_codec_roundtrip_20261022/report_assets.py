"""Standalone scientific figures and a not-yet-annotated listening packet."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import soundfile as sf
from scipy.signal import stft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen_codec_roundtrip_20261022_attempt01'
sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_codec_roundtrip_20261022.pipeline import envelope


def main():
    manifest=json.loads((RUN/'audio_manifest.json').read_text())
    by={(r['sample_id'],r['role']):r for r in manifest['records']}
    timing=json.loads((RUN/'timing/mfa_timing.json').read_text())
    events=list(csv.DictReader((RUN/'timing/phone_events.csv').open()))
    figures=RUN/'figures';figures.mkdir(exist_ok=False)
    for record in timing['records']:
        sid=record['sample_id'];ys={}
        for role in ('N','R'):
            ys[role],sr=sf.read(by[sid,role]['path'])
        fig,axs=plt.subplots(4,1,figsize=(12,9),layout='constrained')
        for role,color in [('N','#2467b2'),('R','#d66519')]:
            env=envelope(ys[role]);axs[0].plot(np.arange(len(env))*.01+.01,env,label=role,color=color,lw=1)
        axs[0].set(title=f'{sid}: original clock, no waveform alignment',ylabel='20 ms RMS')
        axs[0].legend(loc='upper right')
        for ax,role in zip(axs[1:3],('N','R')):
            f,t,z=stft(ys[role],fs=sr,nperseg=400,noverlap=240,nfft=512,boundary=None,padded=False)
            ax.pcolormesh(t,f/1000,20*np.log10(np.maximum(abs(z),1e-7)),shading='auto',vmin=-80,vmax=-15,cmap='magma')
            ax.set(ylabel=f'{role}: kHz',ylim=(0,8))
        er=[e for e in events if e['sample_id']==sid]
        for name in ('start','center','end'):
            axs[3].plot([(float(e['N_start'])+float(e['N_end']))/2 for e in er],
                        [float(e[f'{name}_delta_s'])*1000 for e in er],'.-',label=name,ms=3,lw=.7)
        axs[3].axhline(0,color='black',lw=.6);axs[3].legend(ncol=3)
        axs[3].set(ylabel='MFA R-N (ms)',xlabel='Time in natural recording (s)')
        for ax in axs:ax.set_xlim(0,max(len(y) for y in ys.values())/sr)
        fig.savefig(figures/f'{sid}.png',dpi=130);plt.close(fig)
    # Blinded audio aliases: mapping is outside packet, no labels auto-filled.
    packet=RUN/'timing/blind_annotation_packet';packet.mkdir(exist_ok=False)
    rng=np.random.default_rng(20261022)
    order=rng.permutation(len(manifest['records']))
    private=[]
    with (packet/'annotations_blank.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['alias','event_description','start_s','end_s','confidence','comments']);w.writeheader()
        for index,pos in enumerate(order,1):
            row=manifest['records'][int(pos)];alias=f'clip_{index:02d}.wav'
            shutil.copyfile(row['path'],packet/alias)
            private.append(dict(alias=alias,sample_id=row['sample_id'],role=row['role']))
            w.writerow(dict(alias=alias,event_description='speech_onset'))
            w.writerow(dict(alias=alias,event_description='speech_end'))
            for _ in range(3):w.writerow(dict(alias=alias,event_description='clear_boundary: describe audible event'))
    (RUN/'timing/blind_annotation_key.json').write_text(json.dumps(private,indent=2)+'\n')
    (packet/'README.txt').write_text('NOT ANNOTATED. No human judgment has been performed.\nListen to clips independently with hidden condition labels. Mark speech onset/end and up to three clearly identifiable consonant or pause boundaries. Describe each event so matching can be adjudicated after labeling. Use the original playback timeline; do not align clips. At least two independent annotators are needed to quantify annotation error. Do not supply the private key or MFA results to annotators.\n')
    print('FIGURES',len(list(figures.glob('*.png'))),'BLIND_AUDIO',len(private))


if __name__=='__main__':main()
