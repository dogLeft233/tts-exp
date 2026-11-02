"""Independent CPU readback using an explicit framing/FFT frontend."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
if '--deps' in sys.argv:
    sys.path.insert(0,sys.argv[sys.argv.index('--deps')+1])
import numpy as np
import librosa
import soundfile as sf
from scipy.fft import rfft
from scipy.signal import get_window


def main(a):
    root=a.output
    results=json.loads((root/'results.json').read_text())
    data_manifest=json.loads((root/'sources/data_manifest.json').read_text())
    pins=[]
    for row in results['input_pins']:
        actual=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
        assert actual==row['sha256']==row['replay_sha256']
        pins.append(actual)
    original,sr=sf.read(results['source'],dtype='float64',always_2d=True)
    assert sr==16000 and np.array_equal(original[:,0],original[:,1])
    original=original[:,0]
    checks=[]
    for split in data_manifest['splits']:
        name=split['name'];start=split['source_start_frame']*640
        x=original[start:start+split['audio_samples']]
        loaded={}
        for db,label in [(0,'gain_0'),(-6,'gain_m6'),(6,'gain_p6')]:
            wave=np.load(root/f'arrays/{name}_{label}_decoded.npy')
            assert np.array_equal(wave,(x*10**(db/20)).astype(np.float32))
            pre=np.empty(len(wave),dtype=np.float64)
            pre[0]=wave[0];pre[1:]=wave[1:].astype(float)-.97*wave[:-1].astype(float)
            centered=np.pad(pre,(400,400))
            framed=np.lib.stride_tricks.sliding_window_view(centered,800)[::200]
            magnitude=np.abs(rfft(framed*get_window('hann',800,fftbins=True),axis=1)).T
            basis=librosa.filters.mel(sr=16000,n_fft=800,n_mels=80,fmin=55,fmax=7600)
            mel=basis@magnitude
            saved=np.load(root/f'arrays/{name}_{label}_mel_amplitude.npy')
            max_error=float(abs(mel-saved).max());assert np.allclose(mel,saved,rtol=1e-11,atol=2e-13)
            mapped=np.clip(.08*(20*np.log10(np.maximum(mel,1e-5))-20)+4,-4,4)
            f=np.load(root/f'arrays/{name}_{label}_mapped_mel.npy')
            assert np.allclose(mapped,f,rtol=0,atol=2e-10)
            loaded[db]=(mel,mapped)
            row=next(r for r in results['waveform'] if r['split']==name and r['gain_db']==db
                     and r['path']=='historical_DOUBLE')
            n=int(((x!=0)&(wave==0)).sum());assert n==row['decoded_new_zero_points']
            checks.append(dict(split=name,gain_db=db,explicit_fft_mel_max_error=max_error,
                               independent_decoded_new_zero_points=n))
        m0,f0=loaded[0];m1,f1=loaded[-6]
        independent=(f0>-4)&(f1==-4)
        saved=np.load(root/f'arrays/{name}_newly_lower_clipped.npy')
        assert np.array_equal(independent,saved)
        summary=next(s for s in results['spectral'] if s['split']==name)
        assert int(independent.sum())==summary['newly_lower_clipped']
        assert int(((m0>1e-5)&(m1<=1e-5)).sum())==summary['newly_log_floored']
        frequencies=list(csv.DictReader((root/'spectral_by_frequency.csv').open(encoding='utf-8-sig')))
        assert sum(int(r['newly_lower_clipped']) for r in frequencies if r['split']==name)==int(independent.sum())
        times=list(csv.DictReader((root/'spectral_by_time.csv').open(encoding='utf-8-sig')))
        assert sum(int(r['newly_lower_clipped']) for r in times if r['split']==name)==int(independent.sum())
        native=(x.astype(np.float32)-x.astype(np.float32).mean())/np.sqrt(x.astype(np.float32).var()+1e-7)
        assert np.array_equal((native*.5)*2,native)
        for row in [r for r in results['postnorm'] if r['split']==name]:
            scaled=(native*np.float32(row['scale'])).astype(np.float32)
            assert hashlib.sha256(scaled.tobytes()).hexdigest()==row['historical_input_sha256']
    qa=dict(passed=True,input_files_fresh_sha=len(pins),explicit_fft_checks=checks,
            all_new_floor_masks_and_frequency_time_csv_recomputed=True,
            all_14_historical_postnorm_inputs_exact=True,new_gpu_calls=0,training_updates=0,
            note='Explicit Hann framing + scipy rFFT; no imports from audit.py or official functions')
    (root/'independent_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps(qa,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--deps',type=Path)
    main(p.parse_args())
