"""Complete predeclared F0/voicing descriptions using frozen pyworld settings."""
import importlib.util,json,hashlib
from pathlib import Path
import numpy as np,soundfile as sf
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'scripts/experiments/tts_native_spectrum_generation_cross_20260926.py'
spec=importlib.util.spec_from_file_location('spectrum',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);OUT=m.OUT
rows=[];arrays={}
for row in m.read(OUT/'support.json'):
    for arm in ['N','T']:
        raw=None
        for c in m.COND:
            path=Path(row['arms'][arm][c]['waveform']);assert m.sha(path)==row['arms'][arm][c]['sha256'];x=sf.read(path,dtype='float64')[0];f0,info=m.cal.v1.f0_proxy(x);assert len(f0)>0
            if c=='raw':raw=f0
            assert len(f0)==len(raw)
            voiced=(f0>0)&(raw>0);cents=abs(1200*np.log2(f0[voiced]/raw[voiced]))
            rec={'id':row['id'],'split':row['split'],'arm':arm,'condition':c,'waveform_sha256':m.sha(path),**info,'f0_common_voiced_frames':int(voiced.sum()),'f0_common_voiced_median_abs_cents':float(np.median(cents)) if len(cents) else None,'f0_common_voiced_p95_abs_cents':float(np.percentile(cents,95)) if len(cents) else None,'f0_voicing_disagreement_fraction':float(np.mean((f0>0)!=(raw>0)))}
            rows.append(rec);arrays[row['id']+'__'+arm+'__'+c]=f0
    print('F0',row['id'],flush=True)
keys=['voiced_fraction','f0_common_voiced_median_abs_cents','f0_common_voiced_p95_abs_cents','f0_voicing_disagreement_fraction']
summary={s:{c:{k:m.cal.v1.summary_stats([r[k] for r in rows if r['split']==s and r['condition']==c]) for k in keys} for c in m.COND} for s in ['calibration','evaluation']}
np.savez_compressed(OUT/'f0_descriptive_arrays.npz',**arrays)
m.write(OUT/'f0_descriptive.json',{'rows':rows,'summary':summary,'code_sha256':m.sha(__file__),'frozen_function_source_sha256':m.sha(m.CAL/'code_snapshot/tts_acoustic_modulation_calibration.py'),'note':'Initial waveform CPU environment lacked pyworld; frozen F0 proxy now measured on exactly sealed saved PCM in project .venv. Descriptive only, no gates/selection/score changes.'})
