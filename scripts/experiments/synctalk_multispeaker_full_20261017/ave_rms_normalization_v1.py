"""Whole-recording scalar RMS intervention, before unchanged official AVE mel."""
import numpy as np
TARGET_RMS=0.1
def normalize(wave,target=TARGET_RMS):
    x=np.asarray(wave)
    if x.ndim!=1 or x.size==0 or not np.isfinite(x).all():raise ValueError('finite nonempty mono waveform required')
    if not np.isfinite(target) or target<=0:raise ValueError('positive finite target required')
    x=x.astype(np.float64)
    rms=float(np.sqrt(np.mean(x*x,dtype=np.float64)))
    if not np.isfinite(rms) or rms<=0:raise ValueError('silent/degenerate RMS rejected; no epsilon fallback')
    gain=float(target/rms);y=x*gain
    if not np.isfinite(y).all():raise ValueError('normalization overflow')
    y32=y.astype(np.float32)
    return y32,dict(input_rms=rms,target_rms=float(target),normalization_gain=gain,
        input_peak=float(np.abs(x).max()),output_peak=float(np.abs(y32).max()),
        output_rms_float32=float(np.sqrt(np.mean(y32.astype(np.float64)**2))),
        mean_subtraction=False,clipping=False,limiter=False,epsilon=None,samples=len(x))
