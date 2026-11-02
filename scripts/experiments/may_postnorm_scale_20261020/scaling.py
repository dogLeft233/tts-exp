"""Only scale the float32 native standardized waveform; never normalize again."""
def postnorm_scale(native,scale):
    import numpy as np
    if scale not in (0.5,1.0,2.0):raise ValueError('scale outside frozen grid')
    if native.dtype!=np.float32 or native.ndim!=1 or not np.isfinite(native).all():raise ValueError('invalid native waveform')
    result=(native*np.float32(scale)).astype(np.float32,copy=False)
    if not np.isfinite(result).all():raise ValueError('nonfinite scaled waveform')
    return result

def expected_cells():
    return {(e,t,q) for e in ('hubert_base','wav2vec2_base') for t in (0.5,1.0) for q in (0.5,1.0,2.0)}

def validate_cells(rows):
    keys=[(r['encoder'],r['train_input_scale'],r['inference_input_scale']) for r in rows]
    if len(keys)!=12 or len(set(keys))!=12 or set(keys)!=expected_cells():raise ValueError('missing, duplicate or unexpected scale cell')
    return True
