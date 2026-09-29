"""Frozen float32 projected source displacement; no model or score calls."""
import numpy as np
from scripts.experiments import tts_fixed_generator_phone_stream_20260927 as old


def shift_table(z, labels, mask, own_phone, own_global, other_phone, other_global, kind):
    assert z.dtype == np.float32 and z.shape == (len(labels), 512)
    assert kind in ('phone', 'global') and len(mask) == len(z)
    assert np.isfinite(z).all() and (z >= 0).all()
    active = np.asarray(mask, dtype=bool)
    out = z.copy()
    own, own_meta = old.replacement_table(z, labels, mask, own_phone, own_global, kind)
    other, other_meta = old.replacement_table(z, labels, mask, other_phone, other_global, kind)
    negative = 0
    count = 0
    sums = dict(half_delta=0., applied=0., projection=0., rounding_pre=0., rounding_post=0.)
    maxima = {key: 0. for key in sums}
    for i in np.flatnonzero(active):
        mu_b = own_phone.get(labels[i], own_global) if kind == 'phone' else own_global
        mu_a = other_phone.get(labels[i], other_global) if kind == 'phone' else other_global
        delta = np.subtract(mu_a, mu_b, dtype=np.float32)
        half = np.multiply(delta, np.float32(.5), dtype=np.float32)
        pre = np.add(z[i], half, dtype=np.float32)
        out[i] = np.maximum(pre, np.float32(0))
        # q11 is the ORIGINAL other cell, not this algebraic reconstruction.
        sequential = np.add(own[i], half, dtype=np.float32)
        values = dict(half_delta=half.astype(np.float64), applied=out[i].astype(np.float64)-z[i],
                      projection=out[i].astype(np.float64)-pre,
                      rounding_pre=sequential.astype(np.float64)-other[i],
                      rounding_post=np.maximum(sequential, np.float32(0)).astype(np.float64)-other[i])
        negative += int((pre < 0).sum())
        count += pre.size
        for key, value in values.items():
            sums[key] += float(np.square(value).sum())
            maxima[key] = max(maxima[key], float(np.abs(value).max()))
    assert np.array_equal(out[~active], z[~active]) and np.isfinite(out).all() and (out >= 0).all()
    description = {'speech_frames': int(active.sum()), 'identity_frames': int((~active).sum()),
                   'speech_coordinates': count, 'pre_relu_negative_coordinates': negative,
                   'pre_relu_negative_fraction': negative/count if count else 0.,
                   'own_fallback_frames': own_meta['phone_global_fallback_frames'],
                   'other_fallback_frames': other_meta['phone_global_fallback_frames'],
                   'input_z_raw_sha256': old.ah(z), 'own_z_raw_sha256': old.ah(own),
                   'other_z_raw_sha256': old.ah(other), 'replacement_z_raw_sha256': old.ah(out),
                   'replacement_min': float(out.min()), 'replacement_max': float(out.max()),
                   'descriptive_only': True, 'arithmetic': 'subtract32,multiply32(.5),add32,maximum32(0); inactive copy'}
    for key in sums:
        description[key] = {'RMS': float(np.sqrt(sums[key]/count)) if count else 0.,
                            'L2': float(np.sqrt(sums[key])), 'max_abs': maxima[key]}
    return out, description
