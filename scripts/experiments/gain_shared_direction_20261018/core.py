"""Frozen common displacement and residual, with explicit orthant sensitivity."""
import numpy as np

CONDITIONS = ('BASE', 'BASE_REPEAT', 'GAIN_P3', 'FULL_DIRECTION',
              'COMMON', 'RESIDUAL', 'COMMON_NONNEG', 'RESIDUAL_NONNEG')

def unit(x):
    x=np.asarray(x,dtype=np.float64)
    norm=np.linalg.norm(x,axis=-1,keepdims=True)
    if not np.isfinite(x).all() or np.any(norm<1e-12):
        raise ValueError('nonfinite or degenerate direction')
    return x/norm

def fit_common(rows):
    """Each row is one different speaker: (speaker id, raw, plus3)."""
    ids=[r[0] for r in rows]
    if len(ids)!=len(set(ids)) or len(ids)<2:
        raise ValueError('fit requires unique speakers')
    delta=[]
    for _,z0,z1 in rows:
        if z0.shape!=z1.shape or z0.ndim!=2 or z0.shape[1]!=512:
            raise ValueError('encoder arrays must align')
        delta.append((unit(z1)-unit(z0)).mean(0))
    return np.mean(delta,axis=0)

def make_conditions(z0,z1,g):
    if z0.shape!=z1.shape or z0.ndim!=2 or z0.shape[1]!=512 or np.shape(g)!=(512,):
        raise ValueError('invalid latent shapes')
    u0,u1=unit(z0),unit(z1)
    r0=np.linalg.norm(np.asarray(z0,dtype=float),axis=1,keepdims=True)
    delta=u1-u0
    residual=delta-g
    common_target=u0+g
    residual_target=u0+residual
    out={'BASE':np.array(z0,dtype=np.float32),'BASE_REPEAT':np.array(z0,dtype=np.float32),
         'GAIN_P3':np.array(z1,dtype=np.float32),
         'FULL_DIRECTION':(r0*u1).astype(np.float32),
         'COMMON':(r0*unit(common_target)).astype(np.float32),
         'RESIDUAL':(r0*unit(residual_target)).astype(np.float32),
         'COMMON_NONNEG':(r0*unit(np.maximum(common_target,0))).astype(np.float32),
         'RESIDUAL_NONNEG':(r0*unit(np.maximum(residual_target,0))).astype(np.float32)}
    reconstruction=float(np.max(abs(unit(u0+g+residual)-u1)))
    if reconstruction>1e-12:
        raise ValueError('decomposition failed')
    qa={'reconstruction_max_abs':reconstruction,'common_norm':float(np.linalg.norm(g)),
        'residual_norm_mean':float(np.linalg.norm(residual,axis=1).mean()),'conditions':{}}
    for name,z in out.items():
        normerr=float(np.max(abs(np.linalg.norm(z.astype(float),axis=1,keepdims=True)/r0-1)))
        if name!='GAIN_P3' and normerr>1e-6:
            raise ValueError('norm changed')
        qa['conditions'][name]={'norm_relative_error':normerr,
                                'negative_coordinate_fraction':float(np.mean(z<0)),
                                'angular_dose_degrees':float(np.degrees(np.arccos(np.clip(np.sum(unit(z)*u0,axis=1),-1,1))).mean())}
    return out,qa

def check_disjoint(fit_ids,test_ids):
    overlap=set(fit_ids)&set(test_ids)
    if overlap or len(test_ids)!=len(set(test_ids)):
        raise ValueError('speaker overlap/duplicate: '+str(overlap))

def metrics(matrix,anchor):
    curve=np.asarray(matrix,dtype=np.float32).mean(0,dtype=np.float32).astype(float)
    b=float(np.median(curve)); best=int(np.argmin(curve))
    return {'C':b-float(curve[best]),'Dmin':float(curve[best]),'D0':float(curve[15]),
            'B':b,'Cfixed':b-float(curve[anchor]),'Dfixed':float(curve[anchor]),
            'offset':15-best,'anchor':anchor,'curve':curve.tolist()}
