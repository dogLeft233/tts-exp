"""Direction and rotation-invariant relational comparison; no model calls."""
import numpy as np

def unit(x):
    x=np.asarray(x,dtype=np.float64)
    norm=np.linalg.norm(x,axis=1)
    if np.any(norm==0):raise ValueError('zero-length feature vector')
    return x/norm[:,None]

def centered_gram(x):
    gram=x@x.T
    return gram-gram.mean(axis=0)[None,:]-gram.mean(axis=1)[:,None]+gram.mean()

def cka(x,y):
    a,b=centered_gram(np.asarray(x,dtype=float)),centered_gram(np.asarray(y,dtype=float))
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return None if denom<=1e-24 else float(np.sum(a*b)/denom)

def directions(x,y):
    a,b=unit(x),unit(y);cos=np.clip(np.sum(a*b,axis=1),-1,1)
    return dict(mean_cosine=float(cos.mean()),mean_angle_degrees=float(np.degrees(np.arccos(cos)).mean()),
        unit_direction_CKA=cka(a,b),raw_linear_CKA=cka(x,y),unit_change_rms=float(np.sqrt(np.mean(np.sum((b-a)**2,axis=1)))))

def changes(x0,x1,y0,y1):
    a,b=unit(x1)-unit(x0),unit(y1)-unit(y0)
    aa,bb=np.linalg.norm(a),np.linalg.norm(b)
    valid=(np.linalg.norm(a,axis=1)>1e-10)&(np.linalg.norm(b,axis=1)>1e-10)
    return dict(change_global_cosine=None if aa*bb<=1e-24 else float(np.sum(a*b)/(aa*bb)),
        change_unit_direction_CKA=cka(a,b),first_change_rms=float(aa/np.sqrt(len(a))),second_change_rms=float(bb/np.sqrt(len(b))),
        valid_frame_count=int(valid.sum()),mean_change_cosine=None if not valid.any() else float(np.mean(np.sum(unit(a[valid])*unit(b[valid]),axis=1))))
