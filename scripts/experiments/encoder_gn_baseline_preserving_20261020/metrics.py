import numpy as np

def norm(x):return float(np.sqrt(np.einsum('ij,ij->',x,x,dtype=np.float64)))

def anchor(base,changed,reference):
    return (base.astype(np.float64)+(changed.astype(np.float64)-reference.astype(np.float64))).astype(np.float32)

def matched(base,direction,dose):
    length=norm(direction)
    if length==0:raise ValueError('zero direction')
    return (base.astype(np.float64)+direction.astype(np.float64)*(dose*norm(base)/length)).astype(np.float32)

def moments(x,y):
    assert x.shape==y.shape and x.ndim==2
    values=np.empty((len(x),6),dtype=np.float64)
    for start in range(0,len(x),1024):
        a=x[start:start+1024].astype(float);b=y[start:start+1024].astype(float);d=b-a
        aa=np.einsum('ij,ij->i',a,a);bb=np.einsum('ij,ij->i',b,b)
        assert np.all(aa>0) and np.all(bb>0)
        du=b/np.sqrt(bb[:,None])-a/np.sqrt(aa[:,None])
        values[start:start+len(a)]=np.stack([aa,bb,np.einsum('ij,ij->i',d,d),np.einsum('ij,ij->i',a,b),np.max(abs(d),axis=1),np.einsum('ij,ij->i',du,du)],axis=1)
    return values

def summarize(m):
    a,b,d,dot,mx,unitd=m.T
    cosine=np.clip(dot/np.sqrt(a*b),-1,1)
    angles=np.degrees(np.arccos(cosine));angles[d==0]=0
    return dict(relative_l2=float(np.sqrt(d.sum()/a.sum())),unit_change_rms=float(np.sqrt(unitd.mean())),
        mean_angle_degrees=float(angles.mean()),mean_cosine=float(cosine.mean()),maxabs=float(mx.max()),
        mean_norm_ratio=float(np.sqrt(b/a).mean()))
