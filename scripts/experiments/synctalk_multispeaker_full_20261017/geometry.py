"""Coordinate and mask contracts for training-owned calibration."""
import numpy as np

def official_euler_from_rotation(rotations):
    """Invert pinned util.euler2rot: Rx(theta) Ry(phi) Rz(-psi)."""
    from scipy.spatial.transform import Rotation
    angles=Rotation.from_matrix(np.asarray(rotations)).as_euler('XYZ')
    angles[...,2]*=-1
    return angles

def alpha_flow_mask(portrait_bgr):
    if portrait_bgr.ndim!=3 or portrait_bgr.shape[2]!=3:raise ValueError('require BGR portrait mask')
    alpha=255-portrait_bgr[:,:,1]
    return np.repeat(alpha[:,:,None],3,axis=2).astype(np.uint8)

def fit_training_gauge(original_rot,original_trans,bundle_rot,bundle_trans):
    """Fit X_bundle=s Q X_BFM+T using training camera pairs only."""
    r0=np.asarray(original_rot,dtype=np.float64);r1=np.asarray(bundle_rot,dtype=np.float64)
    t0=np.asarray(original_trans,dtype=np.float64);t1=np.asarray(bundle_trans,dtype=np.float64)
    u,_,vt=np.linalg.svd(np.mean(np.swapaxes(r1,1,2)@r0,axis=0))
    fix=np.eye(3);fix[2,2]=np.linalg.det(u@vt);q=u@fix@vt
    a=np.concatenate([r1,-t0[:,:,None]],axis=2).reshape(-1,4)
    solution,_,rank,_=np.linalg.lstsq(a,-t1.reshape(-1),rcond=None)
    shift=solution[:3];scale=float(solution[3]);condition=float(np.linalg.cond(a))
    if rank!=4 or not np.isfinite(scale) or scale<=0:raise ValueError('unidentifiable positive similarity')
    mapped_r=r0@q.T;mapped_t=scale*t0-np.einsum('nij,j->ni',mapped_r,shift)
    relative=mapped_r@np.swapaxes(r1,1,2)
    angles=np.degrees(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
    errors=np.linalg.norm(mapped_t-t1,axis=1)
    return dict(rotation=q.tolist(),translation=shift.tolist(),scale=scale,rank=int(rank),condition=condition,
        rotation_error_degrees_p95=float(np.quantile(angles,.95)),camera_translation_error_m_p95=float(np.quantile(errors,.95)))

def map_poses_to_training_gauge(rot,trans,gauge):
    q=np.array(gauge['rotation']);shift=np.array(gauge['translation']);scale=gauge['scale']
    r=np.asarray(rot)@q.T
    t=scale*np.asarray(trans)-np.einsum('nij,j->ni',r,shift)
    return r,t

def project(points,rot,trans,focal=1000,cxy=(256,256)):
    camera=np.einsum('nij,nkj->nki',rot,points)+np.asarray(trans)[:,None,:]
    return np.stack([-focal*camera[:,:,0]/camera[:,:,2]+cxy[0],focal*camera[:,:,1]/camera[:,:,2]+cxy[1]],axis=-1)
