"""Independent pooled-template signed float32 shift and frozen dose description."""
import hashlib
import numpy as np
ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def rebuild(z,q,condition,fit,fm):
 kind,sign=condition.split('_');fi=fm['folds'].index('S0912');raw=fit['raw_mu'][fi];count=fit['speaker_counts'][fi];labels={x:i for i,x in enumerate(fm['labels'])};order=[0,1] if sign=='plus' else [1,0];active=np.array(q['speech_mask'],bool);out=z.copy();selfcell=z.copy();othercell=z.copy();acc={k:[] for k in ['half_delta','applied','projection','rounding_pre','rounding_post']};negative=0;fallback=0
 for i in np.flatnonzero(active):
  ix=labels.get(q['phone_labels'][i]);tt=[]
  for src in order:
   mu=raw[src,-1].astype(np.float32)
   if kind=='phone' and ix is not None and count[ix]>0:
    w=np.float64(count[ix])/np.float64(count[ix]+4);mu=(w*raw[src,ix]+(1-w)*raw[src,-1]).astype(np.float32)
   tt.append(mu)
  if kind=='phone' and (ix is None or not count[ix]):fallback+=1
  own,other=tt;selfcell[i]=np.multiply(z[i],np.float32(.5),dtype=np.float32)+np.multiply(own,np.float32(.5),dtype=np.float32);othercell[i]=np.multiply(z[i],np.float32(.5),dtype=np.float32)+np.multiply(other,np.float32(.5),dtype=np.float32);delta=np.subtract(other,own,dtype=np.float32);half=np.multiply(delta,np.float32(.5),dtype=np.float32);pre=np.add(z[i],half,dtype=np.float32);out[i]=np.maximum(pre,np.float32(0));seq=np.add(selfcell[i],half,dtype=np.float32);negative+=int((pre<0).sum())
  vv={'half_delta':half.astype(float),'applied':out[i].astype(float)-z[i],'projection':out[i].astype(float)-pre,'rounding_pre':seq.astype(float)-othercell[i],'rounding_post':np.maximum(seq,np.float32(0)).astype(float)-othercell[i]}
  for k,v in vv.items():acc[k].append(v)
 assert np.array_equal(out[~active],z[~active]) and np.isfinite(out).all() and (out>=0).all();n=int(active.sum())*512
 d={'speech_frames':int(active.sum()),'identity_frames':int((~active).sum()),'speech_coordinates':n,'pre_relu_negative_coordinates':negative,'pre_relu_negative_fraction':negative/n if n else 0.,'own_fallback_frames':fallback,'other_fallback_frames':fallback,'input_z_raw_sha256':ah(z),'own_z_raw_sha256':ah(selfcell),'other_z_raw_sha256':ah(othercell),'replacement_z_raw_sha256':ah(out),'replacement_min':float(out.min()),'replacement_max':float(out.max())}
 for k,vv in acc.items():
  s=sum(float(np.square(v).sum()) for v in vv);d[k]={'RMS':float(np.sqrt(s/n)) if n else 0.,'L2':float(np.sqrt(s)),'max_abs':max([float(abs(v).max()) for v in vv]+[0.])}
 return out,d
