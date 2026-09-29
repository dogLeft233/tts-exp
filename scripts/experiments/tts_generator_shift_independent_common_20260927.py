"""Independent reconstruction of projected shift and its frozen descriptive quantities."""
import hashlib
import numpy as np
ah=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def rebuild(row,arm,kind,fit,meta):
 z=np.load(row['arms'][arm]['features_path'])['z'];q=row['arms'][arm];fi=meta['folds'].index(row['speaker']);own=meta['sources'].index(arm);labels={x:i for i,x in enumerate(meta['labels'])};counts=fit['speaker_counts'][fi];raw=fit['raw_mu'][fi];out=z.copy();selfcell=z.copy();othercell=z.copy();acc={k:[] for k in ['half_delta','applied','projection','rounding_pre','rounding_post']};negative=0;active=np.asarray(q['speech_mask']);fallback=0
 for i in np.flatnonzero(active):
  li=labels.get(q['phone_labels'][i]);templates=[]
  for src in [own,1-own]:
   mu=raw[src,-1].astype(np.float32)
   if kind=='phone' and li is not None and counts[li]>0:
    w=np.float64(counts[li])/np.float64(counts[li]+4);mu=(w*raw[src,li]+(1-w)*raw[src,-1]).astype(np.float32)
   templates.append(mu)
  if kind=='phone' and (li is None or not counts[li]):fallback+=1
  b,a=templates;selfcell[i]=np.multiply(z[i],np.float32(.5),dtype=np.float32)+np.multiply(b,np.float32(.5),dtype=np.float32);othercell[i]=np.multiply(z[i],np.float32(.5),dtype=np.float32)+np.multiply(a,np.float32(.5),dtype=np.float32);delta=np.subtract(a,b,dtype=np.float32);half=np.multiply(delta,np.float32(.5),dtype=np.float32);pre=np.add(z[i],half,dtype=np.float32);out[i]=np.maximum(pre,np.float32(0));seq=np.add(selfcell[i],half,dtype=np.float32);negative+=int((pre<0).sum())
  values={'half_delta':half.astype(float),'applied':out[i].astype(float)-z[i],'projection':out[i].astype(float)-pre,'rounding_pre':seq.astype(float)-othercell[i],'rounding_post':np.maximum(seq,np.float32(0)).astype(float)-othercell[i]}
  for k,v in values.items():acc[k].append(v)
 assert np.array_equal(out[~active],z[~active]) and np.isfinite(out).all() and (out>=0).all();n=int(active.sum())*512
 d={'speech_frames':int(active.sum()),'identity_frames':int((~active).sum()),'speech_coordinates':n,'pre_relu_negative_coordinates':negative,'pre_relu_negative_fraction':negative/n if n else 0.,'own_fallback_frames':fallback,'other_fallback_frames':fallback,'input_z_raw_sha256':ah(z),'own_z_raw_sha256':ah(selfcell),'other_z_raw_sha256':ah(othercell),'replacement_z_raw_sha256':ah(out),'replacement_min':float(out.min()),'replacement_max':float(out.max())}
 for k,values in acc.items():
  s=sum(float(np.square(v).sum()) for v in values);maximum=max([float(abs(v).max()) for v in values]+[0.]);d[k]={'RMS':float(np.sqrt(s/n)) if n else 0.,'L2':float(np.sqrt(s)),'max_abs':maximum}
 return out,d

def compare_dict(got,expected):
 error=0.
 for k,x in got.items():
  y=expected[k]
  if isinstance(x,dict):error=max(error,compare_dict(x,y))
  elif isinstance(x,(int,float,np.number)):error=max(error,float(abs(x-y)))
  else:assert x==y,(k,x,y)
 return error
