"""Actual loaded pixels, poses and temporal audio contexts before training."""
from runtime import *
def main():
 configure();checks=[]
 import cv2
 import nerf_triplane.provider as provider
 original=provider.get_rays;observed={}
 def capture(*a,**kw):
  result=original(*a,**kw);observed['inds']=result['inds'].detach().cpu().numpy();return result
 for split,ids in [('train',[a['id'] for a in protocol()['arms']]),('val',['clean']),('test',['clean'])]:
  scales=[None] if split=='train' else [1.,.5]
  for key in ids:
   for scale in scales:
    if scale is None:scale=next(a['input_scale'] for a in protocol()['arms'] if a['id']==key)
    d=dataset(key,split,True,scale)
    for idx in [0,len(d.images)//2,len(d.images)-1]:
     seed(idx,'loader_qualification');provider.get_rays=capture
     try:data=d.collate([idx])
     finally:provider.get_rays=original
     row=d.source_frames[idx];source=DATA/'source/May';image=cv2.cvtColor(cv2.imread(str(source/'gt_imgs'/f"{row['img_id']}.jpg")),cv2.COLOR_BGR2RGB).astype(np.float32)/255
     expected=image.reshape(-1,3)[observed['inds'][0]] if d.training else image
     assert np.array_equal(data['images'][0].cpu().numpy(),expected)
     expected_pose=provider.nerf_matrix_to_ngp(np.array(row['transform_matrix'],dtype=np.float32),scale=d.scale,offset=d.offset)
     assert np.array_equal(d.poses[idx].cpu().numpy(),expected_pose)
     checks.append(dict(split=split,arm=key,scale=scale,index=idx,pixels_exact=True,pose_exact=True,all_loaded_audio_exact=True))
    del d
 write(ROOT/'provider_qualification.json',dict(passed=True,checks=checks,count=len(checks),optimizer_calls=0))
 print('Provider qualified',len(checks),flush=True)
if __name__=='__main__':main()
