"""Read-only conditioning, pose-coverage and residual-location diagnosis of frozen v2."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation
from geometry import project
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
sys.path.insert(0,str(ROOT/'official/data_utils/face_tracking'))
from util import euler2rot

def pose_summary(rotation,translation):
    relative=rotation@rotation[[2]].transpose(0,2,1)
    vectors=np.degrees(Rotation.from_matrix(relative).as_rotvec())
    angles=np.linalg.norm(vectors,axis=1)
    return dict(relative_rotation_vector_p5_degrees=np.quantile(vectors,.05,axis=0).tolist(),
        relative_rotation_vector_p95_degrees=np.quantile(vectors,.95,axis=0).tolist(),
        relative_rotation_angle_p95_degrees=float(np.quantile(angles,.95)),
        translation_std_mm=np.std(translation,axis=0).tolist())

def run():
    output=ROOT/'diagnostics/DebbieDingell/frozen_v2_failure_attempt01';output.mkdir(parents=True,exist_ok=False)
    result=dict(diagnostic_only=True,inputs_unchanged=True,optimizer_updates=0,identities=[])
    for identity in ['SherrodBrown','PatrickLeahy','DebbieDingell']:
        base=ROOT/'prepared'/identity;qualified=json.loads((base/'geometry_v2/geometry_qualification.json').read_text())
        points=np.load(base/'geometry_v2/fixed_training_points_float64.npy')
        row=dict(identity=identity,triangulation_condition_max=qualified['triangulation_condition_max'],
            point_range_m=(np.max(points,axis=0)-np.min(points,axis=0)).tolist(),
            point_std_m=np.std(points,axis=0).tolist(),recordings={})
        for split in ['train','test']:
            state=torch.load(base/split/'track_params.pt',map_location='cpu',weights_only=False)
            rotation=euler2rot(state['euler']).numpy().astype(np.float64)
            row['recordings'][split]=pose_summary(rotation,state['trans'].numpy().astype(np.float64))
        result['identities'].append(row)
    base=ROOT/'prepared/DebbieDingell';points=np.load(base/'geometry_v2/fixed_training_points_float32.npy').astype(np.float64)
    pose=np.load(base/'geometry_v2/test/pose_arrays.npz');tracks=np.load(base/'geometry_v2/test/track_xys.npy').astype(np.float64)
    focal=float(torch.load(base/'train/bundle_adjustment.pt',weights_only=False)['focal'][0])
    predicted=project(np.broadcast_to(points,(1500,)+points.shape),pose['rotation'],pose['translation'],focal)
    per_frame=np.sqrt(np.mean((predicted-tracks)**2,axis=(1,2)));per_point=np.sqrt(np.mean((predicted-tracks)**2,axis=(0,2)))
    result['independent_test_flow_median']=float(np.median(per_frame));result['independent_test_flow_p95']=float(np.quantile(per_frame,.95))
    keys=np.loadtxt(base/'train/keypoints.txt',dtype=np.int32).reshape(-1,2);worst=np.argsort(per_point)[-30:]
    result['largest_residual_training_keys']=[dict(key=int(i),reference_xy=keys[i].tolist(),rms_px=float(per_point[i]),training_point_m=points[i].tolist()) for i in worst]
    gallery=[]
    reference=cv2.imread(str(base/'train/ori_imgs/2.jpg'))
    for index in worst:cv2.circle(reference,tuple(keys[index]),3,(0,0,255),-1)
    cv2.putText(reference,'Training reference: 30 largest test residual keys',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    gallery.append(reference)
    for frame in [0,500,1000]:
        image=cv2.imread(str(base/'test/ori_imgs'/f'{frame}.jpg'))
        for index in worst:
            target=tuple(np.rint(tracks[frame,index]).astype(int));estimate=tuple(np.rint(predicted[frame,index]).astype(int))
            cv2.line(image,target,estimate,(0,0,255),1);cv2.circle(image,target,2,(255,255,0),-1);cv2.circle(image,estimate,2,(0,255,0),-1)
        cv2.putText(image,f'test {frame} RMS={per_frame[frame]:.3f}px',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);gallery.append(image)
    cv2.imwrite(str(output/'residual_location.png'),np.concatenate(gallery,axis=1))
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':run()
