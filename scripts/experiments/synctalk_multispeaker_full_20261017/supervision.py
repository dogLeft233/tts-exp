"""Original background/torso routines and original MediaPipe eye coefficients."""
import argparse,hashlib,importlib.util,json,shutil,sys,time
from pathlib import Path
import cv2
import numpy as np

def run(root,identity):
    cv2.setNumThreads(2)
    official=root/'official';folder=root/'prepared'/identity
    sys.path.insert(0,str(official/'data_utils'))
    spec=importlib.util.spec_from_file_location('official_process',official/'data_utils/process.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert not (folder/'supervision_receipt.json').exists()
    train=folder/'train'
    assert not (train/'bc.jpg').exists()
    module.extract_background(str(train),str(train/'ori_imgs'))
    background_sha=hashlib.sha256((train/'bc.jpg').read_bytes()).hexdigest()
    for split,count in [('train',6000),('val',750),('test',1500)]:
        data=folder/split
        if split!='train':shutil.copy2(train/'bc.jpg',data/'bc.jpg')
        assert hashlib.sha256((data/'bc.jpg').read_bytes()).hexdigest()==background_sha
        module.extract_torso_and_gt(str(data),str(data/'ori_imgs'))
        assert len(list((data/'gt_imgs').glob('*.jpg')))==count
        assert len(list((data/'torso_imgs').glob('*.png')))==count
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from scipy.signal import savgol_filter
    options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(official/'data_utils/blendshape_capture/face_landmarker.task')),
        output_face_blendshapes=True,output_facial_transformation_matrixes=True,num_faces=1)
    detector=vision.FaceLandmarker.create_from_options(options)
    manifest=json.loads((root/'material'/identity/'split_manifest.json').read_text());eyes=[]
    for s in manifest['splits']:
        data=folder/s['name'];count=s['frames']
        cap=cv2.VideoCapture(str(root/'material'/identity/(s['recording']+'.mkv')))
        cap.set(cv2.CAP_PROP_POS_FRAMES,s['source_start_frame']);values=np.zeros((count,52),dtype=np.float32)
        png_parity=0.
        for i in range(count):
            ok,bgr=cap.read();assert ok
            image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            result=detector.detect(image);assert len(result.face_blendshapes)==1
            scores=np.array([c.score for c in result.face_blendshapes[0]])[1:]
            values[i]=np.append(scores,0)
            if i==0:
                path=data/'eye_input_control.png';cv2.imwrite(str(path),bgr)
                via_png=detector.detect(mp.Image.create_from_file(str(path)))
                compare=np.array([c.score for c in via_png.face_blendshapes[0]])[1:]
                png_parity=float(np.abs(compare-scores).max());assert png_parity==0
            if (i+1)%250==0:print(json.dumps(dict(identity=identity,split=s['name'],phase='eyes',frames=i+1)),flush=True)
        cap.release()
        output=np.zeros(values.shape)
        for column in range(52):output[:,column]=savgol_filter(values[:,column],5,3)
        assert np.isfinite(output).all();np.save(data/'bs.npy',output)
        eyes.append(dict(split=s['name'],frames=count,png_array_input_max_error=png_parity,bs_sha256=hashlib.sha256((data/'bs.npy').read_bytes()).hexdigest()))
    detector.close()
    result=dict(identity=identity,passed=True,background_fit='training frames only',shared_training_background_sha256=background_sha,
        original_background_and_torso_source_sha256=hashlib.sha256((official/'data_utils/process.py').read_bytes()).hexdigest(),
        eyes=eyes,training_admitted=False)
    (folder/'supervision_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.root,a.identity)
