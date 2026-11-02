"""Complete27 cells: fixed videos/ROI, three original-PCM-unit scoring gains."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import cv2,numpy as np,torch,soundfile as sf,python_speech_features
from prepare_score_reference import SYNC,S,calc_pdist,crop_images,visual_forward,audio_forward,summarize,sha
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
from cohort_binding import identity_root,verify_member_pins
METRICS=['sync_c','sync_d','distance_d0','background_median']

def run(identity,seed):
    inputs=identity_root(identity,ROOT,require_admission=True);verify_member_pins(ROOT,identity)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'scores'/identity/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    reference=inputs/'score_reference'/identity;ref=json.loads((reference/'score_reference_receipt.json').read_text());assert ref['passed']
    track=np.load(reference/'shared_gt_track.npy',allow_pickle=False);assert track.shape==(1500,3)
    manifest=json.loads((inputs/'material'/identity/'split_manifest.json').read_text());clips=manifest['eval_clips'];assert len(clips)==6
    weights=SYNC/'data/syncnet_v2.model';assert sha(weights)==ref['syncnet_weights_sha256']
    model=S(num_layers_in_fc_layers=1024).cuda().eval();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    original,sr=sf.read(inputs/'material'/identity/'test_original.wav',dtype='int16');assert sr==16000
    audio={};max_mfcc64=0.
    for clip in clips:
        pcm=original[clip['source_start_frame']*640:(clip['source_start_frame']+250)*640];assert len(pcm)==160000
        for gain in [-6,0,6]:
            values=pcm if gain==0 else pcm.astype(np.float64)*10**(gain/20)
            mfcc=python_speech_features.mfcc(values,16000).T
            cached_mfcc=np.load(reference/clip['id']/f'mfcc_{gain:+d}dB.npy')
            max_mfcc64=max(max_mfcc64,float(np.max(np.abs(mfcc-cached_mfcc))))
            assert np.array_equal(mfcc.astype(np.float32),cached_mfcc.astype(np.float32))
            features=audio_forward(model,mfcc)
            assert np.array_equal(features.numpy(),np.load(reference/clip['id']/f'audio_features_{gain:+d}dB.npy'))
            audio[clip['id'],gain]=features
    result=dict(identity=identity,seed=seed,passed=False,cells=[],source_sha256=sha(Path(__file__)),
        reference_receipt_sha256=sha(reference/'score_reference_receipt.json'),syncnet_weights_sha256=sha(weights),
        precision='strict FP32 TF32 off',source_owned_ROI=True,evaluation_zero_original_PCM16=True,
        evaluation_plusminus_float64_original_PCM_units=True,repeat=None)
    result['actual_FP32_MFCC_input_exact']=True;result['float64_MFCC_max_abs_repeat_error']=max_mfcc64
    for train_gain in [-6,0,6]:
        for driver_gain in [-6,0,6]:
            render=ROOT/'renders'/identity/f'seed{seed}'/f'train_{train_gain:+d}'/f'driver_{driver_gain:+d}'
            metadata=json.loads((render/'render_receipt.json').read_text());assert metadata['passed'] and metadata['rgb_and_pcm_readback_exact']
            movie=render/'original_pcm.mkv';assert sha(movie)==metadata['movie_sha256']
            mux=subprocess.check_output(['ffmpeg','-v','error','-i',str(movie),'-map','0:a:0','-f','s16le','-acodec','pcm_s16le','pipe:1'])
            assert mux==original[50*640:1550*640].tobytes()
            cap=cv2.VideoCapture(str(movie));assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==25
            cells={g:[] for g in [-6,0,6]}
            for index,clip in enumerate(clips):
                assert clip['source_start_frame']==50+index*250
                images=[]
                for j in range(250):
                    ok,bgr=cap.read();assert ok
                    assert hashlib.sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==metadata['frames'][index*250+j]['rgb_sha256']
                    images.append(bgr)
                cropped=crop_images(np.stack(images),track[index*250:(index+1)*250]);vf=visual_forward(model,cropped)
                folder=out/f'train_{train_gain:+d}_driver_{driver_gain:+d}'/clip['id'];folder.mkdir(parents=True)
                np.save(folder/'visual_features.npy',vf.numpy())
                for score_gain in [-6,0,6]:
                    matrix=torch.stack(calc_pdist(vf,audio[clip['id'],score_gain],vshift=15))
                    np.save(folder/f'distances_score_{score_gain:+d}.npy',matrix.numpy())
                    row=dict(clip_id=clip['id'],guard20=summarize(matrix,np.arange(20,225)),full=summarize(matrix,np.arange(245)),
                        distances_sha256=sha(folder/f'distances_score_{score_gain:+d}.npy'))
                    cells[score_gain].append(row)
                    if train_gain==driver_gain==score_gain==0 and index==0:
                        repeated=visual_forward(model,cropped);assert torch.equal(vf,repeated)
                        assert torch.equal(matrix,torch.stack(calc_pdist(repeated,audio[clip['id'],0],vshift=15)))
                        result['repeat']=dict(visual_exact=True,matrix_exact=True,clip_id=clip['id'])
            assert not cap.read()[0];cap.release()
            for score_gain,rows in cells.items():
                cell=dict(training_gain_db=train_gain,driver_gain_db=driver_gain,evaluation_gain_db=score_gain,clips=rows)
                for support in ['guard20','full']:
                    cell['mean_'+support]={metric:float(np.mean([row[support][metric] for row in rows])) for metric in METRICS}
                    cell['mean_'+support]['offsets']=[row[support]['av_offset'] for row in rows]
                result['cells'].append(cell)
            (out/'scores.json').write_text(json.dumps(result,indent=2))
    assert len(result['cells'])==27 and result['repeat']['matrix_exact']
    result['passed']=True;(out/'scores.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(identity=identity,seed=seed,passed=True,cells=27)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.identity,a.seed)
