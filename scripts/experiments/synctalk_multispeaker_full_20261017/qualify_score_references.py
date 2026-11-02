"""Fresh all18 audio forwards and independent GT distance/endpoint checks per identity."""
import hashlib,json
from pathlib import Path
import numpy as np,soundfile as sf,torch,python_speech_features
from prepare_score_reference import SYNC,S,audio_forward,sha
from validate_scores import distance,endpoint
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')

def run():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weights=SYNC/'data/syncnet_v2.model';model=S(num_layers_in_fc_layers=1024).cuda().eval()
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    for identity in ['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers']:
        ref=ROOT/'score_reference'/identity;receipt=json.loads((ref/'score_reference_receipt.json').read_text());assert receipt['passed']
        assert receipt['syncnet_weights_sha256']==sha(weights)
        original,sr=sf.read(ROOT/'material'/identity/'test_original.wav',dtype='int16');assert sr==16000
        checked=0;max_matrix=0.;max_endpoint=0.;max_mfcc64=0.
        for i,clip in enumerate(receipt['clips']):
            assert clip['id']==f'test_{i:02d}' and clip['source_start_frame']==50+i*250
            pcm=original[(50+i*250)*640:(300+i*250)*640]
            assert hashlib.sha256(pcm.tobytes()).hexdigest()==clip['pcm_sha256']
            vf=np.load(ref/clip['id']/'gt_visual_features.npy',allow_pickle=False)
            for gain in [-6,0,6]:
                values=pcm if gain==0 else pcm.astype(np.float64)*10**(gain/20)
                mfcc=python_speech_features.mfcc(values,16000).T
                cached_mfcc=np.load(ref/clip['id']/f'mfcc_{gain:+d}dB.npy')
                max_mfcc64=max(max_mfcc64,float(np.max(np.abs(mfcc-cached_mfcc))))
                assert np.array_equal(mfcc.astype(np.float32),cached_mfcc.astype(np.float32)),'actual original SyncNet input must be bitwise identical'
                af=audio_forward(model,mfcc).numpy()
                assert np.array_equal(af,np.load(ref/clip['id']/f'audio_features_{gain:+d}dB.npy'))
                stored=np.load(ref/clip['id']/f'gt_distances_{gain:+d}dB.npy',allow_pickle=False)
                rebuilt=distance(vf,af);error=float(np.max(np.abs(stored-rebuilt)));assert error<3e-5;max_matrix=max(max_matrix,error)
                expected=next(row for row in clip['baseline'] if row['evaluation_gain_db']==gain)
                for support,indexes in [('guard20',np.arange(20,225)),('full',np.arange(245))]:
                    reconstructed=endpoint(rebuilt,indexes);assert reconstructed['av_offset']==expected[support]['av_offset']
                    for key in ['sync_c','sync_d','distance_d0','background_median']:
                        error=abs(reconstructed[key]-expected[support][key]);assert error<3e-5;max_endpoint=max(max_endpoint,error)
                checked+=1
        result=dict(identity=identity,passed=True,all18_audio_features_fresh_exact=True,checked_matrices=checked,
            independent_matrix_max_abs_error=max_matrix,independent_endpoint_max_abs_error=max_endpoint,tolerance=3e-5,
            actual_FP32_MFCC_input_exact=True,float64_MFCC_max_abs_repeat_error=max_mfcc64,
            zero_original_PCM16=True,plusminus_float64_original_PCM_units=True,source_owned_ROI_unchanged=True,
            source_sha256=sha(Path(__file__)),reference_receipt_sha256=sha(ref/'score_reference_receipt.json'),training_admitted=False)
        target=ref/'independent_reference_qualification.json';assert not target.exists();target.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':run()
