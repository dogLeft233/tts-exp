"""Read-only independent checks of completed media and raw scorer reductions."""
from pathlib import Path
import argparse, hashlib, json, subprocess
import numpy as np
import soundfile as sf
from native_export import exported_frames

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_classifier_probabilities(logits, saved, p_zero):
    import torch
    z=np.asarray(logits,dtype=np.float64)
    assert z.shape==(21,) and np.isfinite(z).all()
    precise=np.exp(z-z.max());precise/=precise.sum()
    # The frozen scorer performs CPU FP32 softmax over the flattened logits.
    # Require an exact replay at that precision, plus an independent FP64 check.
    replay=torch.tensor(logits,dtype=torch.float32).softmax(0).numpy()
    assert np.array_equal(replay,np.asarray(saved)), 'FP32 saved probability replay mismatch'
    assert float(replay[10])==p_zero
    error=float(abs(precise-np.asarray(saved)).max())
    assert error < len(z)*np.finfo(np.float32).eps
    return error


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
    b=Path('/home/data2/wangjiajun/av-evaluation');inputs=read(r/'scoring_inputs.json');rows=inputs['rows'];checks=[]
    for row in rows:
        assert sha(row['original_audio_path'])==row['original_audio_sha256']
        assert sha(row['image_path'])==row['image_sha256']
        assert sha(row['audio_path'])==row['audio_sha256'] and sha(row['video_path'])==row['video_sha256']
        pcm,sr=sf.read(row['audio_path'],dtype='int16');assert sr==16000
        x=pcm.astype(np.float64)/32768
        assert abs(20*np.log10(np.sqrt(np.mean(x*x)))+24)<.002 and abs(x).max()<=10**(-1/20)
        actual=subprocess.check_output([str(b/'tools/bin/ffmpeg'),'-nostdin','-v','error','-i',row['video_path'],'-map','0:a:0','-f','s16le','-'])
        assert actual==pcm.astype('<i2').tobytes()
        receipt=read(r/'generation'/row['id']/'receipt.json')
        assert receipt['frames']==exported_frames(row['expected_frames']) and receipt['unsupported_tail_samples']==row['samples']-receipt['frames']*640
        assert receipt['conditioning_audio_sha256']==row['audio_sha256']
        assert receipt['feature_frames']==receipt['pose_frames']==row['expected_frames']
        assert receipt['decoded_pcm_exact'] and receipt['mean_frame_difference']>.01
        checks.append(dict(id=row['id'],audio_sha256=row['audio_sha256'],video_sha256=row['video_sha256'],samples=len(pcm)))
    errors={}
    for rec in read(r/'syncnet/scores.json'):
        z=np.load(r/'syncnet'/(rec['id']+'_distances.npy')).astype(np.float64);assert z.shape[1]==31
        curve=z.mean(0);c=float(np.median(curve)-curve.min());assert abs(c-rec['C'])<2e-5
        errors['syncnet_max_abs']=max(errors.get('syncnet_max_abs',0),abs(c-rec['C']))
    for model in ['mtd','vocalist']:
        for rec in read(r/model/'batch_scores.json'):
            if rec['status']!='PASS':continue
            result=read(r/model/rec['id']/'deployment_validation.json')
            assert result['strict_load']
            for shift,record in result['results'].items():
                z=np.load(r/model/rec['id']/('distances_'+f'{int(shift):+d}'+'.npy'))
                k=float(np.mean(abs(z.argmax(1)-15)<=1))
                assert abs(k-record['author_metrics']['K5'])<1e-7
    for rec in read(r/'cogenav/scores.json'):
        with np.load(r/'cogenav/embeddings'/(rec['id']+'.npz')) as data:
            audio=data['audio'].astype(np.float64);video=data['video'].astype(np.float64)
        eq=float(np.maximum(0,np.sum(audio*video,-1)/(np.linalg.norm(audio,axis=1)*np.linalg.norm(video,axis=1))).mean())
        assert abs(eq-rec['eq1'])<1e-6
    for rec in read(r/'avs_u/batch_scores.json'):
        if rec['status']!='PASS':continue
        with np.load(r/'avs_u'/rec['id']/'raw_embeddings.npz') as data:
            audio=data['audio_0'].astype(np.float64);video=data['video'].astype(np.float64)
        value=float(np.mean(np.sum(audio*video,-1)/(np.linalg.norm(audio,axis=1)*np.linalg.norm(video,axis=1))))
        expected=read(r/'avs_u'/rec['id']/'deployment_validation.json')['results']['0']['frame_mean']
        assert abs(value-expected)<1e-6
    for model in ['synchformer','sparsesync']:
        for rec in read(r/model/'batch_scores.json'):
            if rec['status']!='PASS':continue
            result=read(r/model/rec['id']/'deployment_validation.json')
            for out in result['results'].values():
                z=np.asarray(out['logits'],dtype=np.float64);assert z.shape==(21,) and np.isfinite(z).all()
                probabilities=np.exp(z-z.max());probabilities/=probabilities.sum()
                assert np.allclose(probabilities,out['probabilities'],rtol=1e-4,atol=1e-8)
                value=verify_classifier_probabilities(out['logits'],out['probabilities'],out['p_zero'])
                errors['classifier_fp64_max_abs']=max(errors.get('classifier_fp64_max_abs',0),value)
    assert len(read(r/'peavs/scores.json')['scores'])==209
    summary=read(r/'summary.json')
    assert summary['coverage']['SyncNet']['complete_pairs']==104
    assert summary['coverage']['SyncReward']['complete_pairs']==104
    assert summary['coverage']['Synchformer']['complete_pairs']==sum(all(min(x['samples']//640,x['actual_video_frames'])>=125 for x in rows if x['source']==source) for source in {x['source'] for x in rows if x['paired_available']})
    assert summary['coverage']['SparseSync']['complete_pairs']==summary['coverage']['Synchformer']['complete_pairs']
    result=dict(status='PASS',verified_media=209,raw_reductions=['SyncNet FP64 crosscheck','MTD K5','VocaLiST K5','CoGenAV Eq1','AVS_u cosine','Synchformer exact CPU FP32 softmax plus FP64','SparseSync exact CPU FP32 softmax plus FP64'],
        score_boundaries='PEAVS/SyncReward raw model outputs checked for complete identity and finite values by analysis; no claim of independent model reimplementation',
        errors=errors,files=checks,summary_sha256=sha(r/'summary.json'),protocol_sha256=sha(r/'protocol.json'))
    (r/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))

if __name__=='__main__':main()
