"""Fixed visual embeddings, three evaluation PCM gains; never invokes a renderer."""
import argparse
import hashlib
import itertools
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def gain_pcm(original, db):
    assert original.dtype == np.int16 and original.ndim == 1
    scaled = np.rint(original.astype(np.float64) * 10 ** (db / 20))
    if scaled.min() < -32768 or scaled.max() > 32767:
        raise ValueError('gain exceeds PCM16 range; clipping is forbidden')
    return scaled.astype(np.int16)


def require_grid(rows):
    expected = set(itertools.product([-6, 0, 6], [-6, 0, 6], [-6, 0, 6], range(4)))
    keys = [(r['training_gain_db'], r['driver_gain_db'], r['evaluation_gain_db'], r['clip_index']) for r in rows]
    if len(keys) != 108 or len(set(keys)) != 108 or set(keys) != expected:
        raise ValueError('incomplete or duplicated 27-cell / 108-clip grid')


def summarize(matrix, indices):
    import torch
    curve = matrix[indices].mean(0)
    value, index = torch.min(curve, 0)
    return dict(sync_c=float(torch.median(curve) - value), sync_d=float(value),
                distance_d0=float(curve[15]), background_median=float(torch.median(curve)),
                av_offset=int(15-index), windows=len(indices), curve=curve.numpy().tolist())


def paths(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    return protocol, Path(protocol['source_root']), Path(protocol['data_root'])


def check_inputs(admission):
    for entry in admission['source_inputs']:
        p = Path(entry['absolute_path'])
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], str(p)


def admit(root):
    import soundfile as sf
    protocol, source, data = paths(root)
    assert not (root / 'admission.json').exists()
    expected = json.loads((root / 'expected_parent_files.json').read_text())
    manifest = json.loads((data / 'manifest.json').read_text())
    clips = [x for x in manifest['eval_clips'] if x['split'] == 'test']
    assert [x['source_start_frame'] for x in clips] == protocol['clip_starts']
    assert all(x['frames'] == 250 for x in clips)
    original_scores = json.loads((source / 'scores/scores.json').read_text())
    assert len(original_scores['cells']) == 9 and original_scores['strict_weights']
    records = []
    for entry in expected['files']:
        p = source / entry['path']
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], str(p)
        records.append(dict(entry, absolute_path=str(p)))
        if not entry['path'].endswith('.mkv'):
            dst = root / 'inputs' / entry['path']
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    audio = []
    for clip in clips:
        p = data / clip['score_audio_path']
        assert sha(p) == clip['score_audio_sha256']
        pcm, sr = sf.read(p, dtype='int16')
        assert sr == 16000 and pcm.shape == (160000,)
        original = root / 'inputs' / 'audio' / (clip['id'] + '.wav')
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, original)
        records.append(dict(absolute_path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
        for db in [-6, 0, 6]:
            gained = gain_pcm(pcm, db)
            audio.append(dict(clip_id=clip['id'], gain_db=db, peak=int(np.abs(gained.astype(np.int32)).max()),
                              pcm_sha256=hashlib.sha256(gained.tobytes()).hexdigest(), samples=len(gained)))
    sync = Path(protocol['syncnet_root'])
    assert sha(sync / 'data/syncnet_v2.model') == original_scores['syncnet_weight_sha256']
    assert sha(sync / 'SyncNetModel.py') == original_scores['network_source_sha256']
    assert sha(sync / 'SyncNetInstance.py') == original_scores['instance_source_sha256']
    for filename in ['SyncNetModel.py', 'SyncNetInstance.py']:
        shutil.copy2(sync / filename, root / 'inputs' / filename)
    save(root / 'admission.json', dict(passed=True, protocol_sha256=sha(root/'protocol.json'),
        source_sha256={p.name:sha(p) for p in root.glob('*.py')},
        source_inputs=records, clips=clips, audio_preflight=audio,
        parent_delivery_manifest_sha256=expected['delivery_manifest_sha256'],
        new_training_runs=0, new_generated_video_frames=0, new_visual_encoder_forwards=0))
    print(json.dumps(dict(admitted=True, source_files=len(records), audio_conditions=len(audio))), flush=True)


def run(root):
    import soundfile as sf
    import torch
    import python_speech_features
    protocol, source, data = paths(root)
    admission = json.loads((root / 'admission.json').read_text())
    assert sha(root / 'protocol.json') == admission['protocol_sha256']
    assert not (root / 'results.json').exists()
    for name, digest in admission['source_sha256'].items():
        assert sha(root/name) == digest
    check_inputs(admission)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, protocol['syncnet_root'])
    from SyncNetModel import S
    from SyncNetInstance import calc_pdist
    model = S(num_layers_in_fc_layers=1024).cuda().eval()
    weight = Path(protocol['syncnet_root']) / 'data/syncnet_v2.model'
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    parent = json.loads((root / 'inputs/scores/scores.json').read_text())
    result = dict(protocol_sha256=sha(root/'protocol.json'), syncnet_weight_sha256=sha(weight),
        precision='FP32', environment=dict(python=sys.version, torch=torch.__version__,
        numpy=np.__version__, host=platform.node(), gpu=torch.cuda.get_device_name()),
        audio_conditions=[], clips=[], new_training_runs=0,new_generated_video_frames=0,new_visual_encoder_forwards=0)
    audio_features = {}
    for clip in admission['clips']:
        pcm, sr = sf.read(root/'inputs/audio'/ (clip['id']+'.wav'),dtype='int16')
        baseline = np.load(root/'inputs/scores'/clip['id']/'original_audio_features.npy')
        for db in [-6,0,6]:
            folder = root/'audio'/clip['id']/f'gain_{db}'
            folder.mkdir(parents=True)
            gained = gain_pcm(pcm,db)
            sf.write(folder/'evaluation.wav',gained,sr,subtype='PCM_16')
            assert np.array_equal(sf.read(folder/'evaluation.wav',dtype='int16')[0],gained)
            mfcc = python_speech_features.mfcc(gained,sr).T
            np.save(folder/'mfcc.npy',mfcc)
            audio = torch.from_numpy(mfcc[None,None].astype(np.float32))
            def encode():
                batches=[]
                with torch.no_grad():
                    for i in range(0,245,20):
                        batch=torch.cat([audio[:,:,:,j*4:j*4+20] for j in range(i,min(i+20,245))])
                        batches.append(model.forward_aud(batch.cuda()).cpu())
                return torch.cat(batches)
            encoded=encode();repeat=encode()
            assert torch.equal(encoded,repeat), 'nonzero audio repeat error'
            if db==0:
                assert np.array_equal(mfcc,np.load(root/'inputs/scores'/clip['id']/'original_mfcc.npy'))
                assert np.array_equal(encoded.numpy(),baseline), '0dB audio features differ from parent'
            np.save(folder/'features.npy',encoded.numpy())
            np.save(folder/'repeat_features.npy',repeat.numpy())
            audio_features[(clip['id'],db)]=encoded
            result['audio_conditions'].append(dict(clip_id=clip['id'],gain_db=db,
                pcm_sha256=hashlib.sha256(gained.tobytes()).hexdigest(),
                original_pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                features_sha256=sha(folder/'features.npy'),repeat_exact=True,
                relative_l2_vs_original=float(np.linalg.norm(encoded.numpy()-baseline)/np.linalg.norm(baseline))))
    for cell in parent['cells']:
        training=cell['training_gain_db'];driver=cell['test_gain_db']
        label={-6:'gain_m6',0:'gain_0',6:'gain_p6'}
        for index,clip in enumerate(cell['clips']):
            old=root/'inputs/scores'/clip['clip_id']/f'train_{label[training]}_test_{label[driver]}'
            visual=torch.from_numpy(np.load(old/'visual_features.npy'))
            assert visual.shape==(245,1024)
            for evaluation in [-6,0,6]:
                audio=audio_features[(clip['clip_id'],evaluation)]
                matrix=torch.stack(calc_pdist(visual,audio,vshift=15))
                if evaluation==0:
                    assert np.array_equal(matrix.numpy(),np.load(old/'distances.npy')),'0dB matrix differs from parent'
                relative=f'matrices/train_{training}_driver_{driver}_eval_{evaluation}_{clip["clip_id"]}.npy'
                (root/'matrices').mkdir(exist_ok=True)
                np.save(root/relative,matrix.numpy())
                result['clips'].append(dict(training_gain_db=training,driver_gain_db=driver,
                    evaluation_gain_db=evaluation,clip_index=index,clip_id=clip['clip_id'],
                    source_start_frame=clip['source_start_frame'],visual_features_sha256=sha(old/'visual_features.npy'),
                    matrix_path=relative,matrix_sha256=sha(root/relative),
                    guard20=summarize(matrix,np.arange(20,225)),full=summarize(matrix,np.arange(245))))
    require_grid(result['clips'])
    check_inputs(admission)
    save(root/'results.json',result)
    print(json.dumps(dict(completed=True,clips=len(result['clips']),audio_forward_calls=24,visual_forwards=0)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=['admit','run'],required=True)
    args=parser.parse_args()
    (admit if args.phase=='admit' else run)(args.root)
