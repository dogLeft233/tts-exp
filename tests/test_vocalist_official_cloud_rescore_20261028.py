import importlib.util
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experiments'))
import vocalist_official_cloud_rescore_20261028 as m


def test_official_k_context_changes_averaging_not_offset_tolerance():
    import numpy as np
    z=np.zeros((40,31),dtype=np.float32);z[:,18]=10
    assert all(v==0 for v in m.independent_reduction(z).values())
    z[:,18]=0;z[:,15]=10
    assert all(v==1 for v in m.independent_reduction(z).values())


def test_author_reduction_matches_independent_on_real_matrix():
    import numpy as np
    import torch
    path=Path('/home/wjj/tts-audio/tts-exp/runs/mtd_official_audit_20261028_attempt01/BAC009S0002W0122_N_0_0_512_matrix.npy')
    if not path.exists():return
    matrix=np.load(path);n=len(matrix);env=m.official_env()
    captured={}
    env['calc_pdist']=lambda *args,**kwargs:list(torch.from_numpy(matrix).unbind())
    class Dummy:
        def eval(self):return self
    def profile(frame,event,arg):
        if event=='return' and frame.f_code is env['eval_model'].__code__:
            captured.update({'K'+str(k):float(frame.f_locals['samplewise_acc_k'+str(k)][0]) for k in m.KS})
    sys.setprofile(profile)
    try:env['eval_model']([(torch.zeros(1,(n+5)*3,48,96),torch.zeros(1,1,80,int(80*((n+5)/25))+16),torch.tensor([n]))],torch.device('cpu'),Dummy())
    finally:sys.setprofile(None)
    assert captured==m.independent_reduction(matrix)


def test_original_model_known_pcm_shift_tracking_first3():
    """Six real perturbations; track offset changes without calibrating native scores."""
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    env = m.official_env()
    from models.model import SyncTransformer
    state = torch.load(m.DEPLOY/'weights/vocalist/vocalist_5f_lrs2.pth', map_location='cpu', weights_only=True)
    model = SyncTransformer()
    model.load_state_dict(state['state_dict'], strict=True)
    del state
    model.eval().cuda()
    rows = [r for r in m.read(m.RUN/'inputs.json') if r['arm'] == 'N'][:3]
    results = []
    for row in rows:
        key = row['id']
        original = m.audit.RUN/'fixtures'/key/'official512boxes'
        audio, sr = sf.read(original/'audio.wav')
        assert sr == 16000
        native = np.load(m.RUN/'matrices'/(key+'.npy'))
        # Exclude the author's invalid-candidate boundary mask from this diagnostic.
        native_offset = int(15-native[20:-20].mean(0).argmax())
        for shift in [-6400, 6400]:
            folder = m.RUN/'shift_controls'/(key+'_'+str(shift))
            folder.mkdir(parents=True, exist_ok=True)
            for image in original.glob('*.jpg'):
                link = folder/image.name
                if not link.exists():
                    link.symlink_to(image)
            shifted = np.zeros_like(audio)
            if shift > 0:
                shifted[shift:] = audio[:-shift]
            else:
                shifted[:shift] = audio[-shift:]
            sf.write(folder/'audio.wav', shifted, sr, subtype='PCM_16')
            v, a, last = m.audit.dataset_get(env, folder)
            matrix_path = folder/'matrix.npy'
            official_metrics = m.capture_official_reduction(env, model, v, a, last, matrix_path)
            matrix = np.load(matrix_path)
            assert official_metrics == m.independent_reduction(matrix)
            offset = int(15-matrix[20:-20].mean(0).argmax())
            observed = offset-native_offset
            expected = -shift/640
            results.append(dict(id=key, shift_samples=shift, native_offset_frames=native_offset,
                                shifted_offset_frames=offset, expected_change_frames=expected,
                                observed_change_frames=observed, error_frames=abs(observed-expected),
                                passed=abs(observed-expected) <= 1))
            m.write(m.RUN/'known_shift_controls.json', dict(controls=results, native_scores_modified=False))
    assert len(results) == 6 and all(r['passed'] for r in results)
