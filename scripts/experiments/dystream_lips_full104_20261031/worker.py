"""Headless invocation of the official DyStream app with acceptance evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import subprocess,threading

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--audio', required=True)
p.add_argument('--output', required=True)
p.add_argument('--listener-audio')
p.add_argument('--steps', type=int, default=5)
p.add_argument('--seed', type=int, default=222)
args = p.parse_args()
deploy = Path('/home/data2/wangjiajun/DyStreamDeploy')
repo = Path('/home/data2/wangjiajun/DyStream')
out = Path(args.output).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
assert not out.exists(), f'Refusing to replace {out}'
assert json.loads((deploy/'delivery/asset_verification.json').read_text())['passed']
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
sys.path.insert(0, str(repo))
os.chdir(repo)
import torch
torch.set_num_threads(4)
assert torch.cuda.is_available()
torch.cuda.set_per_process_memory_fraction(0.55, 0)
torch.cuda.reset_peak_memory_stats()
torch.manual_seed(args.seed)
import numpy as np
np.random.seed(args.seed)

original_load = torch.load
trusted = {(repo/'checkpoints/last.ckpt').resolve(),
           (repo/'tools/pretrained_model/epoch=0-step=312000.ckpt').resolve()}
loads = []
def checkpoint_load(path, *a, **kw):
    if isinstance(path, (str, Path)) and Path(path).resolve() in trusted:
        kw.setdefault('weights_only', False)
        loads.append(str(Path(path).resolve()))
    return original_load(path, *a, **kw)
torch.load = checkpoint_load
original_state = torch.nn.Module.load_state_dict
state_results = []
renderer_results = []
def load_state(self, *a, **kw):
    result = original_state(self, *a, **kw)
    if type(self).__name__ in ('Audio2FaceGPT', 'HeadAnimatorModule'):
        state_results.append({'class':type(self).__name__,
                              'missing_keys':list(result.missing_keys),
                              'unexpected_keys':list(result.unexpected_keys)})
        assert not result.missing_keys, f'Missing model weights: {result.missing_keys}'
    return result
torch.nn.Module.load_state_dict = load_state
devices = {}
def track(module, inputs):
    if any(isinstance(x, torch.Tensor) and x.is_cuda for x in inputs):
        devices[type(module).__name__] = 'cuda:0'
hook = torch.nn.modules.module.register_module_forward_pre_hook(track)
success = False
started = time.time()
evidence={}
stop=threading.Event();nvidia=[]
def monitor():
    while not stop.is_set():
        try:
            lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
            for line in lines:
                pid,gpu,memory=[x.strip() for x in line.split(',')]
                if int(pid)==os.getpid():
                    assert gpu==os.environ['CUDA_VISIBLE_DEVICES']
                    nvidia.append(dict(time=time.time(),pid=int(pid),gpu_uuid=gpu,mib=int(memory)))
        except Exception as e:evidence['monitor_error']=repr(e)
        stop.wait(.5)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
    import app
    from PIL import Image
    import soundfile as sf
    pcm,sr=sf.read(args.audio,dtype='int16');assert sr==16000 and pcm.ndim==1
    original_model_loader=app.load_dystream_model
    def audited_model_loader():
        original_model_loader()
        model=app._dystream_model
        if getattr(model,'_evidence_installed',False):return
        original_inference=model.inference
        def audited_inference(audio,*a,**kw):
            prefix=model.inpainting_length*640
            actual=audio.detach().cpu().numpy()[0]
            assert np.array_equal(actual[:prefix],np.zeros(prefix,dtype=np.float32))
            assert np.array_equal(actual[prefix:],pcm.astype(np.float32)/32768)
            assert torch.count_nonzero(kw['audio_other']).item()==0
            evidence.update(conditioning_pcm_exact=True,native_prefix_samples=prefix,native_model_seed=int(model.cfg.seed),
                feature_processor_do_normalize=bool(model.audio_processor.feature_extractor.do_normalize),
                conditioning_audio_sha256=hashlib.sha256(Path(args.audio).read_bytes()).hexdigest(),
                conditioning_image_sha256=hashlib.sha256(Path(args.source).read_bytes()).hexdigest())
            return original_inference(audio,*a,**kw)
        model.inference=audited_inference;model._evidence_installed=True
    app.load_dystream_model=audited_model_loader
    original_render=app.latents_to_video_frames
    def audited_render(latents,ref):
        frames=original_render(latents,ref)
        evidence.update(feature_frames=int(latents.shape[1]),pose_frames=len(frames))
        assert len(frames)==len(pcm)//640-1
        return frames
    app.latents_to_video_frames=audited_render
    def load_cpu_face_detector():
        if app._face_detector is not None:
            return
        spec = app._ilu.spec_from_file_location('vis_face_detector',
            Path(app.VIS_DIR)/'utils/face_detector.py')
        module = app._ilu.module_from_spec(spec)
        spec.loader.exec_module(module)
        app._face_detector = module.FaceDetector(
            mediapipe_model_asset_path=str(Path(app.VIS_DIR)/'utils/face_landmarker.task'),
            delegate=0, face_detection_confidence=0.5, num_faces=1)
    app.load_face_detector = load_cpu_face_detector
    # The demo constructs training-only VGG/GAN losses before loading the
    # renderer. Instantiate the same four inference components directly and
    # strictly load every parameter they use, avoiding unrelated downloads.
    def load_renderer():
        if app._vis_ctx is not None:
            return
        from omegaconf import OmegaConf
        import torchvision.transforms as T
        cfg = OmegaConf.load(Path(app.VIS_DIR)/'configs/head_animator_best_0506.yaml')
        state = torch.load(repo/'tools/pretrained_model/epoch=0-step=312000.ckpt',
                           map_location='cpu')['state_dict']
        parts = {}
        for name in ('face_encoder','motion_encoder','flow_estimator','face_generator'):
            module = app._project_utils.instantiate(cfg.model[name])
            omitted = []
            if name == 'face_generator':
                # Declared in __init__ but never referenced by either forward
                # function, and absent from the released checkpoint. Removing
                # it makes any accidental use fail instead of using randomness.
                del module.warp_layer
                omitted = ['warp_layer.weight (unused by official forward)']
            prefix = name+'.'
            selected = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
            assert selected, name
            module.load_state_dict(selected, strict=True)
            module.eval().to(app.DEVICE)
            renderer_results.append({'component':name,'parameter_keys':len(selected),'strict':True,
                                     'omitted_unused':omitted})
            parts[name] = module
        parts['transform'] = T.Compose([T.Resize((512,512)),T.ToTensor(),T.Normalize([0.5],[0.5])])
        app._vis_ctx = parts
    app.load_visualization_model = load_renderer
    result, resized, masked, info = app.run_inference(
        Image.open(args.source).convert('RGB'), args.audio, args.listener_audio,
        args.steps, 0.5, 0.5, 0.0, 1.0,
        progress=lambda *a, **kw: print('PROGRESS', a, kw, flush=True))
    torch.cuda.synchronize()
    assert devices and len(loads) == 2
    shutil.copy2(result, out)
    resized.save(out.with_suffix('.reference.png'))
    masked.save(out.with_suffix('.masked.png'))
    print(info, flush=True)
    success = True
finally:
    stop.set();thread.join(timeout=3)
    hook.remove()
    receipt = {'success':success, 'pid':os.getpid(), 'seconds':time.time()-started,
               'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__,
               'cuda_runtime':torch.version.cuda, 'cudnn':torch.backends.cudnn.version(),
               'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),
               'cuda_forward_module_types':devices, 'checkpoint_loads':loads,
               'state_dict_results':state_results,
               'renderer_components':renderer_results,
               'face_detector_delegate':'CPU',
               'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
               'cpu_affinity':sorted(os.sched_getaffinity(0)), 'arguments':vars(args)}
    receipt.update(evidence)
    receipt['nvidia_own_pid_samples']=nvidia
    receipt['peak_nvidia_own_pid_mib']=max([r['mib'] for r in nvidia],default=0)
    if success:assert nvidia and 'monitor_error' not in evidence
    if out.exists():
        receipt['output_sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix('.gpu.json').write_text(json.dumps(receipt, indent=2))
