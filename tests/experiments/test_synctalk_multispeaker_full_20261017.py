import copy
import importlib.util
import unittest
from pathlib import Path

path=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/prepare_materials.py'
spec=importlib.util.spec_from_file_location('multi_prepare',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SourceContracts(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(identity=str(i),main=dict(original_video_id=f'a{i}',seconds=300),
                        test=dict(original_video_id=f'b{i}',seconds=120)) for i in range(4)]

    def test_same_original_recording_rejected(self):
        self.rows[0]['test']['original_video_id']='a0'
        with self.assertRaises(ValueError):module.validate_selection(self.rows)

    def test_material_and_context_shortage_rejected(self):
        for main,test in [(250,170),(300,60),(300,200)]:
            rows=copy.deepcopy(self.rows);rows[0]['main']['seconds']=main;rows[0]['test']['seconds']=test
            with self.assertRaises(ValueError):module.validate_selection(rows)

    def test_missing_or_duplicate_identity_rejected(self):
        with self.assertRaises(ValueError):module.validate_selection(self.rows[:3])
        self.rows[3]['identity']='0'
        with self.assertRaises(ValueError):module.validate_selection(self.rows)


class IndependentDuplicateGate(unittest.TestCase):
    def test_competing_faces_rejected_but_small_low_confidence_artifact_recorded(self):
        qspec=importlib.util.spec_from_file_location('multi_faces',path.with_name('preprocess_images_landmarks.py'))
        faces=importlib.util.module_from_spec(qspec);qspec.loader.exec_module(faces)
        primary=[100,100,300,400,.999]
        self.assertEqual(faces.require_primary_face([primary,[10,10,30,30,.54]]),1)
        for boxes in [[],[primary,[10,10,200,300,.6]],[primary,[10,10,30,30,.95]]]:
            with self.assertRaises(ValueError):faces.require_primary_face(boxes)

    def test_gain_and_shifted_reupload_detected(self):
        import numpy as np
        qspec=importlib.util.spec_from_file_location('multi_qa',path.with_name('independent_material_qa.py'))
        qa=importlib.util.module_from_spec(qspec);qspec.loader.exec_module(qa)
        rng=np.random.default_rng(87)
        a=rng.normal(0,.05,24000);b=rng.normal(0,.05,20000)
        b[3100:11100]=a[8000:16000]*1.7
        result=qa.duplicate_audio(a,b)
        self.assertTrue(result['flag'])
        self.assertAlmostEqual(result['match_start_seconds'],1.55)
        self.assertFalse(qa.duplicate_audio(a,rng.normal(0,.05,20000))['flag'])


class CoordinateAndMaskContracts(unittest.TestCase):
    def geometry(self):
        s=importlib.util.spec_from_file_location('multi_geometry',path.with_name('geometry.py'))
        g=importlib.util.module_from_spec(s);s.loader.exec_module(g);return g

    def test_portrait_visualization_is_not_a_flow_mask(self):
        import numpy as np
        g=self.geometry()
        image=np.array([[[255,255,255],[0,0,255],[100,100,255]]],dtype=np.uint8)
        flow=g.alpha_flow_mask(image)
        np.testing.assert_array_equal(flow[0],[[0,0,0],[255,255,255],[155,155,155]])

    def test_official_euler_uses_negative_z_rotation(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        g=self.geometry()
        angles=np.random.default_rng(19).uniform(-.6,.6,(30,3))
        expected=angles.copy();expected[:,2]*=-1
        matrices=Rotation.from_euler('XYZ',expected).as_matrix()
        np.testing.assert_allclose(g.official_euler_from_rotation(matrices),angles,atol=1e-12)

    def test_training_only_similarity_preserves_heldout_projection(self):
        import numpy as np
        g=self.geometry();rng=np.random.default_rng(173)
        rotations=[]
        for angle in np.linspace(-.3,.4,60):
            c,s=np.cos(angle),np.sin(angle);rotations.append([[c,0,s],[0,1,0],[-s,0,c]])
        r0=np.array(rotations);t0=rng.normal(0,.01,(60,3));t0[:,2]-=.6
        angle=.13;c,s=np.cos(angle),np.sin(angle);q=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        scale=1.23;shift=np.array([.02,-.013,.04]);r1=r0@q.T;t1=scale*t0-np.einsum('nij,j->ni',r1,shift)
        gauge=g.fit_training_gauge(r0,t0,r1,t1)
        self.assertAlmostEqual(gauge['scale'],scale,places=10)
        np.testing.assert_allclose(gauge['rotation'],q,atol=1e-12)
        np.testing.assert_allclose(gauge['translation'],shift,atol=1e-12)
        # New poses and new points never participate in the fitted calibration.
        points=rng.normal(0,.025,(7,31,3));rtest=r0[10:17];ttest=t0[10:17]+rng.normal(0,.01,(7,3))
        rm,tm=g.map_poses_to_training_gauge(rtest,ttest,gauge)
        mapped=scale*np.einsum('ij,nkj->nki',q,points)+shift
        np.testing.assert_allclose(g.project(points,rtest,ttest),g.project(mapped,rm,tm),atol=1e-9)

    def test_triangulation_recovers_known_rigid_points(self):
        import sys
        import numpy as np
        sys.path.insert(0,str(path.parent))
        import qualify_pose
        g=self.geometry();rng=np.random.default_rng(71)
        rot=np.repeat(np.eye(3)[None],40,axis=0);trans=rng.normal(0,.02,(40,3));trans[:,2]-=.6
        points=rng.normal(0,.025,(11,3))
        tracks=g.project(np.broadcast_to(points,(40,11,3)),rot,trans,950)
        np.testing.assert_allclose(qualify_pose.triangulate(rot,trans,tracks,950),points,atol=1e-10)

    def test_attention_uses_original_recording_beyond_split_edges(self):
        import ast,sys
        import numpy as np,torch
        sys.path.insert(0,str(path.parent))
        import audio_context
        official=Path('/dev/shm/synctalk-multispeaker-data-20261017/official_exact/nerf_triplane/utils.py')
        if not official.exists():self.skipTest('requires pinned official runtime')
        node=next(x for x in ast.parse(official.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='get_audio_features')
        scope={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(official),'exec'),scope)
        original=scope['get_audio_features'];mapped=audio_context.wrap(original)
        first=torch.arange(100,dtype=torch.float32)[:,None,None].repeat(1,1,512)
        second=first+1000;ids=np.arange(40,50)
        local=first[ids].clone();other=second[ids].clone()
        audio_context.register(local,first,ids);audio_context.register(other,second,ids)
        self.assertTrue(torch.equal(mapped(local,2,0),first[36:44]))
        self.assertTrue(torch.equal(mapped(local,2,9),first[45:53]))
        self.assertTrue(torch.equal(mapped(other,2,0),second[36:44]))
        self.assertFalse(torch.equal(mapped(local,2,0),original(local,2,0)))
        self.assertTrue(torch.equal(mapped(local,0,torch.tensor(7)),first[[47]]))


class ExecutionControls(unittest.TestCase):
    def test_coordinator_reconnects_completed_epoch_without_duplicate_model_work(self):
        import json,sys,tempfile
        from unittest.mock import patch
        sys.path.insert(0,str(path.parent))
        import training_coordinator as coordinator
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);remote={};executed=set();live={'identity':None};interrupt={'pending':True}
            import cohort_binding
            selected=dict(version=cohort_binding.VERSION,identities=cohort_binding.IDS,members={})
            for identity in cohort_binding.IDS:
                selected['members'][identity]=dict(revision=cohort_binding.REVISIONS[identity],
                    remote_root=cohort_binding.expected_root(cohort_binding.REMOTE,identity).as_posix(),
                    persistent_root=cohort_binding.expected_root(cohort_binding.FROOT,identity).as_posix())
            (root/'cohort_selection.json').write_text(json.dumps(selected))
            admission=dict(passed=True,training_ready=True,identities=coordinator.IDS,execution_contract_sha256='source',windows_controller_sources={},
                cohort_version=cohort_binding.VERSION,cohort_selection_sha256=cohort_binding.sha(root/'cohort_selection.json'))
            (root/'data_qualification.json').write_text(json.dumps(admission))
            def read_remote(name):return admission if name=='data_qualification.json' else dict(remote[name])
            def restore(identity,mode):
                self.assertIsNone(live['identity']);live['identity']=identity
                p=cohort_binding.expected_root(root,identity)/'derived_packages'/identity/'full_restoration_verification.json';p.parent.mkdir(parents=True);p.write_text('{"passed":true}')
            def handoff(identity):
                if identity!=coordinator.IDS[-1]:live['identity']=None
            def wait_job(identity,seed,gain,phase,epoch,prior,reference,log):
                relative=f'training/{identity}/seed{seed}/gain_{gain:+d}/{phase}/ep{epoch:04d}/attempt01'
                key=(identity,seed,gain,phase,epoch)
                if key not in executed:
                    executed.add(key)
                    remote[relative+'/epoch_receipt.json']=dict(passed=True,identity=identity,seed=seed,gain_db=gain,phase=phase,epoch=epoch,
                        checkpoint_path=f'/dev/shm/synctalk-multispeaker-data-20261017/{relative}/workspace/checkpoints/state.pth')
                    if len(executed)==70 and interrupt['pending']:
                        interrupt['pending']=False;raise RuntimeError('synthetic coordinator loss after committed remote work')
            def persist(relative):
                out=root/relative/'artifacts';out.mkdir(parents=True)
                (out/'epoch_receipt.json').write_text(json.dumps(remote[relative+'/epoch_receipt.json']))
                (out.parent/'verification.json').write_text('{"passed":true}');return dict(passed=True)
            def epoch(receipts,traces):
                self.assertEqual(set(receipts),{-6,0,6});base=receipts[0]
                return dict(passed=True,identity=base['identity'],seed=base['seed'],phase=base['phase'],epoch=base['epoch'])
            def phase(epochs,receipts):
                base=receipts[0];self.assertEqual(len(epochs),10 if base['phase']=='coarse' else 7)
                return dict(passed=True,identity=base['identity'],seed=base['seed'],phase=base['phase'])
            with patch.multiple(coordinator,ROOT=root,read_remote=read_remote,sha=lambda _: 'source',restore=restore,handoff=handoff,
                wait_job=wait_job,persist=persist,verify_persisted=lambda *a,**k:dict(passed=True),
                retire_consumed_previous=lambda *a:None,qualify_epoch=epoch,qualify_phase=phase,publish=lambda *a:None):
                with self.assertRaisesRegex(RuntimeError,'synthetic coordinator loss'):coordinator.run()
                self.assertEqual(len(executed),70);coordinator.run()
                self.assertEqual(len(executed),408)
                result=json.loads((root/'training_coordinator_receipt.json').read_text())
                self.assertTrue(result['passed']);self.assertEqual(result['coordinator_restarts'],1);self.assertEqual(len(result['blocks']),16)
                coordinator.run();self.assertEqual(len(executed),408)

    def test_coordinator_lock_excludes_second_process_and_releases(self):
        import os,subprocess,sys,tempfile
        sys.path.insert(0,str(path.parent))
        from execution_journal import exclusive_lock
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'owner.lock'
            code="from execution_journal import exclusive_lock;import sys\ntry:\n with exclusive_lock(sys.argv[1]):pass\nexcept OSError:sys.exit(23)"
            env=dict(os.environ);env['PYTHONPATH']=str(path.parent)+os.pathsep+env.get('PYTHONPATH','')
            with exclusive_lock(target):
                result=subprocess.run([sys.executable,'-c',code,str(target)],env=env,capture_output=True)
                self.assertEqual(result.returncode,23,result.stderr)
            result=subprocess.run([sys.executable,'-c',code,str(target)],env=env,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_detached_job_reconnects_without_replaying_command(self):
        import sys,tempfile,time
        sys.path.insert(0,str(path.parent))
        from remote_epoch_job import start_job,status
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);counter=root/'count.txt';directory=root/'job'
            command=[sys.executable,'-c',"import time;from pathlib import Path;time.sleep(.5);p=Path(%r);p.write_text('once');print('completed')"%str(counter)]
            first=start_job(directory,command);second=start_job(directory,command)
            self.assertEqual(first['command_sha256'],second['command_sha256'])
            deadline=time.monotonic()+10
            while True:
                state=status(directory)
                if state['state']=='completed':break
                self.assertIn(state['state'],['starting','running']);self.assertLess(time.monotonic(),deadline);time.sleep(.05)
            self.assertTrue(state['passed']);self.assertEqual(counter.read_text(),'once')
            self.assertEqual(start_job(directory,command),state)
            with self.assertRaises(AssertionError):start_job(directory,command+['different'])

    def test_independent_matrix_retains_original_epsilon_and_lags(self):
        import ast,sys
        import numpy as np,torch
        sys.path.insert(0,str(path.parent))
        from validate_scores import distance,endpoint
        original=Path('/home/wjj/tts-audio/tts-exp/third_party/syncnet_python/SyncNetInstance.py')
        if not original.exists():self.skipTest('requires original scoring source')
        node=next(x for x in ast.parse(original.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='calc_pdist')
        scope={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(original),'exec'),scope)
        rng=np.random.default_rng(23);vf=rng.normal(size=(245,1024)).astype(np.float32);af=vf+rng.normal(0,.01,size=vf.shape).astype(np.float32)
        actual=torch.stack(scope['calc_pdist'](torch.from_numpy(vf),torch.from_numpy(af),vshift=15)).numpy()
        rebuilt=distance(vf,af);self.assertLess(float(np.max(np.abs(actual-rebuilt))),3e-5)
        self.assertEqual(endpoint(rebuilt,np.arange(20,225))['av_offset'],0)
        self.assertEqual(endpoint(rebuilt,np.arange(245))['av_offset'],0)

    def test_formal_video_preserves_rgb_channels_and_original_pcm(self):
        import shutil,subprocess,sys,tempfile
        import cv2,numpy as np,soundfile as sf
        sys.path.insert(0,str(path.parent))
        if not shutil.which('ffmpeg'):self.skipTest('requires FFmpeg')
        from render_model import Video
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audio=root/'source.wav';movie=root/'formal.mkv'
            pcm=np.arange(4*640,dtype=np.int16)-1280;sf.write(audio,pcm,16000,subtype='PCM_16')
            writer=Video(movie,audio,frames=4);expected=[]
            for i in range(4):
                rgb=np.zeros((512,512,3),dtype=np.uint8);rgb[:,:,0]=17+i;rgb[:,:,1]=71;rgb[:,:,2]=223-i
                expected.append(rgb);writer.write(rgb)
            writer.close();cap=cv2.VideoCapture(str(movie))
            for rgb in expected:
                ok,bgr=cap.read();self.assertTrue(ok);self.assertTrue(np.array_equal(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),rgb))
            self.assertFalse(cap.read()[0]);cap.release()
            decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(movie),'-map','0:a:0','-f','s16le','pipe:1'])
            self.assertEqual(decoded,pcm.tobytes())

    def test_paired_budget_accepts_skips_then_rejects_unequal_phase(self):
        import copy,hashlib,json,sys,tempfile
        sys.path.insert(0,str(path.parent))
        from paired_budget import qualify_epoch,qualify_phase
        with tempfile.TemporaryDirectory() as tmp:
            receipts={};traces={}
            for gain in [-6,0,6]:
                rows=[];updates=0
                for i in range(6000):
                    success=not(gain==6 and i==12);updates+=int(success)
                    rows.append(dict(epoch=1,global_step=i+1,frame_index=i,source_audio_id=i+50,lip_mode=False,
                        pixel_index_sha256='same_real_measurement_placeholder',actual_rays=65536,rect=None,
                        successful_update=success,amp_skip=not success,successful_updates=updates))
                raw=('\n'.join(json.dumps(row) for row in rows)+'\n').encode();target=Path(tmp)/f'{gain}.jsonl';target.write_bytes(raw);traces[gain]=target
                receipts[gain]=dict(identity='control',seed=42,phase='coarse',epoch=1,gain_db=gain,global_step=6000,
                    passed=True,epoch_attempts=6000,protocol_sha256='p',execution_contract_sha256='e',epoch_order_sha256='o',sampling_sequence_sha256='s',
                    trace_sha256=hashlib.sha256(raw).hexdigest(),epoch_successful_updates=updates,epoch_amp_skips=6000-updates,
                    cumulative_successful_updates=updates,phase_successful_updates=updates)
            accepted=qualify_epoch(receipts,traces)
            self.assertTrue(accepted['passed']);self.assertFalse(accepted['actual_successful_updates_equal_this_epoch'])
            epochs=[]
            for epoch in range(1,11):
                row=copy.deepcopy(accepted);row['epoch']=epoch;epochs.append(row)
            for gain in [-6,0,6]:receipts[gain]['phase_successful_updates']*=10
            result=qualify_phase(epochs,receipts)
            self.assertFalse(result['passed']);self.assertEqual(result['status'],'CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES')
            # A valid file hash cannot conceal a different measured sampling sequence.
            changed=[json.loads(x) for x in traces[6].read_text().splitlines()];changed[17]['pixel_index_sha256']='different'
            raw=('\n'.join(json.dumps(row) for row in changed)+'\n').encode();traces[6].write_bytes(raw)
            receipts[6]['trace_sha256']=hashlib.sha256(raw).hexdigest()
            with self.assertRaises(AssertionError):qualify_epoch(receipts,traces)

    def test_original_full_checkpoint_resume_matches_uninterrupted_toy(self):
        import copy,tempfile
        import numpy as np,torch
        from types import SimpleNamespace
        state=self.state_module()
        try:
            from controlled_training import ControlledTrainer
        except ModuleNotFoundError:self.skipTest('requires complete official runtime')
        if not torch.cuda.is_available():self.skipTest('requires actual CUDA FP16')
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.02,.03,.04]))
                self.register_buffer('density_grid',torch.arange(4,dtype=torch.float32))
                self.cuda_ray=False;self.mean_count=0;self.mean_density=.1;self.mean_density_torso=0.;self.iter_density=0;self.local_step=0
        class Toy(ControlledTrainer):
            def train_step(self,data):
                self.model.iter_density+=1;self.model.local_step+=1
                # Small FP32 expression inside the unchanged original autocast loop.
                target=torch.tensor(data,dtype=torch.float32,device='cuda')
                loss=(self.model.weight-target).square().mean()
                return self.model.weight,target,loss
        class Loader:
            batch_size=1
            def __len__(self):return 12
            def __iter__(self):return iter([[.01,.02,.03]]*12)
        def make(folder,phase='coarse',checkpoint='scratch'):
            opt=SimpleNamespace(finetune_lips=phase=='fine',init_lips=False,patch_size=1,update_extra_interval=16)
            model=Model()
            factor=.5 if phase=='coarse' else .05;budget=60000 if phase=='coarse' else 100000
            return Toy('toy',opt,model,device=torch.device('cuda'),workspace=str(folder),fp16=True,mute=True,
                optimizer=lambda m:torch.optim.AdamW(m.parameters(),lr=.01,betas=(0,.99),eps=1e-8),
                lr_scheduler=lambda o:torch.optim.lr_scheduler.LambdaLR(o,lambda step:factor**(step/budget)),
                ema_decay=.95,use_checkpoint=checkpoint,use_tensorboardX=False,scheduler_update_every_step=True)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);continuous=make(root/'continuous');continuous.global_step=988
            continuous.epoch=1;continuous.train_one_epoch(Loader())
            continuous.save_checkpoint(name='boundary',full=True,remove_old=False)
            checkpoint=root/'continuous/checkpoints/boundary.pth'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            counters=state.model_counters(continuous.model)
            # Match real post-save validation: EMA collected_params are refreshed.
            continuous.ema.store();continuous.ema.copy_to();continuous.ema.restore()
            continuous.stats['valid_loss'].append(.125);continuous.stats['results'].append(.125)
            resumed=make(root/'resumed',checkpoint=str(checkpoint));state.verify_full_checkpoint(resumed,saved)
            self.assertEqual(resumed.model.iter_density,0)
            state.restore_model_counters(resumed.model,counters);resumed.stats=copy.deepcopy(continuous.stats)
            resumed.ema.store();self.assertEqual(state.tree_sha(resumed.ema.state_dict()),state.tree_sha(continuous.ema.state_dict()))
            for trainer in [continuous,resumed]:trainer.epoch=2;trainer.train_one_epoch(Loader())
            for name in ['model','optimizer','lr_scheduler','scaler','ema']:
                self.assertEqual(state.tree_sha(getattr(continuous,name).state_dict()),state.tree_sha(getattr(resumed,name).state_dict()),name)
            self.assertEqual(state.tree_sha(state.model_counters(continuous.model)),state.tree_sha(state.model_counters(resumed.model)))
            self.assertEqual(continuous.stats,resumed.stats);self.assertEqual(continuous.global_step,resumed.global_step)
            # Official phase switch retains own coarse LR/state while the new lambda is fine.
            fine=make(root/'fine',phase='fine',checkpoint=str(checkpoint));state.verify_full_checkpoint(fine,saved)
            self.assertEqual(fine.model.iter_density,0);self.assertEqual(fine.model.local_step,0)
            self.assertEqual(fine.optimizer.param_groups[0]['lr'],saved['optimizer']['param_groups'][0]['lr'])
            fine.epoch=2;fine.train_one_epoch(Loader())
            self.assertAlmostEqual(fine.optimizer.param_groups[0]['lr'],.01*.05**(fine.lr_scheduler.last_epoch/100000),places=14)

    def state_module(self):
        import sys
        sys.path.insert(0,str(path.parent))
        import execution_state
        return execution_state

    def test_validation_rng_and_renderer_counters_restore_exactly(self):
        import random
        import numpy as np,torch
        from types import SimpleNamespace
        state=self.state_module()
        before=state.tree_sha(state.rng_state())
        with state.preserve_rng():
            random.random();np.random.normal(size=100);torch.rand(10)
            if torch.cuda.is_available():torch.rand(10,device='cuda')
        self.assertEqual(before,state.tree_sha(state.rng_state()))
        model=SimpleNamespace(mean_count=123,mean_density=.23,mean_density_torso=0.,iter_density=375,local_step=6000)
        saved=state.model_counters(model);model.iter_density=0;model.local_step=0
        state.restore_model_counters(model,saved)
        self.assertEqual(state.tree_sha(saved),state.tree_sha(state.model_counters(model)))

    def test_actual_optimizer_hook_excludes_real_amp_overflow_skip(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('requires actual CUDA GradScaler')
        state=self.state_module()
        weight=torch.nn.Parameter(torch.ones(3,device='cuda'))
        optimizer=torch.optim.AdamW([weight],lr=.01,betas=(0,.99),eps=1e-8)
        counter=state.UpdateCounter(optimizer);scaler=torch.cuda.amp.GradScaler()
        scaler.scale(weight.square().sum()).backward();scaler.step(optimizer);scaler.update()
        self.assertEqual(counter.successful,1);after=weight.detach().clone()
        optimizer.zero_grad();scaler.scale((weight*float('inf')).sum()).backward();scaler.step(optimizer);scaler.update()
        self.assertEqual(counter.successful,1);self.assertTrue(torch.equal(after,weight));counter.close()

    def test_final_ema_snapshot_clones_before_restore(self):
        import sys,tempfile
        import torch
        sys.path.insert(0,str(path.parent))
        try:
            from controlled_training import ControlledTrainer
            from torch_ema import ExponentialMovingAverage
        except ModuleNotFoundError:self.skipTest('requires complete official runtime')
        state=self.state_module();model=torch.nn.Linear(3,2)
        model.register_buffer('density_grid',torch.arange(4,dtype=torch.float32))
        trainer=object.__new__(ControlledTrainer);trainer.log_ptr=None;trainer.model=model
        trainer.epoch=17;trainer.global_step=102000;trainer.ema=ExponentialMovingAverage(model.parameters(),decay=.95)
        for p in trainer.ema.shadow_params:p.fill_(.123)
        before=state.tree_sha(model.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'ema.pt';trainer.save_final_ema(target)
            snapshot=torch.load(target,weights_only=False)
            for name,parameter in model.named_parameters():self.assertTrue(torch.equal(snapshot['model'][name],torch.full_like(parameter,.123)))
            self.assertTrue(torch.equal(snapshot['model']['density_grid'],model.density_grid))
        self.assertEqual(before,state.tree_sha(model.state_dict()))

if __name__=='__main__':unittest.main()
