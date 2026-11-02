"""Outer controls around pinned Trainer; loss, backward, optimizer and EMA stay original."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from runtime_common import configure,seed_domain,tensor_state_sha
from execution_state import UpdateCounter,model_counters,preserve_rng,rng_state,restore_rng,tree_sha
configure()
from nerf_triplane.utils import Trainer
from nerf_triplane import provider

class ControlledLoader:
    batch_size=1
    has_gt=True
    def __init__(self,data,trainer,seed):self._data=data;self.trainer=trainer;self.seed=seed
    def __len__(self):return len(self._data.poses)
    def __iter__(self):
        assert len(self)==6000,'complete official epoch must contain6000 frames'
        epoch=self.trainer.epoch
        order_seed=int.from_bytes(hashlib.sha256(f'{self.seed}/{epoch}/frame_order'.encode()).digest()[:8],'little')%(2**63-1)
        generator=torch.Generator().manual_seed(order_seed)
        order=torch.randperm(len(self),generator=generator).tolist()
        self.trainer.epoch_order_sha256=hashlib.sha256(np.array(order,dtype='<u4').tobytes()).hexdigest()
        original=provider.get_rays
        for frame in order:
            captured=[]
            def observe(*args,**kwargs):
                result=original(*args,**kwargs)
                indexes=result['inds'].detach().cpu().numpy()
                assert (indexes>=0).all() and (indexes<512*512).all()
                captured.append(dict(pixel_index_sha256=hashlib.sha256(indexes.astype('<u4').tobytes()).hexdigest(),
                    actual_rays=int(indexes.size),rect=copy.deepcopy(kwargs.get('rect'))))
                return result
            provider.get_rays=observe
            seed_domain(self.seed,self.trainer.global_step+1,'pixel_ray_patch_rectangle')
            try:data=self._data.collate([frame])
            finally:provider.get_rays=original
            assert len(captured)==1
            self.trainer.current_sampling=dict(frame_index=frame,source_audio_id=int(self._data.original_source_ids[frame]),
                lip_mode=bool(self._data.opt.finetune_lips),**captured[0])
            yield data

class ControlledTrainer(Trainer):
    def attach_controls(self,seed,trace,initial_updates=0,reference_trace=None):
        self.control_seed=seed;self.trace=Path(trace).open('x',encoding='utf-8')
        self.update_counter=UpdateCounter(self.optimizer,initial_updates)
        self.current_sampling={};self.last_loss=None;self.epoch_order_sha256=None
        self.reference_rows=iter([json.loads(x) for x in Path(reference_trace).read_text().splitlines()]) if reference_trace else None
        self.sampling_hasher=hashlib.sha256()
        original_occupancy=self.model.update_extra_state
        def occupancy(*args,**kwargs):
            with preserve_rng():
                seed_domain(seed,self.global_step,'occupancy_randomness')
                return original_occupancy(*args,**kwargs)
        self.model.update_extra_state=occupancy
        original_step=self.scaler.step;original_update=self.scaler.update
        def step(optimizer,*args,**kwargs):
            self._updates_before=self.update_counter.successful;self._scale_before=float(self.scaler.get_scale())
            return original_step(optimizer,*args,**kwargs)
        def update(*args,**kwargs):
            result=original_update(*args,**kwargs)
            delta=self.update_counter.successful-self._updates_before;assert delta in [0,1]
            row=dict(epoch=self.epoch,global_step=self.global_step,loss=float(self.last_loss),
                successful_update=bool(delta),successful_updates=self.update_counter.successful,
                amp_skip=not bool(delta),scale_before=self._scale_before,scale_after=float(self.scaler.get_scale()),
                optimizer_lr=[float(x['lr']) for x in self.optimizer.param_groups],**self.current_sampling)
            assert np.isfinite(row['loss'])
            fields=['epoch','global_step','frame_index','source_audio_id','lip_mode','pixel_index_sha256','actual_rays','rect']
            sampling={k:row[k] for k in fields}
            if self.reference_rows is not None:
                reference=next(self.reference_rows)
                assert sampling=={k:reference[k] for k in fields},'actual frame/ray/patch sequence differs across gains'
            self.sampling_hasher.update(json.dumps(sampling,sort_keys=True,separators=(',',':')).encode()+b'\n')
            self.trace.write(json.dumps(row,separators=(',',':'))+'\n')
            if self.global_step%1000==0:self.trace.flush()
            return result
        self.scaler.step=step;self.scaler.update=update

    def train_step(self,data):
        seed_domain(self.control_seed,self.global_step,'render_perturbation_and_regularization')
        result=super().train_step(data)
        self.last_loss=float(result[2].detach())
        return result

    def validate_all_gains(self,datasets,output):
        output=Path(output);output.mkdir(exist_ok=False)
        before=dict(model=tensor_state_sha(self.model.state_dict()),optimizer=tree_sha(self.optimizer.state_dict()),
            scheduler=tree_sha(self.lr_scheduler.state_dict()),scaler=tree_sha(self.scaler.state_dict()),
            ema_shadow=tree_sha(self.ema.shadow_params),ema_updates=self.ema.num_updates)
        rows=[];self.model.eval()
        with preserve_rng():
            self.ema.store();self.ema.copy_to()
            try:
                for gain,data in datasets.items():
                    assert len(data.poses)==750
                    losses=[]
                    for i in range(750):
                        item=data.collate([i])
                        with torch.no_grad(),torch.cuda.amp.autocast(enabled=self.fp16):
                            result=super().eval_step(item)
                        loss=float(result[6]);assert np.isfinite(loss);losses.append(loss)
                        if gain==0 and i in [0,375,749]:
                            import cv2
                            rgb=(result[0][0].detach().float().cpu().numpy()*255).clip(0,255).astype(np.uint8)
                            assert cv2.imwrite(str(output/f'original_driver_frame{i:04d}.jpg'),rgb[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,90])
                    array=np.array(losses,dtype=np.float64);np.save(output/f'validation_{gain:+d}dB_mse.npy',array)
                    rows.append(dict(gain_db=gain,frames=750,mean_mse=float(array.mean())))
            finally:self.ema.restore()
        after=dict(model=tensor_state_sha(self.model.state_dict()),optimizer=tree_sha(self.optimizer.state_dict()),
            scheduler=tree_sha(self.lr_scheduler.state_dict()),scaler=tree_sha(self.scaler.state_dict()),
            ema_shadow=tree_sha(self.ema.shadow_params),ema_updates=self.ema.num_updates)
        assert before==after,'validation modified training state'
        primary=next(x['mean_mse'] for x in rows if x['gain_db']==0)
        self.stats['valid_loss'].append(primary);self.stats['results'].append(primary)
        receipt=dict(epoch=self.epoch,global_step=self.global_step,rows=rows,primary_gain_db=0,
            train_state_unchanged=True,rng_restored_exactly=True,checkpoint_selection=False)
        (output/'validation_receipt.json').write_text(json.dumps(receipt,indent=2));return receipt

    def save_final_ema(self,path):
        before=tensor_state_sha(self.model.state_dict());self.ema.store();self.ema.copy_to()
        try:
            state={name:value.detach().cpu().clone() for name,value in self.model.state_dict().items()}
            checkpoint=dict(model=state,epoch=self.epoch,global_step=self.global_step,model_counters=model_counters(self.model),
                ema=True,ema_updates=self.ema.num_updates,model_tensor_sha256=tensor_state_sha(state))
            torch.save(checkpoint,path)
        finally:self.ema.restore()
        assert before==tensor_state_sha(self.model.state_dict());return checkpoint['model_tensor_sha256']

    def close_controls(self):
        if self.reference_rows is not None:assert next(self.reference_rows,None) is None,'reference trace has unconsumed attempts'
        self.trace.flush();self.trace.close();self.update_counter.close()
