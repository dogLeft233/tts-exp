"""Bounded CoGenAV-to-waveform pilot: frozen feature cache, warmup, joint reconstruction."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from common import (Adapter,CONTEXT25,HOP25,SEGMENT25,SEED,SR,SpectralLoss,configure,
                    context_crop,dump,feature_loss,freeze_generator,resource_gate,sha,teacher_grid)

REPO=Path('/home/wjj/tts-audio/tts-exp')
DATA=REPO/'data/aishell3_decoder_5h_20261025_attempt01'
COGEN=REPO/'runs/cogenav_sync_audit_20261024_attempt01'
KNN=REPO/'third_party/knn-vc'


def protocol_file(run):
    latest=run/'protocol_revision03.json'
    if latest.exists():return latest
    revised=run/'protocol_revision02.json'
    return revised if revised.exists() else run/'protocol.json'


def schedule_file(run):
    latest=run/'schedule_revision03.json'
    return latest if latest.exists() else run/'schedule_frozen.json'


def load_rows():
    rows=[]
    for split in ('train','validation'):
        rows.extend(json.loads(line) for line in (DATA/(split+'.jsonl')).read_text(encoding='utf-8').splitlines())
    for r in rows:r['id']=Path(r['audio_path']).stem
    assert len({r['id'] for r in rows})==len(rows)
    assert len(rows)==4631
    return rows


def monitor_ids(rows):
    val=[r for r in rows if r['split']=='validation']
    ids=[]
    for speaker in sorted({r['speaker'] for r in val}):
        ids.extend(r['id'] for r in sorted((r for r in val if r['speaker']==speaker),
            key=lambda r:hashlib.sha256(('monitor:'+r['member']).encode()).hexdigest())[:4])
    return ids


def prepare(run,generator_checkpoint):
    run.mkdir(parents=True,exist_ok=True)
    protocol=run/'protocol.json'
    if protocol.exists():raise FileExistsError(protocol)
    rows=load_rows()
    pins={str(p):sha(p) for p in [DATA/'file_manifest.jsonl',DATA/'train.jsonl',DATA/'validation.jsonl',
        COGEN/'weights/base_cogen_mirror.pt',COGEN/'base_resolved.yaml',Path(generator_checkpoint),
        Path.home()/'.cache/torch/hub/checkpoints/WavLM-Large.pt',KNN/'hifigan/models.py',
        KNN/'hifigan/config_v1_wavlm.json',KNN/'wavlm/WavLM.py',COGEN/'upstream/models/cogenav.py']}
    assert pins[str(COGEN/'weights/base_cogen_mirror.pt')]=='90f8c5165282c190c81c23d8fe92db2aa700862e021b1312fb96779caf9cfe61'
    dump(protocol,dict(seed=SEED,status='PREPARED',generator_checkpoint=str(generator_checkpoint),pins=pins,
        source_split='official AISHELL-3 test repurposed; custom split isolated',
        train_ids=[r['id'] for r in rows if r['split']=='train'],monitor_ids=monitor_ids(rows),
        validation_ids=[r['id'] for r in rows if r['split']=='validation'],
        encoder='frozen official CoGenAV Base audio-only pre-upsampler FP32 raw (no L2 normalization)',
        teacher='official WavLM Large output_layer6; no input normalization or VAD; native400/320 grid',
        clock='CoGen ceil(N/640) frames native40ms grid; adapter2x align_corners=False; teacher native50Hz unchanged, replicate/mask missing tail; waveform right-pad for loss only; infer crop to exactN',
        adapter='fixed train-only per-dimension affine standardization + 512-channel temporal residual convs dilations1/2/4 ->1024D at50Hz',
        waveform='same prepared PCM as target, no loudness normalization or silence trimming',
        loss='stage1 teacherMSE plus0.1logMel+0.05MRSTFT every4 updates through frozenG; stage2 0.1teacherMSE+10logMel+MRSTFT every update; NO adversarial/feature-matching fine-tuning',
        crop=dict(center_frames25=SEGMENT25,context_frames25=CONTEXT25),
        budget=dict(warmup_updates=2000,joint_updates=4000,batch=4,accumulation=2,
                    stage1_adapter_lr=2e-4,stage2_adapter_lr=1e-4,stage2_generator_lr=1e-5),
        gpu_policy='shared GPU; <=7.5GiB allocator cap; enter phases only free>=10GiB; pause external footprint>6GiB',
        validation='Fixed16 balanced monitor for model selection by waveform logMel only; final all429; no SyncNet or downstream optimization',
        prepared_before_training=True))
    dump(run/'rows.json',rows)
    print(json.dumps(dict(prepared=True,monitor_ids=monitor_ids(rows))),flush=True)


def verify_pins(run):
    protocol=json.loads(protocol_file(run).read_text())
    for path,digest in protocol['pins'].items():
        if sha(path)!=digest:raise ValueError('Pinned input changed: '+path)
    return protocol


def load_generator(run,device):
    protocol=json.loads(protocol_file(run).read_text())
    sys.path.insert(0,str(KNN))
    from hifigan.models import Generator
    from hifigan.utils import AttrDict
    config=AttrDict(json.loads((KNN/'hifigan/config_v1_wavlm.json').read_text()))
    model=Generator(config)
    checkpoint=torch.load(protocol['generator_checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['generator'],strict=True)
    return model.to(device)


def cache(run,limit=None):
    configure()
    verify_pins(run)
    resource_gate(run,'cache',initial=True)
    from transformers import WhisperFeatureExtractor
    sys.path.insert(0,str(COGEN/'upstream'))
    from models.cogenav import CoGenAV
    sys.path.insert(0,str(KNN))
    from wavlm.WavLM import WavLM,WavLMConfig
    cogen=CoGenAV(str(COGEN/'base_resolved.yaml'),model_tensor=None)
    cogen.load_state_dict(torch.load(COGEN/'weights/base_cogen_mirror.pt',map_location='cpu',weights_only=True),strict=True)
    cogen.eval().requires_grad_(False).cuda()
    checkpoint=torch.load(Path.home()/'.cache/torch/hub/checkpoints/WavLM-Large.pt',map_location='cpu',weights_only=False)
    wavlm=WavLM(WavLMConfig(checkpoint['cfg']))
    wavlm.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint
    wavlm.eval().requires_grad_(False).cuda()
    extractor=WhisperFeatureExtractor()
    rows=json.loads((run/'rows.json').read_text())
    cache_root=run/'cache';cache_root.mkdir(exist_ok=True)
    total_count=0
    sums=torch.zeros(768,dtype=torch.float64);squares=sums.clone()
    receipts=[]
    started=time.monotonic()
    for i,row in enumerate(rows[:limit] if limit else rows):
        target=cache_root/(row['id']+'.pt')
        if target.exists():raise FileExistsError(target)
        path=DATA/row['audio_path']
        if sha(path)!=row['output_sha256']:raise ValueError('Source PCM changed')
        waveform,sr=sf.read(path,dtype='float32')
        if sr!=SR or len(waveform)!=row['output_frames']:raise ValueError('Input shape/rate')
        n25=math.ceil(len(waveform)/HOP25)
        with torch.inference_mode():
            mel=torch.from_numpy(extractor(waveform,sampling_rate=SR).input_features[0])[None].cuda()
            a=cogen(None,mel,use_upsampler=False)[0,:n25].float().cpu()
            native=wavlm.extract_features(torch.from_numpy(waveform)[None].cuda(),
                output_layer=6,ret_layer_results=False)[0][0].float().cpu()
            teacher,mask=teacher_grid(native,n25)
        assert a.shape==(n25,768) and teacher.shape==(2*n25,1024)
        assert torch.isfinite(a).all() and torch.isfinite(teacher).all()
        value=dict(id=row['id'],cogen=a,teacher=teacher,teacher_mask=mask,
            waveform_samples=len(waveform),source_sha256=row['output_sha256'],
            native_teacher_frames=len(native),waveform_path=str(path))
        torch.save(value,target)
        if row['split']=='train':
            sums+=a.double().sum(0);squares+=a.double().square().sum(0);total_count+=len(a)
        receipts.append(dict(id=row['id'],path=str(target),sha256=sha(target),frames25=n25,
                             teacher_frames=len(native),split=row['split']))
        if (i+1)%20==0 or i==0:
            state=dict(stage='cache',done=i+1,total=len(rows),elapsed_s=time.monotonic()-started,
                allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
            dump(run/'cache_status.json',state);print(json.dumps(state),flush=True)
        if (i+1)%100==0:resource_gate(run,'cache_'+str(i+1))
    mean=sums/total_count
    std=(squares/total_count-mean.square()).clamp_min(1e-8).sqrt()
    torch.save(dict(mean=mean.float(),std=std.float(),train_frames=total_count),run/'normalization.pt')
    dump(run/('cache_smoke_receipt.json' if limit else 'cache_complete.json'),
         dict(status='PASS',utterances=len(receipts),rows=receipts,elapsed_s=time.monotonic()-started))


class Samples:
    def __init__(self,run,rows):
        self.run,self.rows=run,rows
        self.rng=np.random.default_rng(SEED)
        self.order=[]
        self.seen=Counter()

    def batch(self,size):
        values=[]
        for _ in range(size):
            if not self.order:self.order=list(self.rng.permutation(len(self.rows)))
            row=self.rows[self.order.pop()]
            value=torch.load(self.run/'cache'/(row['id']+'.pt'),map_location='cpu',weights_only=True)
            wave=torch.from_numpy(sf.read(value['waveform_path'],dtype='float32')[0])
            start=int(self.rng.integers(0,len(value['cogen'])-SEGMENT25+1))
            values.append(context_crop(value['cogen'],value['teacher'],value['teacher_mask'],wave,start))
            self.seen[row['id']]+=1
        return tuple(torch.stack(x).cuda(non_blocking=True) for x in zip(*values))


def norm_adapter(run):
    stats=torch.load(run/'normalization.pt',map_location='cpu',weights_only=True)
    return Adapter(stats['mean'],stats['std']).cuda()


def output_wave(generator,adapted,n):
    wave=generator(adapted).squeeze(1)
    if wave.shape[-1]<n:wave=F.pad(wave,(0,n-wave.shape[-1]))
    return wave[:,:n]


def reconstruct(adapter,generator,cogen,n):
    """Use the same replicated conditioning context as every training crop, including edges."""
    extended=F.pad(cogen.transpose(1,2),(CONTEXT25,CONTEXT25),mode='replicate').transpose(1,2)
    generated=generator(adapter(extended)).squeeze(1)
    start=CONTEXT25*HOP25
    output=generated[:,start:start+n]
    if output.shape[-1]!=n:raise ValueError('Full reconstruction duration mismatch')
    return output


@torch.inference_mode()
def validate(run,adapter,generator,criterion,ids,save_dir=None,oracle=False):
    adapter.eval();generator.eval()
    scores=[]
    for id in ids:
        value=torch.load(run/'cache'/(id+'.pt'),map_location='cpu',weights_only=True)
        target=torch.from_numpy(sf.read(value['waveform_path'],dtype='float32')[0]).cuda()[None]
        if oracle:
            wave=output_wave(generator,value['teacher'].cuda()[None],target.shape[-1])
        else:
            wave=reconstruct(adapter,generator,value['cogen'].cuda()[None],target.shape[-1])
        mel,stft=criterion(wave,target)
        scores.append(dict(id=id,logmel=float(mel),mrstft=float(stft),samples=wave.shape[-1],
                           peak=float(wave.abs().max()),rms=float(wave.square().mean().sqrt())))
        if save_dir:
            save_dir.mkdir(parents=True,exist_ok=True)
            sf.write(save_dir/(id+'.wav'),wave[0].cpu().numpy(),SR,subtype='FLOAT')
    return dict(logmel=float(np.mean([r['logmel'] for r in scores])),
                mrstft=float(np.mean([r['mrstft'] for r in scores])),rows=scores)


def smoke(run,output_name='smoke.json'):
    configure();verify_pins(run);resource_gate(run,'smoke',initial=True)
    adapter=norm_adapter(run);generator=load_generator(run,'cuda');criterion=SpectralLoss().cuda()
    rows=[r for r in json.loads((run/'rows.json').read_text()) if r['split']=='train']
    samples=Samples(run,rows)
    timings=[]
    for frozen in (True,False):
        freeze_generator(generator,frozen)
        opt=torch.optim.AdamW(list(adapter.parameters())+([] if frozen else list(generator.parameters())),lr=1e-4)
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for i in range(4):
            a,target,mask,wave=samples.batch(4)
            pred=adapter(a)
            center=pred[:,CONTEXT25*2:(CONTEXT25+SEGMENT25)*2]
            generated=generator(pred).squeeze(1)[:,CONTEXT25*HOP25:(CONTEXT25+SEGMENT25)*HOP25]
            mel,mrstft=criterion(generated,wave)
            loss=feature_loss(center,target,mask)+mel+mrstft
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),10)
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None)
            if frozen:assert all(p.grad is None for p in generator.parameters())
            opt.step()
        torch.cuda.synchronize()
        timings.append(dict(frozen_generator=frozen,batch=4,updates=4,elapsed_s=time.monotonic()-started,
                            peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,last_loss=float(loss)))
        del opt
    dump(run/output_name,dict(status='PASS',timings=timings,context25=CONTEXT25,
                             scope='discarded smoke states, no main training updates'))
    print(json.dumps(timings),flush=True)


def checkpoint(run,name,adapter,generator,opt,stage,step,score):
    path=run/'checkpoints'/name;path.parent.mkdir(exist_ok=True)
    temp=path.with_suffix('.writing')
    torch.save(dict(adapter=adapter.state_dict(),generator=generator.state_dict(),optimizer=opt.state_dict(),
                    stage=stage,step=step,score=score,protocol_sha256=sha(protocol_file(run))),temp)
    temp.replace(path)


def train(run):
    configure();protocol=verify_pins(run)
    resource_gate(run,'train',initial=True)
    if not schedule_file(run).exists():raise ValueError('Parent-approved smoke timing and schedule required')
    schedule=json.loads(schedule_file(run).read_text())
    bound_protocol=schedule.get('effective_protocol_sha256',schedule.get('protocol_sha256'))
    if bound_protocol!=sha(protocol_file(run)):
        raise ValueError('Frozen schedule is not bound to effective protocol revision')
    for key,value in dict(warmup_updates=2000,joint_updates=4000,batch=4,accumulation=2,context25=CONTEXT25).items():
        if schedule.get(key)!=value:raise ValueError('Frozen schedule does not match executable: '+key)
    if (run/'training_status.json').exists():raise FileExistsError('Main training already started; explicit new resume needed')
    adapter=norm_adapter(run);generator=load_generator(run,'cuda');criterion=SpectralLoss().cuda()
    baseline=validate(run,adapter,generator,criterion,protocol['monitor_ids'],run/'audio/untrained_monitor')
    dump(run/'untrained_monitor.json',baseline)
    oracle=validate(run,adapter,generator,criterion,protocol['monitor_ids'],run/'audio/oracle_monitor',oracle=True)
    dump(run/'oracle_monitor.json',oracle)
    quality_path=run/'parent_quality_selection.json'
    quality=json.loads(quality_path.read_text(encoding='utf-8'))
    quality_ids=[Path(r['audio_path']).stem for r in quality['rows']]
    assert len(quality_ids)==40 and not set(quality_ids)&set(protocol['monitor_ids'])
    dump(run/'diagnostic_subset_receipt.json',dict(selection_sha256=sha(quality_path),ids=quality_ids,
                                                  selection='parent fixed before results; excluded training monitor'))
    dump(run/'untrained_quality.json',validate(run,adapter,generator,criterion,quality_ids,run/'audio/untrained_quality'))
    dump(run/'oracle_quality.json',validate(run,adapter,generator,criterion,quality_ids,run/'audio/oracle_quality',oracle=True))
    rows=[r for r in json.loads((run/'rows.json').read_text()) if r['split']=='train']
    best=float('inf');global_step=0;started=time.monotonic()
    log=(run/'train_metrics.jsonl').open('x')
    for stage,updates in [('warmup',2000),('joint',4000)]:
        frozen=stage=='warmup';freeze_generator(generator,frozen)
        groups=[dict(params=adapter.parameters(),lr=2e-4 if frozen else 1e-4)]
        if not frozen:groups.append(dict(params=generator.parameters(),lr=1e-5))
        opt=torch.optim.AdamW(groups,betas=(.8,.99))
        samples=Samples(run,rows)
        for step in range(1,updates+1):
            adapter.train();generator.eval() if frozen else generator.train()
            opt.zero_grad(set_to_none=True);metrics=[]
            waveform_step=not frozen or step%4==0
            for accumulation in range(2):
                a,target,mask,wave=samples.batch(4)
                pred=adapter(a)
                center=pred[:,CONTEXT25*2:(CONTEXT25+SEGMENT25)*2]
                feat=feature_loss(center,target,mask)
                if waveform_step:
                    generated=generator(pred).squeeze(1)[:,CONTEXT25*HOP25:(CONTEXT25+SEGMENT25)*HOP25]
                    mel,mrstft=criterion(generated,wave)
                    loss=feat+.1*mel+.05*mrstft if frozen else .1*feat+10*mel+mrstft
                else:
                    mel=mrstft=feat.detach()*0;loss=feat
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                (loss/2).backward()
                metrics.append([float(loss.detach()),float(feat.detach()),float(mel.detach()),float(mrstft.detach())])
            parameters=list(adapter.parameters())+([] if frozen else list(generator.parameters()))
            if any(not torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):raise ValueError('Nonfinite gradient')
            gradnorm=float(torch.nn.utils.clip_grad_norm_(parameters,10))
            opt.step();global_step+=1
            if step==1 or step%20==0:
                vals=np.mean(metrics,axis=0)
                record=dict(stage=stage,step=step,global_step=global_step,loss=float(vals[0]),teacher_mse=float(vals[1]),
                    logmel=float(vals[2]),mrstft=float(vals[3]),waveform_step=waveform_step,gradient_norm=gradnorm,
                    elapsed_s=time.monotonic()-started,unique_training_utterances=len(samples.seen),
                    allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
                log.write(json.dumps(record)+'\n');log.flush();dump(run/'training_status.json',record)
                print(json.dumps(record),flush=True)
            if step%100==0:resource_gate(run,stage+'_'+str(step))
            if step%500==0 or step==updates:
                score=validate(run,adapter,generator,criterion,protocol['monitor_ids'])
                dump(run/f'validation_{stage}_{step:05d}.json',score)
                if score['logmel']<best:
                    best=score['logmel'];checkpoint(run,'best.pt',adapter,generator,opt,stage,step,score)
                checkpoint(run,'last.pt',adapter,generator,opt,stage,step,score)
        dump(run/(stage+'_coverage.json'),dict(unique=len(samples.seen),total=len(rows),
            min_exposures=min(samples.seen.values()),max_exposures=max(samples.seen.values()),counts=dict(samples.seen)))
        assert len(samples.seen)==len(rows)
        if frozen:checkpoint(run,'warmup.pt',adapter,generator,opt,stage,updates,score)
        del opt
    log.close()
    dump(run/'training_complete.json',dict(status='PASS',warmup_updates=2000,joint_updates=4000,
        best_monitor_logmel=best,elapsed_s=time.monotonic()-started,adversarial_finetuning=False))


def infer(run):
    configure();protocol=verify_pins(run);resource_gate(run,'infer',initial=True)
    if not (run/'training_complete.json').exists():raise ValueError('Main training incomplete')
    adapter=norm_adapter(run);generator=load_generator(run,'cuda');criterion=SpectralLoss().cuda()
    cp=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    adapter.load_state_dict(cp['adapter'],strict=True);generator.load_state_dict(cp['generator'],strict=True)
    scores=validate(run,adapter,generator,criterion,protocol['validation_ids'],run/'audio/reconstructed_validation')
    dump(run/'final_validation.json',dict(status='PASS',checkpoint_sha256=sha(run/'checkpoints/best.pt'),
        checkpoint_stage=cp['stage'],checkpoint_step=cp['step'],**scores))
    print(json.dumps(dict(status='PASS',logmel=scores['logmel'],mrstft=scores['mrstft'],count=len(scores['rows']))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','cache','smoke','smoke_revision02','train','infer'])
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--generator-checkpoint')
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.stage=='prepare':prepare(args.run,args.generator_checkpoint)
    elif args.stage=='cache':cache(args.run,args.limit)
    elif args.stage.startswith('smoke'):smoke(args.run,args.stage+'.json')
    elif args.stage=='train':train(args.run)
    elif args.stage=='infer':infer(args.run)
