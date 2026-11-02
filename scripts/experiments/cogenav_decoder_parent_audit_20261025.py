"""Independent end-state audit of the fixed decoder pilot; never trains a model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def expected_coverage(rows, updates):
    import numpy as np
    rows=[row for row in rows if row['split']=='train']
    rng=np.random.default_rng(20261025)
    order=[]
    counts=Counter()
    for _ in range(updates*8):
        if not order:
            order=list(rng.permutation(len(rows)))
        row=rows[order.pop()]
        n25=(row['output_frames']+639)//640
        rng.integers(0,n25-11+1)
        counts[row['id']]+=1
    return dict(counts)


def expected_logged_steps():
    return ([s for s in range(1,2001) if s==1 or s%20==0]+
            [2000+s for s in range(1,4001) if s==1 or s%20==0])


def audit(root):
    import numpy as np
    import soundfile as sf
    import torch
    root=Path(root)
    torch.set_num_threads(4)
    protocol=root/'protocol_revision03.json'
    protocol_sha=sha(protocol)
    assert protocol_sha=='d50671c24c7da3020ae279052d4dfa1c25b18be1be00d4ab1141589913a76c75'
    spec=json.loads(protocol.read_text())
    for name in ('train','infer'):
        assert json.loads((root/(name+'_exit.json')).read_text())['exit_code']==0
    complete=json.loads((root/'training_complete.json').read_text())
    assert complete['warmup_updates']==2000 and complete['joint_updates']==4000
    assert complete['adversarial_finetuning'] is False
    rows=json.loads((root/'rows.json').read_text())
    by_id={r['id']:r for r in rows}
    coverage={}
    for stage,updates in [('warmup',2000),('joint',4000)]:
        actual=json.loads((root/(stage+'_coverage.json')).read_text())
        expected=expected_coverage(rows,updates)
        assert actual['counts']==expected and len(expected)==4202
        coverage[stage]=dict(utterances=len(expected),crop_exposures=sum(expected.values()),
                             min_exposures=min(expected.values()),max_exposures=max(expected.values()))
    log=[json.loads(line) for line in (root/'train_metrics.jsonl').read_text().splitlines()]
    assert [r['global_step'] for r in log]==expected_logged_steps()
    assert all(np.isfinite(r[k]) for r in log for k in ['loss','teacher_mse','logmel','mrstft','gradient_norm'])
    candidates=[]
    for p in root.glob('validation_*.json'):
        _,stage,step=p.stem.split('_')
        score=json.loads(p.read_text())
        assert {r['id'] for r in score['rows']}==set(spec['monitor_ids'])
        assert abs(np.mean([r['logmel'] for r in score['rows']])-score['logmel'])<1e-12
        candidates.append((score['logmel'],(2000 if stage=='joint' else 0)+int(step),stage,int(step)))
    assert len(candidates)==12
    chosen=min(candidates)
    checkpoint=torch.load(root/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    assert checkpoint['stage']==chosen[2] and checkpoint['step']==chosen[3]
    assert checkpoint['protocol_sha256']==protocol_sha
    original=torch.load(root/'g_02500000.pt',map_location='cpu',weights_only=True)['generator']
    warmup=torch.load(root/'checkpoints/warmup.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(original[k],warmup['generator'][k]) for k in original)
    changed=sum(not torch.equal(original[k],checkpoint['generator'][k]) for k in original)
    if checkpoint['stage']=='joint':
        assert changed>0
    assert all(torch.isfinite(v).all() for state in ('adapter','generator') for v in checkpoint[state].values())
    stats=torch.load(root/'normalization.pt',map_location='cpu',weights_only=True)
    assert torch.equal(stats['mean'],checkpoint['adapter']['mean'])
    assert torch.equal(stats['std'],checkpoint['adapter']['std'])
    final=json.loads((root/'final_validation.json').read_text())
    assert final['checkpoint_sha256']==sha(root/'checkpoints/best.pt')
    assert {r['id'] for r in final['rows']}==set(spec['validation_ids'])
    assert abs(np.mean([r['logmel'] for r in final['rows']])-final['logmel'])<1e-12
    audio=[]
    for identity in sorted(spec['validation_ids']):
        path=root/'audio/reconstructed_validation'/(identity+'.wav')
        waveform,rate=sf.read(path,dtype='float32')
        assert rate==16000 and waveform.ndim==1
        assert len(waveform)==by_id[identity]['output_frames']
        assert np.isfinite(waveform).all() and np.max(np.abs(waveform))>0
        assert sf.info(path).subtype=='FLOAT'
        audio.append(dict(id=identity,path=str(path.relative_to(root)),sha256=sha(path),samples=len(waveform)))
    assert len(list((root/'audio/reconstructed_validation').glob('*.wav')))==429
    for name in ('parent_final_quality','parent_final_asr'):
        quality=json.loads((root/name/'result.json').read_text())
        assert len(quality['rows'])==160
        ids={r['id'] for r in quality['rows']}
        assert len(ids)==40 and not ids&set(spec['monitor_ids'])
        assert quality['selection_sha256']==sha(root/'parent_quality_selection.json')
    result=dict(status='PASS',coverage=coverage,training_log_rows=len(log),monitor_checkpoints=len(candidates),
                selected_checkpoint=dict(stage=chosen[2],step=chosen[3],global_step=chosen[1],logmel=chosen[0]),
                frozen_warmup_generator_tensors=len(original),selected_generator_changed_tensors=changed,
                normalization_buffers_unchanged=True,validation_wavs_verified=429,
                exact_native_sample_counts=True,independent_quality_clips=40,quality_monitor_overlap=0,
                effective_protocol_sha256=protocol_sha,validation_audio=audio)
    with (root/'parent_final_audit.json').open('x') as out:
        json.dump(result,out,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='validation_audio'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    audit(parser.parse_args().root)
