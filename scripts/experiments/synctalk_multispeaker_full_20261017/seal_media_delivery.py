"""Seal complete per-identity/seed results, original matrices and reproducibility state."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
from cohort_binding import identity_root,verify_member_pins

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def run(identity,seed):
    inputs=identity_root(identity,ROOT,require_admission=True);verify_member_pins(ROOT,identity)
    assert identity in ['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers'] and seed in [42,2026]
    scores=ROOT/'scores'/identity/f'seed{seed}'
    qa=json.loads((scores/'independent_score_qualification.json').read_text());assert qa['passed'] and qa['cells']==27 and qa['clip_matrices']==162
    assert json.loads((scores/'scores.json').read_text())['passed']
    for gain in [-6,0,6]:
        for driver in [-6,0,6]:
            receipt=json.loads((ROOT/'renders'/identity/f'seed{seed}'/f'train_{gain:+d}'/f'driver_{driver:+d}'/'render_receipt.json').read_text());assert receipt['passed'] and receipt['rgb_and_pcm_readback_exact']
    preview=ROOT/'preview'/identity/f'seed{seed}';preview.mkdir(parents=True,exist_ok=False)
    videos=[ROOT/'renders'/identity/f'seed{seed}'/f'train_{gain:+d}'/'driver_+0/original_pcm.mkv' for gain in [-6,0,6]]
    command=['ffmpeg','-v','error','-n']
    for video in videos:command+=['-i',str(video)]
    command+=['-filter_complex','[0:v][1:v][2:v]hstack=inputs=3[v]','-map','[v]','-map','1:a:0','-t','10',
        '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k',str(preview/'train_gains_driver0_watch_only.mp4')]
    subprocess.run(command,check=True)
    (preview/'preview_receipt.json').write_text(json.dumps(dict(passed=True,identity=identity,seed=seed,columns_training_gain_db=[-6,0,6],
        driver_gain_db=0,evaluation_use=False,semantics='H264/AAC watching only; every score uses formal FFV1/original PCM'),indent=2))
    bases=[ROOT/'renders'/identity/f'seed{seed}',scores,preview,inputs/'score_reference'/identity,ROOT/'blocks'/identity/f'seed{seed}']
    paths=[p for base in bases for p in base.rglob('*') if p.is_file() and not p.is_symlink()]
    training=ROOT/'training'/identity/f'seed{seed}'
    for gain in [-6,0,6]:
        for phase,epoch in [('coarse',10),('fine',17)]:
            base=training/f'gain_{gain:+d}'/phase/f'ep{epoch:04d}/attempt01'
            receipt=json.loads((base/'epoch_receipt.json').read_text());paths+=[base/'epoch_receipt.json',Path(receipt['checkpoint_path'])]
            if phase=='fine':paths.append(Path(receipt['final_ema_path']))
    paths+=[p for p in training.rglob('*') if p.is_file() and p.name in ['attempts.jsonl','epoch_receipt.json','validation_receipt.json']]
    paths+=[p for p in training.rglob('validation_*dB_mse.npy') if p.is_file()]
    paths += [inputs/'capacity_qualification'/identity/f'initial_seed{seed}.pt',inputs/'capacity_qualification'/identity/'capacity_qualification.json',
        ROOT/'data_qualification.json',ROOT/'protocol.json',ROOT/'execution_contract_v1.json',ROOT/'preprocessing_contract_v2.json']
    paths += [ROOT/name for name in ['cohort_binding.py','cohort_selection.json','render_model.py','score_identity_seed.py','validate_scores.py','seal_media_delivery.py','prepare_score_reference.py',
        'train_epoch.py','controlled_training.py','execution_state.py','runtime_common.py','audio_context.py','remote_epoch_job.py','remote_result_job.py']]
    paths += [ROOT/'official/nerf_triplane'/name for name in ['network.py','renderer.py','utils.py','provider.py']]
    rows=[]
    for path in sorted(set(paths)):
        assert path.resolve().is_relative_to(ROOT.resolve()) and path.is_file() and not path.is_symlink()
        rows.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    directory=ROOT/'media_manifests'/identity/f'seed{seed}';directory.mkdir(parents=True,exist_ok=False)
    manifest=dict(identity=identity,seed=seed,passed=True,formal_videos=9,grid_cells=27,files=rows,raw_bytes=sum(x['bytes'] for x in rows),source_sha256=sha(Path(__file__)))
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2));(directory/'files.nul').write_bytes(b''.join(x['path'].encode()+b'\0' for x in rows))
    print(json.dumps(dict(identity=identity,seed=seed,passed=True,files=len(rows),bytes=manifest['raw_bytes'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.identity,a.seed)
