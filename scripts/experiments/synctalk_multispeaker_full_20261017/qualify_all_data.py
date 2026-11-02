"""Independent F-side all-identity admission; complete archive bytes and numerical gates."""
import hashlib,json,shlex,subprocess,tarfile
from pathlib import Path
from persist_prepared import ROOT,REMOTE,SSH,sha
IDS=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']
SOURCES=['train_epoch.py','controlled_training.py','execution_state.py','runtime_common.py','audio_context.py','build_datasets.py',
         'qualify_capacity.py','qualify_loading.py','runtime_exec.sh','reclaim_derived_supervision.py','remote_epoch_job.py']

def read_archive(directory):
    proof=json.loads((directory/'verification.json').read_text());assert proof['passed'] and proof['all_archived_member_bytes_verified']
    archive=Path(proof['archive']);assert archive.resolve().is_relative_to(directory.resolve())
    assert sha(archive)==proof['archive_sha256']
    manifest_path=directory/'manifest.json';assert sha(manifest_path)==proof['manifest_sha256']
    files={x['path']:x for x in json.loads(manifest_path.read_text())['files']};seen=set();notes={}
    with tarfile.open(archive,'r|gz') as source:
        for item in source:
            assert item.isfile() and item.name in files and item.name not in seen
            row=files[item.name];assert row['bytes']==item.size
            stream=source.extractfile(item);h=hashlib.sha256();parts=[]
            collect=item.name.endswith('.json') and item.size<4*1024**2
            for part in iter(lambda:stream.read(4*1024**2),b''):
                h.update(part)
                if collect:parts.append(part)
            assert h.hexdigest()==row['sha256'];seen.add(item.name)
            if collect:notes[item.name]=json.loads(b''.join(parts))
    assert seen==files.keys();return notes,proof

def qualify_geometry(g):
    assert g['passed'] and g['train_original_BA_preserved'] and g['heldout_geometry_never_fitted']
    assert len(g['controls'])==2
    for c in g['controls']:
        assert c['passed'] and c['rotation_p95_degrees']<=2 and c['translation_p95_m']<=.005
        assert c['flow_rms_px_median']<=2 and c['flow_rms_px_p95']<=5 and c['independent_projection_max_px']<=.0001
        assert c['fixed_training_points_unchanged']
    assert {s['split']:s['frames'] for s in g['splits']}==dict(train=6000,val=750,test=1500)
    for split in g['splits']:
        assert split['flow_rms_px_median']<=2 and split['flow_rms_px_p95']<=5
        assert split['landmark_rms_px_median']<=5 and split['landmark_rms_px_p95']<=10

def qualify_identity(identity,runtime_proof):
    material=json.loads((ROOT/'material'/identity/'independent_material_qa.json').read_text());assert material['passed']
    stage1=json.loads((ROOT/'preprocessed'/identity/'stage1_receipt.json').read_text())
    assert stage1['passed']
    assert {s['name']:s['frames'] for s in stage1['splits']}==dict(train=6000,val=750,test=1500)
    if stage1['review_required']:
        review=json.loads((ROOT/f'{identity}_face_adjudication.json').read_text())
        assert review['all_five_review_flags_resolved'] and review['complete_landmark_support_frames']==8250
        assert {(r['split'],r['frame']) for r in review['reviewed_frames']}=={(s['name'],f['frame'] if isinstance(f,dict) else f) for s in stage1['splits'] for f in s['review_frames']}
        assert all(x['official_primary_face_correct'] for x in review['reviewed_frames'])
        for row in review['reviewed_frames']:
            directory=ROOT/'preprocessed'/identity/row['split']
            assert sha(directory/'ori_imgs'/f"{row['frame']}.jpg")==row['image_sha256']
            assert sha(directory/'ori_imgs'/f"{row['frame']}.lms")==row['landmarks_sha256']
    else:assert stage1['face_coverage_qualified']
    collected={};proofs=[]
    runtime_directory=Path(runtime_proof['archive']).parent
    for directory in [ROOT/'derived_packages'/identity,ROOT/'qualification_packages'/identity,runtime_directory]:
        notes,proof=read_archive(directory);collected.update(notes);proofs.append(proof)
    g=collected[f'prepared/{identity}/geometry_v2/geometry_qualification.json'];qualify_geometry(g)
    assert g['contract_sha256']==sha(Path(__file__).with_name('preprocessing_contract_v2.json'))
    assert collected[f'prepared/{identity}/supervision_receipt.json']['passed']
    ave=collected[f'features/{identity}/ave_qualification.json'];assert ave['passed'] and ave['frozen_model_unchanged'] and len(ave['rows'])==6
    reference=collected[f'score_reference/{identity}/score_reference_receipt.json']
    assert reference['passed'] and reference['repeat_audio_and_visual_exact'] and not reference['score_based_exclusions'] and len(reference['clips'])==6
    loading=collected[f'loading_qualification/{identity}/loading_qualification.json']
    assert loading['passed'] and loading['cases']==90 and loading['actual_pixel_rays_and_conditions_exact'] and loading['whole_recording_attention_boundaries_exact']
    capacity=collected[f'capacity_qualification/{identity}/capacity_qualification.json']
    assert capacity['passed'] and capacity['optimizer_updates']==0 and len(capacity['rows'])==18 and len(capacity['initializations'])==2
    assert all(x['initialization_exact'] and x['parameter_bytes_unchanged'] and x['num_rays_argument']==65536 and x['max_steps']==16 for x in capacity['rows'])
    reference_proof=json.loads((ROOT/'qualification_state/independent_reference_persistence.json').read_text());assert reference_proof['passed']
    reference_file=next(x for x in reference_proof['files'] if x['identity']==identity)
    assert sha(Path(reference_file['path']))==reference_file['sha256']
    independent=json.loads(Path(reference_file['path']).read_text())
    assert independent['passed'] and independent['all18_audio_features_fresh_exact'] and independent['actual_FP32_MFCC_input_exact'] and independent['checked_matrices']==18
    return dict(identity=identity,archive_proofs=proofs,passed=True,training_admitted=False)

def run():
    output=ROOT/'data_qualification';output.mkdir(exist_ok=False)
    progress=dict(passed=False,training_ready=False,identities=[],checks=[])
    def save():(output/'receipt.json').write_text(json.dumps(progress,indent=2))
    save()
    worker=json.loads((ROOT/'runtime_qualification_worker_receipt.json').read_text())
    assert worker.get('passed') and [x['identity'] for x in worker['identities']]==IDS
    stage1_proof=json.loads((ROOT/'stage1_persistence_receipt.json').read_text())
    assert stage1_proof['passed'] and [r['identity'] for r in stage1_proof['verified_identities']]==IDS
    assert all(r['passed'] and r['all_extracted_files_verified'] and r['files']==41251 for r in stage1_proof['verified_identities'])
    audit=json.loads((ROOT/'qualification_state/official_source_audit.json').read_text());assert audit['passed']
    unit_log=(ROOT/'qualification_state/execution_unit_tests_attempt06.log').read_text()
    assert '\nRan 20 tests ' in unit_log and unit_log.rstrip().endswith('\nOK'),'complete real runtime unit controls must pass without skips'
    io=json.loads((ROOT/'controller_qualification/io_state_rotation_attempt02.json').read_text());assert io['passed']
    windows=json.loads((ROOT/'controller_qualification/windows_process_lock_and_resume_attempt01.json').read_text());assert windows['passed']
    for name in ['persist_training_epoch.py','paired_budget.py']:assert io['sources'][name]==sha(Path(__file__).with_name(name))
    for identity in IDS:
        progress['current_identity']=identity;save()
        runtime_proof=next(x for x in worker['identities'] if x['identity']==identity)
        check=qualify_identity(identity,runtime_proof)
        progress['identities'].append(identity);progress['checks'].append(check);save()
    code="""import hashlib,json,zipfile
from pathlib import Path
r=Path(%r);names=%r;audit=%r
assert hashlib.sha256((r/'official_pinned.zip').read_bytes()).hexdigest()==audit['archive_sha256']
allowed={row['path']:row['working_sha256'] for row in audit['python_differences']};checked=0
with zipfile.ZipFile(r/'official_pinned.zip') as z:
 for info in z.infolist():
  if info.filename.endswith('.py'):
   rel=info.filename.split('/',1)[1];p=r/'official'/rel
   if p.exists():
    checked+=1;actual=hashlib.sha256(p.read_bytes()).hexdigest();reference=hashlib.sha256(z.read(info)).hexdigest()
    assert actual==(allowed[rel] if rel in allowed else reference),(rel,'official execution source drift')
assert checked==audit['python_files_checked']
print(json.dumps(dict(sources={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},
official_execution_sources={n:hashlib.sha256((r/'official'/n).read_bytes()).hexdigest() for n in ['nerf_triplane/network.py','nerf_triplane/renderer.py','nerf_triplane/utils.py','nerf_triplane/provider.py']},
protocol_sha256=hashlib.sha256((r/'protocol.json').read_bytes()).hexdigest(),execution_contract_sha256=hashlib.sha256((r/'execution_contract_v1.json').read_bytes()).hexdigest(),
worker=json.loads((r/'stage2_v2_worker_receipt.json').read_text()),official_python_files_rechecked=checked)))
"""%(REMOTE,SOURCES,audit)
    actual=json.loads(subprocess.check_output(SSH+['python3 -c '+shlex.quote(code)]))
    assert actual['worker']['passed'] and [x['identity'] for x in actual['worker']['identities']]==IDS
    for name,digest in actual['sources'].items():assert sha(Path(__file__).with_name(name))==digest
    assert actual['protocol_sha256']==sha(Path(__file__).with_name('protocol.json'))
    assert actual['execution_contract_sha256']==sha(Path(__file__).with_name('execution_contract_v1.json'))
    progress.update(passed=True,training_ready=True,protocol_sha256=actual['protocol_sha256'],
        execution_contract_sha256=actual['execution_contract_sha256'],executing_sources=actual['sources'],
        official_execution_sources=actual['official_execution_sources'],
        windows_controller_sources={name:sha(Path(__file__).with_name(name)) for name in ['training_coordinator.py','persist_training_epoch.py','paired_budget.py','restore_supervision.py','execution_journal.py','full_experiment_supervisor.py']},
        future_actual_optimizer_budget_equality='not established by zero-update probes; mandatory paired phase-end acceptance')
    save();(ROOT/'data_qualification.json').write_text(json.dumps(progress,indent=2))
    subprocess.run(SSH+[f'cat > {REMOTE}/data_qualification.json'],input=json.dumps(progress,indent=2).encode(),check=True)
    return progress
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--preflight-identity',choices=IDS);a=p.parse_args()
    if a.preflight_identity:
        worker=json.loads((ROOT/'runtime_qualification_worker_receipt.json').read_text())
        proof=next(x for x in worker['identities'] if x['identity']==a.preflight_identity)
        result=qualify_identity(a.preflight_identity,proof)
        out=ROOT/'data_identity_preflight'/a.preflight_identity;out.mkdir(parents=True,exist_ok=False)
        (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    else:print(json.dumps(run()))
