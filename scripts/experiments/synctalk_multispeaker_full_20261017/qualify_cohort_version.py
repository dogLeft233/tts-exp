"""Fresh all-four admission, preserving all rejected namespaces and false receipts."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
import qualify_all_data as original
import persist_cohort_runtime_qualification as persistence
from persist_prepared import SSH
from execution_journal import atomic_json
from cohort_binding import FROOT, REMOTE, IDS, VERSION, REVISIONS, POST_RECEIPTS, expected_root, sha, selection

HERE=Path(__file__).parent
EXECUTING=['train_epoch.py','controlled_training.py','execution_state.py','runtime_common.py','audio_context.py',
           'build_datasets.py','qualify_capacity.py','qualify_loading.py','runtime_exec.sh',
           'reclaim_derived_supervision.py','remote_epoch_job.py','cohort_binding.py','cohort_requalification.py',
           'cohort_reference_requalification.py','qualify_score_references.py']
CONTROLLERS=['training_coordinator.py','persist_training_epoch.py','paired_budget.py','restore_supervision.py',
             'restore_cohort_supervision.py','cohort_binding.py','execution_journal.py','versioned_experiment_supervisor.py',
             'qualify_cohort_version.py','qualify_all_data.py','persist_cohort_runtime_qualification.py','persist_prepared.py']


def remote(code):
    return json.loads(subprocess.check_output(SSH+['python3 -c '+shlex.quote(code)]))


def publish(path, payload):
    code='from pathlib import Path;import sys;p=Path(%r);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(sys.stdin.buffer.read())'%path.as_posix()
    subprocess.run(SSH+['python3 -c '+shlex.quote(code)],input=payload,check=True)


def candidate_ready():
    for identity,revision in REVISIONS.items():
        if revision is None:continue
        path=expected_root(FROOT,identity)/POST_RECEIPTS[identity]
        if not path.exists():return False
        row=json.loads(path.read_text())
        assert row.get('passed') is not False, (identity,row)
        if not row.get('passed'):return False
        assert row['state']=='revision_independently_qualified_pending_cohort_version_admission'
    return True


def collect_remote_member(identity):
    code='''import hashlib,json
from pathlib import Path
r=Path(%r);i=%r
paths=[]
for name in ['material','features','datasets','score_reference']:
 for p in (r/name/i).rglob('*'):
  if p.is_file() and not p.is_symlink() and p.suffix in ['.json','.wav','.npy','.bin']:paths.append(p)
paths += [r/'preprocessed'/i/'stage1_receipt.json',r/'prepared'/i/'geometry_v2/geometry_qualification.json',
          r/'prepared'/i/'supervision_receipt.json',r/'capacity_qualification'/i/'capacity_qualification.json',
          r/'loading_qualification'/i/'loading_qualification.json',r/'snapshots'/i/'manifest.json',
          r/'snapshots'/i/'F_archive_verification.json']
paths += list((r/'capacity_qualification'/i).glob('initial_seed*.pt'))
paths += [p for p in (r/'prepared'/i).rglob('*') if p.is_file() and not p.is_symlink()
          and not any(kind in p.relative_to(r/'prepared'/i).parts for kind in ['gt_imgs','torso_imgs'])]
files=[]
for p in sorted(set(paths)):
 assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(r.resolve())
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 files.append(dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size,sha256=h.hexdigest()))
assert len(list((r/'capacity_qualification'/i).glob('initial_seed*.pt')))==2
material=json.loads((r/'material'/i/'material_receipt.json').read_text());assert material['media_preparation_passed']
roles={x['role']:x['source'] for x in material['recordings']}
assert set(roles)=={'main','test'} and roles['main']['original_video_id']!=roles['test']['original_video_id']
assert 360<=sum(x['seconds'] for x in roles.values())<=480
print(json.dumps(dict(files=files,roles=roles)))
'''%(expected_root(REMOTE,identity).as_posix(),identity)
    return remote(code)


def verify_official_and_controls():
    audit=json.loads((FROOT/'qualification_state/official_source_audit.json').read_text());assert audit['passed']
    code='''import hashlib,json,zipfile
from pathlib import Path
r=Path(%r);audit=%r
assert hashlib.sha256((r/'official_pinned.zip').read_bytes()).hexdigest()==audit['archive_sha256']
allowed={x['path']:x['working_sha256'] for x in audit['python_differences']};checked=0
with zipfile.ZipFile(r/'official_pinned.zip') as z:
 for info in z.infolist():
  if info.filename.endswith('.py'):
   rel=info.filename.split('/',1)[1];p=r/'official'/rel
   if p.exists():
    checked+=1;actual=hashlib.sha256(p.read_bytes()).hexdigest();reference=hashlib.sha256(z.read(info)).hexdigest()
    assert actual==(allowed[rel] if rel in allowed else reference),(rel,'official drift')
assert checked==audit['python_files_checked']
print(json.dumps(dict(checked=checked,sources={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in %r},
official={n:hashlib.sha256((r/'official'/n).read_bytes()).hexdigest() for n in ['nerf_triplane/network.py','nerf_triplane/renderer.py','nerf_triplane/utils.py','nerf_triplane/provider.py']})))
'''%(REMOTE.as_posix(),audit,EXECUTING+['protocol.json','execution_contract_v1.json'])
    actual=remote(code)
    for name,digest in actual['sources'].items():assert sha(HERE/name)==digest,(name,'executing source drift')
    unit=(FROOT/'qualification_state/execution_unit_tests_attempt09.log').read_text()
    assert '\nRan 20 tests ' in unit and unit.rstrip().endswith('\nOK')
    control=json.loads((FROOT/'controller_qualification/cohort_execution_controls_attempt03.json').read_text());assert control['passed']
    for row in control['files']:assert sha(Path(row['path']))==row['sha256']
    for name in ['io_state_rotation_attempt02.json','windows_process_lock_and_resume_attempt01.json']:
        assert json.loads((FROOT/'controller_qualification'/name).read_text())['passed']
    io=json.loads((FROOT/'controller_qualification/io_state_rotation_attempt02.json').read_text())
    for name in ['persist_training_epoch.py','paired_budget.py']:assert io['sources'][name]==sha(HERE/name)
    return actual


def run():
    assert candidate_ready(), 'candidate preparation is still running'
    policy=FROOT/'identity_replacement_policy_v1'
    active=json.loads((policy/'active_candidate_attempt02.json').read_text())
    assert active['candidate']['identity']==IDS[2]=='ChrisVanHollen'
    assert active['policy_sha256']==sha(policy/'policy.json')
    revision=json.loads((expected_root(FROOT,IDS[2])/'revision_contract.json').read_text())
    assert revision['policy_sha256']==active['policy_sha256']
    assert revision['source_availability_audit_sha256']==sha(policy/'source_availability_attempt01.json')
    audit=json.loads((policy/'source_availability_attempt01.json').read_text())
    assert audit['selected']==active['candidate'] and not audit['selection_uses_scores']
    directory=FROOT/'cohort_versions'/VERSION;directory.mkdir(parents=True,exist_ok=False)
    status=dict(passed=False,training_ready=False,cohort_version=VERSION,identities=[],checks=[])
    atomic_json(directory/'receipt.json',status)
    contract=dict(version=VERSION,identities=IDS,members={},identity_replacement_policy_sha256=active['policy_sha256'],
                  source_availability_audit_sha256=revision['source_availability_audit_sha256'])
    for identity in IDS:
        selected=expected_root(FROOT,identity)
        preflight=json.loads((selected/'data_identity_preflight'/identity/'verification.json').read_text());assert preflight['passed']
        prior=original.ROOT
        try:
            original.ROOT=selected
            check=original.qualify_identity(identity,preflight['archive_proofs'][-1])
        finally:original.ROOT=prior
        member=collect_remote_member(identity)
        for source in member['roles'].values():
            raw=FROOT/'source'/source['file'];assert raw.is_file() and sha(raw)==source['sha256']
        if REVISIONS[identity]:
            revision_contract=selected/'revision_contract.json'
            assert revision_contract.is_file()
        contract['members'][identity]=dict(revision=REVISIONS[identity],remote_root=expected_root(REMOTE,identity).as_posix(),
            persistent_root=selected.as_posix(),roles=member['roles'],remote_pins=member['files'],
            archive_proofs=check['archive_proofs'],preflight_sha256=sha(selected/'data_identity_preflight'/identity/'verification.json'),
            revision_contract_sha256=sha(selected/'revision_contract.json') if REVISIONS[identity] else None)
        status['checks'].append(check);status['identities'].append(identity);atomic_json(directory/'receipt.json',status)
    raw=json.dumps(contract,indent=2).encode()
    (directory/'selection.json').write_bytes(raw)
    assert not (FROOT/'cohort_selection.json').exists();(FROOT/'cohort_selection.json').write_bytes(raw)
    selection(FROOT)
    publish(REMOTE/'cohort_selection.json',raw)
    actual=verify_official_and_controls()
    status['binding_requalification']=[]
    for identity in IDS:
        log=directory/(identity+'_selected_runtime.log')
        command=['flock','-x',(REMOTE/'preparation_GPU.lock').as_posix(),'bash',(REMOTE/'runtime_exec.sh').as_posix(),
                 (REMOTE/'cohort_requalification.py').as_posix(),'--identity',identity]
        with log.open('xb') as stream:subprocess.run(SSH+[shlex.join(command)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        prior=persistence.ROOT,persistence.REMOTE
        try:
            persistence.ROOT=directory;persistence.REMOTE=(REMOTE/'cohort_versions'/VERSION).as_posix()
            proof=persistence.run(identity,1)
        finally:persistence.ROOT,persistence.REMOTE=prior
        result=remote('import json;from pathlib import Path;print((Path(%r)/"cohort_versions"/%r/"binding_qualification"/%r/"verification.json").read_text())'%(REMOTE.as_posix(),VERSION,identity))
        assert result['passed'] and result['selection_sha256']==hashlib.sha256(raw).hexdigest()
        atomic_json(directory/(identity+'_binding_verification.json'),result)
        status['binding_requalification'].append(dict(identity=identity,passed=True,runtime_archive=proof,binding=result))
        atomic_json(directory/'receipt.json',status)
    # Recheck every source after the GPU queue and every pinned input at the
    # epoch boundary; no existing rejected global stage receipts are touched.
    assert actual==verify_official_and_controls()
    command=['flock','-x',(REMOTE/'preparation_GPU.lock').as_posix(),
             '/home/wjj/.venvs/syncnet/bin/python',(REMOTE/'cohort_reference_requalification.py').as_posix()]
    with (directory/'fresh_reference_requalification.log').open('xb') as stream:
        subprocess.run(SSH+[shlex.join(command)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    for identity in IDS:
        reference=remote('from pathlib import Path;print(Path(%r).read_text())'%(
            REMOTE/'cohort_versions'/VERSION/'score_reference'/identity/'independent_reference_qualification.json').as_posix())
        assert reference['passed'] and reference['checked_matrices']==18 and reference['all18_audio_features_fresh_exact']
        atomic_json(directory/(identity+'_fresh_reference_verification.json'),reference)
    assert actual==verify_official_and_controls()
    code='from cohort_binding import *;import json;print(json.dumps([verify_member_pins(REMOTE,i) for i in IDS]))'
    subprocess.run(SSH+['cd '+shlex.quote(REMOTE.as_posix())+' && python3 -c '+shlex.quote(code)],check=True)
    status.update(passed=True,training_ready=True,cohort_selection_sha256=hashlib.sha256(raw).hexdigest(),
        protocol_sha256=actual['sources']['protocol.json'],execution_contract_sha256=actual['sources']['execution_contract_v1.json'],
        executing_sources={n:actual['sources'][n] for n in EXECUTING},official_execution_sources=actual['official'],
        windows_controller_sources={n:sha(HERE/n) for n in CONTROLLERS},
        future_actual_optimizer_budget_equality='mandatory paired phase-end acceptance; zero-update probes establish no training equality')
    atomic_json(directory/'receipt.json',status)
    target=FROOT/'data_qualification.json';assert not target.exists(), 'preserve and inspect prior admission before activation'
    payload=json.dumps(status,indent=2).encode();target.write_bytes(payload)
    publish(REMOTE/'data_qualification.json',payload)
    return status


if __name__=='__main__':run()
