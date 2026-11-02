"""Cold-store failed-pair canonical bytes; remove local copies only after independent full readback."""
import json,os,traceback
from pathlib import Path
from cloud_archive import run as backup
from persist_prepared import ROOT,sha
from execution_journal import atomic_json,exclusive_lock
def run():
    with exclusive_lock(ROOT/'failed_material_cold_backup.lock'):
        receipt=ROOT/'failed_material_cold_backup_receipt.json';assert not receipt.exists()
        status=dict(pid=os.getpid(),state='running',verified=[],training_admitted=False)
        atomic_json(receipt,status)
        try:
            for identity in ['DebbieDingell','CathyMcMorrisRodgers']:
                folder=ROOT/'material'/identity
                metadata=json.loads((folder/'material_receipt.json').read_text())
                failure=ROOT/'geometry_diagnostics'/(identity+'_failed_v2_attempt01')/'full_failure_bundle'
                proof=json.loads((failure/'verification.json').read_text())
                assert proof['passed'] and proof['geometry_qualified'] is False and sha(failure/'failure.tar.gz')==proof['archive_sha256']
                locator=dict(identity=identity,original_pair_geometry_qualified=False,roles=[],failure_archive_sha256=proof['archive_sha256'])
                locator_path=folder/'failed_pair_cold_storage_locator.json';assert not locator_path.exists()
                for role in ['main','test']:
                    status.update(identity=identity,role=role,state='independent_cloud_backup');atomic_json(receipt,status)
                    original=next(r for r in metadata['recordings'] if r['role']==role)
                    target=folder/(role+'.mkv')
                    assert target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(folder.resolve())
                    assert target.stat().st_size==original['bytes'] and sha(target)==original['sha256']
                    name=f'failed-v1-{identity}-{role}'
                    cloud=backup(target,original['sha256'],name,1)
                    cloud_dir=ROOT/'cloud_delivery'/name/'transport_attempt01'
                    persisted=json.loads((cloud_dir/'readback_persistence_verification.json').read_text())
                    assert cloud['passed'] and cloud['independent_linux_cloud_readback'] and persisted['passed']
                    assert cloud['sha256']==persisted['archive_sha256']==sha(target)==original['sha256'] and cloud['bytes']==original['bytes']
                    row=dict(role=role,original_local_path=str(target),bytes=original['bytes'],sha256=original['sha256'],
                        cloud_manifest=str(cloud_dir/'manifest.json'),independent_readback_receipt=str(cloud_dir/'cloud_readback_receipt.json'),
                        receipt_sha256=sha(cloud_dir/'cloud_readback_receipt.json'),requires_explicit_cloud_restore_for_original_byte_redecode=True)
                    # Record the durable locator before retiring this verified duplicate.
                    locator['roles'].append(dict(row,local_copy_retired=False));atomic_json(locator_path,locator)
                    target.unlink();locator['roles'][-1]['local_copy_retired']=True;atomic_json(locator_path,locator)
                    status['verified'].append(dict(identity=identity,**row,local_copy_retired=True));atomic_json(receipt,status)
            status.update(state='completed',passed=True);atomic_json(receipt,status)
        except BaseException:
            status.update(state='failed',passed=False,error=traceback.format_exc());atomic_json(receipt,status);raise
if __name__=='__main__':run()
