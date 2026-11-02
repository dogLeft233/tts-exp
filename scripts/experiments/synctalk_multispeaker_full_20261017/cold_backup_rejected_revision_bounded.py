"""Independent cloud proof before retiring only rejected-role archive/flow duplicates."""
import gzip,hashlib,json,os,tarfile,traceback
from pathlib import Path
from cloud_archive_bounded import run as backup
from persist_prepared import ROOT,sha
from execution_journal import atomic_json,exclusive_lock


def run():
    owner=ROOT/'dataset_revisions/DebbieDingell_recording_roles_v2'
    with exclusive_lock(owner/'cold_backup.lock'):
        output=owner/'cold_backup_rejected_revision_attempt02.json';assert not output.exists()
        status=dict(pid=os.getpid(),state='verifying_rejected_inputs',passed=False,verified=[],training_admitted=False)
        def save():atomic_json(output,status)
        save()
        try:
            failed=owner/'failed_preparation_preservation_attempt01'
            failure=json.loads((failed/'verification.json').read_text());assert failure['passed'] and not failure['geometry_qualified']
            archive=Path(failure['archive']);assert archive.resolve().is_relative_to(failed.resolve()) and sha(archive)==failure['archive_sha256']
            flow_dir=owner/'flow_caches/DebbieDingell/train';flow=json.loads((flow_dir/'persistence.json').read_text())
            packed=Path(flow['path']);assert packed.resolve().is_relative_to(flow_dir.resolve()) and flow['decompressed_source_verified']
            assert sha(packed)==flow['gzip_sha256']
            h=hashlib.sha256();count=0
            with gzip.open(packed,'rb') as stream:
                for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block);count+=len(block)
            assert count==flow['raw_bytes'] and h.hexdigest()==flow['raw_sha256']
            for kind,path,digest,directory in [('flow',packed,flow['gzip_sha256'],flow_dir),('bundle',archive,failure['archive_sha256'],failed)]:
                if kind=='bundle':
                    status['state']='retain_failure_arrays_and_PCM';save()
                    manifest=json.loads((failed/'manifest.json').read_text());expected={r['path']:r for r in manifest['files']};seen=set();thin=[]
                    with tarfile.open(archive,'r|gz') as source:
                        for member in source:
                            assert member.isfile() and member.name in expected and member.name not in seen
                            row=expected[member.name];assert row['bytes']==member.size
                            keep=Path(member.name).suffix not in ['.jpg','.png','.mkv'] or 'flow_masks' in Path(member.name).parts
                            target=failed/'thin'/member.name;assert target.resolve().is_relative_to((failed/'thin').resolve())
                            if keep:target.parent.mkdir(parents=True,exist_ok=True);stream=target.open('xb')
                            h=hashlib.sha256()
                            try:
                                with source.extractfile(member) as data:
                                    for block in iter(lambda:data.read(4*1024**2),b''):
                                        h.update(block)
                                        if keep:stream.write(block)
                            finally:
                                if keep:stream.close()
                            assert h.hexdigest()==row['sha256'];seen.add(member.name)
                            if keep:assert sha(target)==row['sha256'];thin.append(row)
                    assert seen==expected.keys();atomic_json(failed/'thin_manifest.json',dict(passed=True,files=thin,archive_sha256=digest))
                status.update(state='independent_cloud_readback',kind=kind);save()
                name='failed-revision-DebbieDingell-roles-v2-'+kind
                cloud=backup(path,digest,name,2);cloud_dir=ROOT/'cloud_delivery'/name/'transport_attempt02'
                proof=json.loads((cloud_dir/'readback_persistence_verification.json').read_text())
                assert cloud['passed'] and cloud['independent_linux_cloud_readback'] and proof['passed']
                assert digest==cloud['sha256']==proof['archive_sha256']==sha(path)
                locator=dict(passed=True,kind=kind,original_local_path=str(path),bytes=path.stat().st_size,sha256=digest,
                    cloud_manifest=str(cloud_dir/'manifest.json'),independent_receipt=str(cloud_dir/'cloud_readback_receipt.json'),
                    receipt_sha256=sha(cloud_dir/'cloud_readback_receipt.json'),local_copy_retired=False,
                    requires_explicit_cloud_restore_for_full_original_bytes=True)
                if kind=='flow':locator.update(raw_bytes=flow['raw_bytes'],raw_sha256=flow['raw_sha256'])
                else:
                    for row in thin:assert sha(failed/'thin'/row['path'])==row['sha256']
                    locator.update(local_failure_arrays_PCM_metadata_retained=True,thin_manifest_sha256=sha(failed/'thin_manifest.json'))
                atomic_json(directory/'cold_storage_locator.json',locator)
                assert path.resolve().is_relative_to(owner.resolve());path.unlink()
                locator['local_copy_retired']=True;atomic_json(directory/'cold_storage_locator.json',locator)
                status['verified'].append(locator);save()
            status.update(state='completed',passed=True);save()
        except BaseException:
            status.update(state='failed',passed=False,error=traceback.format_exc());save();raise


if __name__=='__main__':run()
