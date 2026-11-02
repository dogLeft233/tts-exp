"""Retire only this task's duplicate F JPEG/parser cache after fresh full archive verification."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
from persist_prepared import ROOT,REMOTE,SSH,sha
from execution_journal import atomic_json
IDS=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']

def verify_backup(root,identity,source):
    packages=root/'preprocessed_packages';directory=root/'preprocessed'/identity
    archive=packages/f'{identity}_stage1.tar.gz';manifest=packages/f'{identity}_stage1_manifest.json'
    proof=json.loads((packages/f'{identity}_verification.json').read_text())
    assert proof['passed'] and proof['all_extracted_files_verified'] and proof['identity']==identity
    assert archive.stat().st_size==source['bytes'] and sha(archive)==source['sha256']==proof['archive_sha256']
    assert sha(manifest)==source['manifest_sha256']
    rows=json.loads(manifest.read_text());expected={row['path']:row for row in rows};assert len(expected)==len(rows)==proof['files']
    seen=set()
    with tarfile.open(archive,'r|gz') as stream:
        for member in stream:
            parts=Path(member.name).parts;assert parts and parts[0]==identity and '..' not in parts
            if member.isdir():continue
            assert member.isfile() and not member.issym() and not member.islnk()
            relative=Path(*parts[1:]).as_posix();assert relative in expected and relative not in seen
            row=expected[relative];assert row['bytes']==member.size
            h=hashlib.sha256();file=stream.extractfile(member)
            for part in iter(lambda:file.read(4*1024**2),b''):h.update(part)
            assert h.hexdigest()==row['sha256'];seen.add(relative)
    assert seen==expected.keys()
    for relative,row in expected.items():
        file=directory/relative
        assert not file.is_symlink() and file.resolve().is_relative_to(directory.resolve())
        assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256']
    return rows

def reclaim(root,identity,rows,keep):
    directory=(root/'preprocessed'/identity).resolve();assert directory.is_relative_to((root/'preprocessed').resolve())
    selected=[row for row in rows if row['path'] not in keep]
    # Every target is checked before the first unlink; never use recursive deletion.
    for row in selected:
        file=directory/row['path'];assert file.resolve().is_relative_to(directory) and file.is_file() and not file.is_symlink()
        assert sha(file)==row['sha256']
    for row in selected:(directory/row['path']).unlink()
    return dict(retired_files=len(selected),retired_bytes=sum(row['bytes'] for row in selected),retained_files=sorted(keep))

def run(identity):
    assert identity in IDS
    output=ROOT/'stage1_raw_cache_retirement'/identity;output.mkdir(parents=True,exist_ok=False)
    source=json.loads(subprocess.check_output(SSH+[f'cat {REMOTE}/{identity}_stage1_package.json']))
    assert source['passed'] and source['identity']==identity and source['files']==41251
    rows=verify_backup(ROOT,identity,source)
    keep={row['path'] for row in rows if row['path'].endswith('.json')}
    if identity=='SherrodBrown':
        reviewed=json.loads((ROOT/'SherrodBrown_face_adjudication.json').read_text())
        assert reviewed['all_five_review_flags_resolved']
        for row in reviewed['reviewed_frames']:
            for suffix in ['.jpg','.lms']:keep.add(f"{row['split']}/ori_imgs/{row['frame']}{suffix}")
    assert keep<=set(row['path'] for row in rows) and 'stage1_receipt.json' in keep
    atomic_json(output/'backup_verification.json',dict(passed=True,archive_sha256=source['sha256'],manifest_sha256=source['manifest_sha256'],
        all_41251_archive_member_bytes_and_loose_cache_bytes_verified=True,source_metadata=source))
    result=reclaim(ROOT,identity,rows,keep)
    result.update(identity=identity,passed=True,archive_sha256=source['sha256'],manifest_sha256=source['manifest_sha256'],
        backup_retained=True,original_canonical_material_retained=True,SHM_preprocessing_unchanged=True,
        semantics='only duplicate loose F stage1 cache; full verified tar and manifest remain, JSON and adjudication controls retained')
    atomic_json(output/'verification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identities',nargs='+',choices=IDS,default=IDS);a=p.parse_args()
    for identity in a.identities:run(identity)
