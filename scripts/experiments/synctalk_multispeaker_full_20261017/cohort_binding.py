"""Bind every scientific input to one immutable, independently qualified cohort."""
import hashlib
import json
from pathlib import Path

REMOTE = Path('/dev/shm/synctalk-multispeaker-data-20261017')
FROOT = Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017')
IDS = ['SherrodBrown', 'PatrickLeahy', 'ChrisVanHollen', 'CathyMcMorrisRodgers']
VERSION = 'metadata_identity_replacement_v3'
REVISIONS = {'SherrodBrown': None, 'PatrickLeahy': None,
             'ChrisVanHollen': 'ChrisVanHollen_metadata_order_v1',
             'CathyMcMorrisRodgers': 'CathyMcMorrisRodgers_recording2_v2'}
POST_RECEIPTS = {'ChrisVanHollen': 'post_geometry_worker_attempt04_receipt.json',
                 'CathyMcMorrisRodgers': 'post_geometry_worker_attempt03_receipt.json'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def expected_root(base, identity):
    assert identity in IDS
    revision = REVISIONS[identity]
    return Path(base) if revision is None else Path(base) / 'dataset_revisions' / revision


def selection(base, require_admission=False):
    base = Path(base)
    path = base / 'cohort_selection.json'
    if not path.exists():
        assert not require_admission, 'versioned cohort selection has not been admitted'
        return None
    assert not path.is_symlink()
    contract = json.loads(path.read_text())
    assert contract['version'] == VERSION and contract['identities'] == IDS
    assert set(contract['members']) == set(IDS)
    for identity in IDS:
        member = contract['members'][identity]
        # Both platforms validate both namespaces: a Windows path must never be
        # interpreted as a POSIX relative path, or vice versa.
        assert member['remote_root'] == expected_root(REMOTE, identity).as_posix()
        assert member['persistent_root'] == expected_root(FROOT, identity).as_posix()
        assert member['revision'] == REVISIONS[identity]
    admission_path = base / 'data_qualification.json'
    if require_admission:
        admission = json.loads(admission_path.read_text())
        assert admission['passed'] and admission['training_ready']
        assert admission['cohort_version'] == VERSION and admission['identities'] == IDS
        assert admission['cohort_selection_sha256'] == sha(path), 'cohort selection drift'
    return contract


def identity_root(identity, base=REMOTE, require_admission=False):
    contract = selection(base, require_admission)
    assert identity in IDS
    if contract is None:
        return Path(base)  # Only original, unadmitted preparation probes.
    selected = expected_root(base, identity)
    assert not selected.is_symlink(), 'a selected revision root cannot alias rejected data'
    return selected


def verify_member_pins(base, identity, contract=None):
    contract = contract or selection(base)
    assert contract is not None
    root = identity_root(identity, base)
    rows = contract['members'][identity]['remote_pins']
    assert rows and len({row['path'] for row in rows}) == len(rows)
    for row in rows:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root / relative
        assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
    return dict(identity=identity, root=str(root), files=len(rows), passed=True)
