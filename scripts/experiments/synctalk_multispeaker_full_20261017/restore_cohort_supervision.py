"""Use each selected archive's original receiver without moving rejected data."""
from pathlib import Path
import restore_supervision as original
from cohort_binding import FROOT, REMOTE, identity_root


def run(identity, mode):
    persistent = identity_root(identity, FROOT, require_admission=True)
    # Windows has no local POSIX registry. Both namespace paths were validated
    # by identity_root's registry check; select the matching remote explicitly.
    from cohort_binding import expected_root
    remote = expected_root(REMOTE, identity)
    prior = original.ROOT, original.REMOTE
    try:
        original.ROOT, original.REMOTE = persistent, remote.as_posix()
        return original.run(identity, mode)
    finally:
        original.ROOT, original.REMOTE = prior
