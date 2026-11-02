"""Fresh MFCC/audio forward and raw distance QA without overwriting prior references."""
import shutil
from cohort_binding import REMOTE, VERSION, IDS, identity_root, verify_member_pins, sha
import qualify_score_references


def run():
    view=REMOTE/'cohort_versions'/VERSION
    for identity in IDS:
        inputs=identity_root(identity,REMOTE);verify_member_pins(REMOTE,identity)
        source=inputs/'score_reference'/identity
        target=view/'score_reference'/identity;target.mkdir(parents=True,exist_ok=False)
        receipt=source/'score_reference_receipt.json'
        shutil.copyfile(receipt,target/receipt.name);assert sha(receipt)==sha(target/receipt.name)
        for index in range(6):
            (target/f'test_{index:02d}').symlink_to(source/f'test_{index:02d}',target_is_directory=True)
        material=view/'material'/identity;material.parent.mkdir(exist_ok=True)
        assert not material.exists();material.symlink_to(inputs/'material'/identity,target_is_directory=True)
    qualify_score_references.ROOT=view
    qualify_score_references.run()
    for identity in IDS:verify_member_pins(REMOTE,identity)


if __name__=='__main__':run()
