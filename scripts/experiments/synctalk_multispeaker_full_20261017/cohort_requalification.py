"""Re-run unchanged loader90/resource18 through the selected execution paths."""
import argparse
import json
from pathlib import Path
import qualify_loading
import qualify_capacity
from cohort_binding import REMOTE, VERSION, IDS, identity_root, selection, verify_member_pins, sha


def run(identity):
    assert identity in IDS
    selection(REMOTE)
    inputs = identity_root(identity, REMOTE)
    verify_member_pins(REMOTE, identity)
    view = REMOTE / 'cohort_versions' / VERSION
    view.mkdir(parents=True, exist_ok=True)
    for name in ['official', 'audio_context.py', 'execution_contract_v1.json', 'execution_unit_tests_attempt02.log']:
        alias = view / name
        if alias.is_symlink():
            assert alias.resolve() == (REMOTE/name).resolve()
        else:
            assert not alias.exists()
            alias.symlink_to(REMOTE/name)
    for name in ['datasets', 'snapshots']:
        directory = view/name
        directory.mkdir(exist_ok=True)
        alias = directory/identity
        if alias.is_symlink():
            assert alias.resolve() == (inputs/name/identity).resolve()
        else:
            assert not alias.exists()
            alias.symlink_to(inputs/name/identity, target_is_directory=True)
    # Only diagnostics are written to the fresh view. All actual dataset()
    # calls still resolve through the exact BASE execution module and registry.
    qualify_loading.ROOT = view
    qualify_capacity.ROOT = view
    qualify_loading.run(identity)
    qualify_capacity.run(identity)
    new_loading = json.loads((view/'loading_qualification'/identity/'loading_qualification.json').read_text())
    old_loading = json.loads((inputs/'loading_qualification'/identity/'loading_qualification.json').read_text())
    assert new_loading['passed'] and new_loading['cases'] == 90
    assert new_loading['rows'] == old_loading['rows'], 'selected loading path changed a pixel/ray/condition'
    new_capacity = json.loads((view/'capacity_qualification'/identity/'capacity_qualification.json').read_text())
    old_capacity = json.loads((inputs/'capacity_qualification'/identity/'capacity_qualification.json').read_text())
    assert new_capacity['passed'] and new_capacity['optimizer_updates'] == 0 and len(new_capacity['rows']) == 18
    assert [(x['seed'],x['model_tensor_sha256']) for x in new_capacity['initializations']] == [
        (x['seed'],x['model_tensor_sha256']) for x in old_capacity['initializations']]
    for row in new_capacity['rows']:
        assert row['initialization_exact'] and row['parameter_bytes_unchanged']
        assert row['num_rays_argument'] == 65536 and row['max_steps'] == 16 and row['optimizer_updates'] == 0
    verify_member_pins(REMOTE, identity)
    result = dict(identity=identity, passed=True, cohort_version=VERSION,
                  selection_sha256=sha(REMOTE/'cohort_selection.json'),
                  original_90_inputs_and_rays_exact=True, original_initializations_exact=True,
                  full65536_ray_18_resource_cases=True, optimizer_updates=0,
                  selected_data_root=str(inputs), training_admitted=False)
    output = view/'binding_qualification'/identity
    output.mkdir(parents=True, exist_ok=False)
    (output/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--identity',required=True,choices=IDS)
    run(parser.parse_args().identity)
