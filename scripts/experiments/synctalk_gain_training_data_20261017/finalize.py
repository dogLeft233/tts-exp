"""Seal the verified preparation and retain a small repository receipt."""
import hashlib
import argparse
import json
import shutil
import tarfile
from pathlib import Path

REPO = Path('E:/Documents/tts-exp')
DATA = Path('F:/Datasets/tts-exp/synctalk_gain_training_20261017')
CODE = REPO / 'scripts/experiments/synctalk_gain_training_data_20261017'
THIN = REPO / 'runs/synctalk_gain_training_data_20261017'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=int, default=1)
    revision = parser.parse_args().revision
    suffix = '' if revision == 1 else f'_v{revision}'
    manifest_name = f'delivery_manifest{suffix}.json'
    task_snapshot = f'task_spec{suffix}.md'
    assert not (DATA / manifest_name).exists()
    local = json.loads((DATA / 'local_complete_qa.json').read_text())
    remote = json.loads((DATA / 'remote_independent_qa.json').read_text())
    clock = json.loads((DATA / 'source_clock_qa.json').read_text())
    cloud = json.loads((DATA / 'cloud_bundle_receipt.json').read_text())
    rawcloud = json.loads((DATA / 'cloud_storage_receipt.json').read_text())
    joined = json.loads((DATA / 'local_remote_join_qa.json').read_text())
    manifest = json.loads((DATA / 'manifest.json').read_text())
    assert local['passed'] and remote['passed'] and clock['passed'] and cloud['passed'] and joined['passed']
    assert rawcloud['independent_linux_cloud_readback']
    assert local['ready_for_engineering_pilot'] and not local['formal_mandarin_cohort_ready']
    spec = REPO / 'basic-memory/Tasks/准备 SyncTalk 训练电平先导数据.md'
    assert spec.exists() and 'status: done' in spec.read_text(encoding='utf-8')
    assert '[ ]' not in spec.read_text(encoding='utf-8'), 'task checkboxes must match actual acceptance'
    shutil.copyfile(spec, DATA / task_snapshot)
    sources = sorted(p for p in CODE.iterdir() if p.is_file())
    sources += [REPO / 'tests/experiments/test_synctalk_gain_training_data_20261017.py']
    archive = DATA / f'repro_sources{suffix}.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as t:
        for path in sources:
            t.add(path, arcname=str(path.relative_to(REPO)).replace('\\', '/'))
        t.add(DATA / task_snapshot, arcname='task_spec.md')
    result = {
        'schema': 'synctalk_gain_data_delivery_v1', 'status': 'DATA_PREPARATION_VERIFIED',
        'revision': revision, 'task_spec_snapshot': task_snapshot,
        'metadata_correction': 'Revision 2 checks five completed task steps; raw data and QA unchanged' if revision == 2 else None,
        'data_root': str(DATA), 'repository_receipt_root': str(THIN),
        'task_spec_permalink': 'tts-exp/tasks/准备-sync-talk-训练电平先导数据',
        'source_matches_official': True, 'split_matches_official': False,
        'natural_real_video_supervision': True, 'natural_speech_with_scalar_gain': True,
        'tts_replacement': False, 'synthetic_video_supervision': False, 'frame_reordering': False,
        'source_archive_sha256': manifest['source_archive_sha256'],
        'official_original_train_frames': 5520, 'official_original_train_seconds': 220.8,
        'new_train_frames': 4150, 'new_train_seconds': 166,
        'data_amount_sufficiency': 'UNVERIFIED',
        'generator_convergence_tested': False, 'training_runs': 0, 'generator_forwards': 0,
        'independent_identities': 1, 'language': 'English',
        'formal_mandarin_cohort_ready': False,
        'ready_for_engineering_pilot': True,
        'training_admission': 'Await separate frozen budget and learning-curve/data-amount protocol; no training started',
        'splits': [{k: s[k] for k in ['name','source_start_frame','source_end_frame_exclusive',
                                    'supervision_frames','audio_seconds','supervision_seconds']} for s in manifest['splits']],
        'gains_db': [-6,0,6], 'arm_count': 9, 'fixed_original_pcm_eval_clips': len(manifest['eval_clips']),
        'verification': {k: local[k] for k in ['source_files_verified','source_bytes_verified',
                         'image_files_decoded','landmark_files_verified','arms_verified','features_verified']},
        'actual_provider_collates': 27, 'zero_gain_full_repeats': 3,
        'contract_rejection_tests_passed': 5, 'contact_sheet_frames_visually_reviewed': 36,
        'source_clock_image_samples': 9, 'source_clock_audio_segments': 3,
        'source_clock_best_offset_samples': [r['best_sample_lag_in_100ms'] for r in clock['audio_clock_samples']],
        'cross_platform_compared_files': joined['derived_files_compared_to_linux'],
        'json_only_newline_difference_count': len(joined['json_newline_differences']),
        'cloud_directory': 'tts-exp-research-models/synctalk-gain-training-data-20261017',
        'cloud_data_verified_by_independent_linux_readback': True,
        'source_file_count_and_identity_are_not_independent_statistical_samples': True,
        'sources': [{'path': str(p.relative_to(REPO)).replace('\\','/'), 'sha256': sha(p)} for p in sources],
        'files': []
    }
    targets = [p for p in DATA.iterdir() if p.is_file()]
    targets += sorted((DATA / 'features_attempt03').rglob('*'))
    for p in targets:
        if p.is_file():
            result['files'].append({'path': str(p.relative_to(DATA)).replace('\\','/'),
                                    'bytes': p.stat().st_size, 'sha256': sha(p)})
    (DATA / manifest_name).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    THIN.mkdir(exist_ok=True)
    for name in [manifest_name,'local_complete_qa.json','remote_independent_qa.json',
                 'source_clock_qa.json','cloud_storage_receipt.json','cloud_bundle_receipt.json',
                 'local_remote_join_qa.json','bundle_receipt.json',task_snapshot]:
        shutil.copyfile(DATA / name, THIN / name)
    (THIN / 'dataset_location.json').write_text(json.dumps({'data_root': str(DATA),
            'reason': 'E/root space constrained; F is persistent, SHM is volatile',
            'cloud_directory': result['cloud_directory']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pointer = {'manifest': manifest_name, 'sha256': sha(DATA / manifest_name), 'revision': revision}
    for destination in [DATA, THIN]:
        (destination / 'current_delivery.json').write_text(json.dumps(pointer,indent=2)+'\n')
    print(json.dumps({'passed':True,'data_root':str(DATA),'delivery_manifest_sha256':sha(DATA/manifest_name),
                      'repro_sources_sha256':sha(archive),'json_newline_differences':result['json_only_newline_difference_count']}))


if __name__ == '__main__':
    main()
