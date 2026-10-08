import datetime, hashlib, json, math
from pathlib import Path

root = Path('E:/Documents/tts-exp')
packet = root / '.codex/scratch/candidate_v3_cpu_preflight_attempt03'
receipt_path = packet / 'downloads_attempt02_revision02/cpu_preflight_receipt_attempt03.json'
index = json.loads((packet / 'request_index_attempt02.json').read_bytes())
raw = receipt_path.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert len(raw) == 45226 and digest == '8c44b1c80a08e48465f25672ef6b990002653ff7d250ede7369be13dab3a3b09'
r = json.loads(raw)
canonical = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
payload = dict(r); embedded = payload.pop('receipt_sha256')
assert hashlib.sha256(canonical(payload)).hexdigest() == embedded == '66add979541447ec5f8a525576196fd34d9d25306a1c0126fa7957988e3d0486'
assert r['status'] == 'PASS_CPU_PREFLIGHT' and r['failure'] is None
run_request = next(x for x in index['requests'] if x['name'] == 'q_run_strict_cpu_preflight')
response = json.loads(Path(run_request['path'] + '.response.json').read_bytes())
assert response['ok'] and response['code'] == 0 and not response['stderr']
assert not response['timed_out'] and not response['incomplete_stream'] and not response['stream_error']
assert response['truncated'] == {'stdout':False, 'stderr':False}
stdout = json.loads(response['stdout'])
assert stdout['status'] == r['status'] and stdout['receipt_bytes'] == len(raw)
assert stdout['receipt_sha256'] == digest and stdout['receipt_self_sha256'] == embedded
download_request = index['requests'][-1]
download = json.loads(Path(download_request['path'] + '.response.json').read_bytes())
qdownload = json.loads(Path(download_request['path']).read_bytes())
assert download['ok'] and download['remote'] == qdownload['remote'] and download['local'] == qdownload['local']
assert download['size'] == len(raw) and download['sha256'] == digest
for pin in index['source_evidence'].values():
    actual = Path(pin['path']).read_bytes()
    assert len(actual) == pin['bytes'] and hashlib.sha256(actual).hexdigest() == pin['sha256']
plan = json.loads(Path(index['candidate_plan']['path']).read_bytes())
expected = {}
frames = {}
for row in plan['rows']:
    sid = row['sample_id']; natural = row['natural_pcm']
    if sid in expected: assert expected[sid] == natural
    expected[sid] = natural
    if row['model_id'] == 'wav2lip_gan_bac9a81e':
        count = row['wav2lip_native_frame_binding']['expected_inference_frame_count']
        if sid in frames: assert frames[sid] == count
        frames[sid] = count
assert len(expected) == len(frames) == 20
assert r['inputs']['dev_sample_ids'] == sorted(expected) and r['inputs']['planned_rows'] == 180
assert r['inputs']['plan']['sha256'] == index['candidate_plan']['raw_sha256']
assert r['inputs']['plan']['plan_sha256'] == index['candidate_plan']['plan_sha256']
assert r['inputs']['plan_manifest']['sha256'] == index['candidate_manifest']['raw_sha256']
assert r['inputs']['plan_manifest']['manifest_sha256'] == index['candidate_manifest']['manifest_sha256']
assert r['inputs']['candidate_config_cli03']['sha256'] == index['source_evidence']['candidate_config03']['sha256']
assert r['inputs']['cli02_to_cli03_config_change'] == {'only_changed_field':'source_pins.cli', 'projection_equal_after_cli_pin_update':True}
runtime = r['runtime']; assert runtime['device'] == 'cpu' and runtime['strict_process_count'] == 1
assert runtime['rows_reused_one_torch_process'] and not runtime['torch_cuda_initialized_before_rows']
assert runtime['python_version'] == [3,10,12] and runtime['torch_version'] == '2.5.1+cu124' and runtime['numpy_version'] == '1.26.4'
assert runtime['sys_executable'] == '/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python' and runtime['sys_executable_resolved'] == '/usr/bin/python3.10'
assert not runtime['user_site_enabled'] and runtime['python_no_user_site'] == '1'
assert runtime['bootstrap']['sha256'] == index['source_evidence']['strict_bootstrap03']['sha256']
policy = runtime['bootstrap_policy']; assert policy['torch_num_threads'] == policy['torch_num_interop_threads'] == 1
assert policy['torch_use_deterministic_algorithms'] and policy['torch_cudnn_deterministic']
assert not policy['torch_allow_tf32'] and not policy['torch_cudnn_benchmark']
scope = r['scope']; assert scope['candidate_audio_rows_completed'] == scope['candidate_audio_rows_expected'] == 20
assert scope['same_profile_reference_rows_completed'] == scope['same_profile_reference_rows_expected'] == 1
for key in ('candidate_full180_started','candidate_selected','confirm_audio_read','confirm_effects_read','cuda_initialized','gpu_used','scorer_called','source_audio_modified','video_rendered'): assert scope[key] is False
rows = r['candidate_rows']; assert len(rows) == 21
assert rows[0]['kind'] == 'same_profile_reference' and rows[0]['sample_id'] == 'BAC009S0052W0121'
assert [x['sample_id'] for x in rows[1:]] == sorted(expected)
verified = []
for row in rows:
    sid = row['sample_id']; inp = row['input']
    assert row['strict_policy_preserved'] and row['cuda_initialized_after_row'] is False
    assert inp['channels'] == row['output_channels'] == 1
    assert inp['sample_rate_hz'] == inp['source_sample_rate_hz'] == row['output_sample_rate_hz'] == 16000
    assert inp['sample_count'] == inp['source_frames'] == row['output_sample_count']
    assert inp['resampler'] == 'none'
    assert math.isfinite(row['output_rms']) and abs(row['output_rms'] - 0.0376838172675746) < 1e-8
    assert math.isfinite(row['output_peak']) and row['output_peak'] > 0
    assert row['output_wav']['bytes'] == 44 + 4 * row['output_sample_count']
    for pin in (row['output_wav'], row['cli_metadata']):
        assert len(pin['sha256']) == 64 and pin['path'].startswith(index['output']['root'] + '/')
    assert len(row['output_pcm_float32_sha256']) == 64
    if row['kind'] == 'candidate_dev':
        src = expected[sid]
        assert inp['path'] == src['path'] and inp['sha256'] == src['file_sha256']
        assert inp['pcm_float32_sha256'] == src['pcm_sha256'] and inp['sample_count'] == src['sample_count']
        assert row['native_expected_frame_count'] == frames[sid]
    else:
        assert row['native_expected_frame_count'] is None
        assert row['output_wav']['sha256'] == index['reference']['output_wav']['sha256']
    verified.append({'sample_id':sid,'kind':row['kind'],'sample_count':row['output_sample_count'],'pcm_sha256':row['output_pcm_float32_sha256'],'expected_w2l_frames':row['native_expected_frame_count']})
comparison = r['reference_comparison']; assert comparison['exact_pcm_equal'] and not comparison['exact_pcm_equality_is_gate']
assert comparison['max_abs_difference'] == comparison['rmse_difference'] == 0
config = json.loads(Path(index['source_evidence']['candidate_config03']['path']).read_bytes())
for p in r['staged_provenance']:
    src = p['source']; isolated = p['isolated_copy']; assert src['bytes'] == isolated['bytes'] and src['sha256'] == isolated['sha256']
    assert src['path'].startswith(index['repo_root'] + '/runs/') and isolated['path'].startswith(index['task_root'] + '/project/runs/')
assert r['resource']['filesystem_free_bytes_after'] >= r['resource']['reserve_bytes']
result = {'schema':'root_actual_cpu03_receipt_qc_v1','status':'PASS_ROOT_ACTUAL_CPU03_RECEIPT_QC','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'receipt_raw_sha256':digest,'receipt_self_sha256':embedded,'scope':scope,'reference_pcm_exact_equal':True,'reference_max_abs_difference':0.0,'rows':verified,'receipt_producer_audio_checks_reviewed':True,'root_downloaded_all_21_audio_payloads':False,'native_video_or_score_evidence':False}
out = root / '.codex/scratch/root_candidate_cpu03_actual_receipt_qc_attempt01.json'
assert not out.exists(); out.write_bytes(json.dumps(result,sort_keys=True,indent=2).encode()+b'\n')
print(json.dumps({'status':result['status'],'checked_rows':len(rows),'reference_pcm_exact_equal':True,'qc_bytes':out.stat().st_size,'qc_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
