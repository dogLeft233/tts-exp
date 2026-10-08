import hashlib, json, pathlib, zipfile

ROOT = pathlib.Path(r'E:\Documents\tts-exp')
SCRATCH = ROOT / '.codex/scratch'
BASE = SCRATCH / 'dev20_future_cohort_source_stage_readback_attempt02'
RECEIPT = BASE / 'downloads_attempt02/source_stage_qc_attempt01.json'
INDEX = BASE / 'request_index_attempt02.json'
LOCAL = SCRATCH / 'root_future_dev20_full_pcm_qc_attempt01.json'
ZIP = SCRATCH / 'dev20_future_cohort_metadata_attempt06/future_dev20_source_bundle_attempt02.zip'
MANIFEST = SCRATCH / 'dev20_future_cohort_metadata_attempt06/future_dev20_portable_source_manifest_attempt02.json'

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def self_verify(obj, key='self_sha256'):
    payload = {k: v for k, v in obj.items() if k != key}
    actual = digest(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
    assert actual == obj[key], (key, actual, obj[key])
    return actual

index = json.loads(INDEX.read_bytes())
self_verify(index)
for name, pin in index['source_evidence'].items():
    raw = pathlib.Path(pin['path']).read_bytes()
    assert (len(raw), digest(raw)) == (pin['bytes'], pin['sha256']), name
assert len(index['requests']) == 1
request_pin = index['requests'][0]
request_raw = pathlib.Path(request_pin['path']).read_bytes()
assert (len(request_raw), digest(request_raw)) == (request_pin['bytes'], request_pin['sha256'])
request = json.loads(request_raw)
response = json.loads(pathlib.Path(request_pin['path'] + '.response.json').read_bytes())
expect = index['receipt_expectation']
advertisement = json.loads(pathlib.Path(expect['advertised_by_response']['path']).read_bytes())
assert advertisement['ok'] and advertisement['code'] == 0 and not advertisement['stderr']
assert not advertisement['timed_out'] and not advertisement['incomplete_stream'] and not advertisement['stream_error']
assert not any(advertisement['truncated'].values())
advertised = json.loads(advertisement['stdout'])
assert (expect['remote_path'], expect['bytes'], expect['sha256']) == (advertised['receipt_path'], advertised['receipt_bytes'], advertised['receipt_sha256'])
assert request['op'] == 'download' and request['remote'] == expect['remote_path']
assert pathlib.Path(request['local']).resolve() == RECEIPT.resolve()
raw = RECEIPT.read_bytes()
assert (len(raw), digest(raw)) == (expect['bytes'], expect['sha256'])
assert response['ok'] and response['remote'] == request['remote']
assert pathlib.Path(response['local']).resolve() == RECEIPT.resolve()
assert (response['size'], response['sha256']) == (len(raw), digest(raw))
receipt = json.loads(raw)
self_verify(receipt)
assert receipt['status'] == 'ALL_20_SOURCE_AND_RAW_F32_VERIFIED_CREATE_ONLY'
assert all(receipt[k] is False for k in ('asr_or_scores_run', 'model_or_gpu_run', 'excluded_or_confirm_audio_accessed'))
assert receipt['sample_clock_preserved'] is True
assert receipt['stage_root'] == index['remote_stage_root']
local = json.loads(LOCAL.read_bytes())
assert local['status'] == 'PASS_ALL20_FULL_SOURCE_AND_INDEPENDENT_RAW_F32'
assert len(local['rows']) == len(receipt['rows']) == 20
expected_outputs = {}
for before, after in zip(local['rows'], receipt['rows']):
    for oldkey, newkey in (('sample_id','sample_id'), ('speaker_id','speaker_id'), ('N','sample_count'), ('sample_rate','sample_rate_hz'), ('source_bytes','source_bytes'), ('source_sha256','source_sha256'), ('raw_f32_bytes','canonical_bytes'), ('raw_f32_sha256','canonical_sha256')):
        assert before[oldkey] == after[newkey], (before['sample_id'], oldkey)
    assert after['dtype'] == '<f4' and after['sample_rate_hz'] == 16000
    assert after['canonical_bytes'] == 4 * after['sample_count']
    assert 64000 <= after['sample_count'] <= 128000
    source_member = 'source_wav/train/' + after['speaker_id'] + '/' + after['sample_id'] + '.wav'
    f32_member = 'canonical_float32/train/' + after['speaker_id'] + '/' + after['sample_id'] + '.f32'
    assert after['source_member'] == source_member and after['canonical_member'] == f32_member
    expected_outputs[source_member] = (after['sample_id'], after['source_bytes'], after['source_sha256'])
    expected_outputs[f32_member] = (after['sample_id'], after['canonical_bytes'], after['canonical_sha256'])
assert receipt['speaker_counts'] == local['speaker_counts'] == {f'S{i:04d}': 2 for i in range(102,112)}
assert receipt['source_file_count'] == receipt['canonical_f32_file_count'] == 20
assert receipt['source_file_total_bytes'] == local['source_bytes'] == sum(r['source_bytes'] for r in receipt['rows'])
assert receipt['canonical_f32_total_bytes'] == local['canonical_bytes'] == sum(r['canonical_bytes'] for r in receipt['rows'])
zip_raw = ZIP.read_bytes()
assert (len(zip_raw), digest(zip_raw)) == (receipt['bundle']['bytes'], receipt['bundle']['sha256'])
assert digest(zip_raw) == local['zip_raw_sha256']
manifest_raw = MANIFEST.read_bytes()
manifest = json.loads(manifest_raw)
self_verify(manifest)
assert (len(manifest_raw), digest(manifest_raw), manifest['self_sha256']) == (receipt['source_manifest']['bytes'], receipt['source_manifest']['sha256'], receipt['source_manifest']['self_sha256'])
assert digest(manifest_raw) == local['manifest_raw_sha256']
with zipfile.ZipFile(ZIP) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names)) == receipt['bundle']['member_count'] == 28
    assert archive.testzip() is None
    assert archive.read('source_manifest.json') == manifest_raw
    for name in names:
        payload = archive.read(name)
        if name.startswith('source_wav/'):
            assert (len(payload), digest(payload)) == expected_outputs[name][1:]
        else:
            identity = 'source_manifest' if name == 'source_manifest.json' else name
            expected_outputs[name] = (identity, len(payload), digest(payload))
outputs = receipt['written_outputs']
assert len(outputs) == len(expected_outputs) == 48
assert len({o['member'] for o in outputs}) == 48
assert {o['member']: (o['identity'], o['bytes'], o['sha256']) for o in outputs} == expected_outputs
result = {'status': 'PASS_ACTUAL_RECEIPT_ALL20_ROWS_AND48_OUTPUT_PINS', 'receipt_bytes': len(raw), 'receipt_raw_sha256': digest(raw), 'receipt_self_sha256': receipt['self_sha256'], 'ordered_rows': 20, 'written_outputs': 48, 'source_total_bytes': receipt['source_file_total_bytes'], 'canonical_f32_total_bytes': receipt['canonical_f32_total_bytes'], 'full_remote_media_downloaded': False, 'scope': 'ROOT independently verified full actual producer receipt against all local ROOT PCM rows, portable ZIP and pinned metadata; remote WAV/F32 bodies were verified/read back by producer, not downloaded again by ROOT', 'source_sha256': digest(pathlib.Path(__file__).read_bytes())}
with (SCRATCH / 'root_future_dev20_actual_stage_receipt_qc_attempt01.json').open('xb') as stream:
    stream.write(json.dumps(result, sort_keys=True, indent=2).encode('utf-8') + b'\n')
print(json.dumps(result, sort_keys=True))
