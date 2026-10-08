import json,hashlib,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'.codex/scratch/fit40_d0_gatedc_training_packet_attempt02'
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
raw=(D/'source_receipt_readback_attempt01/downloads_attempt01/source_stage_receipt_attempt02.json').read_bytes()
assert len(raw)==9594 and sha(raw)=='d4a2a5d8e6b8c4edd9c57e13f53bee6d727d77b518bea1311ad8be87896c5161'
v=json.loads(raw);seal=v.pop('self_sha256');assert seal==sha(canon(v))
assert v['status']=='PASS_SOURCE_CLOSURE_EXTRACTED_NO_MODEL_OR_GPU' and len(v['members'])==23 and set(v['scope'].values())=={False}
mraw=(D/'source_bundle_manifest_attempt02.json').read_bytes();m=json.loads(mraw);ms=m.pop('self_sha256');assert ms==sha(canon(m))
assert v['manifest']['bytes']==len(mraw) and v['manifest']['sha256']==sha(mraw) and v['manifest']['self_sha256']==ms
a=(D/'fit40_d0_gatedc_source_attempt02.zip').read_bytes();assert len(a)==v['archive']['bytes'] and sha(a)==v['archive']['sha256']
with zipfile.ZipFile(D/'fit40_d0_gatedc_source_attempt02.zip')as z:
    assert len(z.namelist())==23
    for actual,expected in zip(v['members'],m['members']):
        assert all(actual[k]==expected[k]for k in ['relative_path','bytes','sha256'])
        assert actual['path']==v['destination']+'/'+expected['relative_path']
        b=z.read(expected['archive_path']);assert len(b)==expected['bytes'] and sha(b)==expected['sha256']
        assert (R/expected['relative_path']).read_bytes()==b
o={'status':'PASS_ACTUAL_STAGE_RECEIPT_AND_ALL23_SOURCE_BYTES','receipt_bytes':len(raw),'receipt_raw_sha256':sha(raw),'receipt_self_sha256':seal,'member_count':23,'all_no_gpu_scope_flags_false':True,'source_sha256':sha(Path(__file__).read_bytes())}
with (R/'.codex/scratch/root_d0_actual_source_stage_receipt_qc_attempt01.json').open('xb')as f:f.write((json.dumps(o,sort_keys=True,indent=2)+'\n').encode())
print(json.dumps(o))
