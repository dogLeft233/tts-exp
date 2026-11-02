"""Permit old CoGenAV feature hashes only when scientific settings and bytes match."""
import json
from prepare import RUN,sha

def compatible(old,new):
    ignored={'sources','input_manifest_sha256','reuse'}
    return {k:v for k,v in old.items() if k not in ignored}=={k:v for k,v in new.items() if k not in ignored}

def verified_parent_protocol(current):
    old=RUN/'cogenav/reuse_protocol_original.json'
    receipt=json.loads((RUN/'reuse_manifest.json').read_text())
    assert sha(old)==receipt['cogenav_original_protocol_sha256']
    assert compatible(json.loads(old.read_text()),current),'Changed CoGenAV scientific configuration'
    expected=[r for r in receipt['files'] if r['path'].startswith('cogenav/embeddings/')]
    assert len(expected)==84
    for r in expected:assert sha(RUN/r['path'])==r['sha256']
    return {sha(old)}
