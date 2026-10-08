import json,hashlib,struct,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2]
D=R/'.codex/scratch/dev20_future_cohort_metadata_attempt06'
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(D/'future_dev20_portable_source_manifest_attempt02.json').read_bytes()
assert len(raw)==24072 and sha(raw)=='fff18619b7be8f992002f99c33121ef3eb25acce6aa3caa3da55f398eb3faffa'
m=json.loads(raw); seal=m.pop('self_sha256')
assert sha(json.dumps(m,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())==seal
assert m['selection']['selected_speakers']==[f'S{i:04d}'for i in range(102,112)]
assert m['selection']['performance_or_score_selection'] is False
assert m['selection']['confirm_audio_or_scores_accessed'] is False
assert len(m['rows'])==20 and len({r['sample_id']for r in m['rows']})==20
index=json.loads((R/'.codex/scratch/dev20_future_cohort_upload_packet_attempt02/request_index_attempt02.json').read_bytes())
waves={k[len('waveform_'):]:v for k,v in index['source_evidence'].items()if k.startswith('waveform_')}
zraw=(D/'future_dev20_source_bundle_attempt02.zip').read_bytes()
assert len(zraw)==3548546 and sha(zraw)=='6f9de4953da6fd52a4e916604c21b0fd187ab145fefc92ea47cd8e274a460d9a'
out=[]
with zipfile.ZipFile(D/'future_dev20_source_bundle_attempt02.zip')as z:
    expected={'source_manifest.json'}|{r['bundle_path']for r in m['rows']}|{e['archive_path']for e in m['evidence']}
    assert len(z.namelist())==28 and set(z.namelist())==expected and z.testzip() is None
    assert z.read('source_manifest.json')==raw
    for e in m['evidence']:
        b=z.read(e['archive_path']);assert len(b)==e['bytes'] and sha(b)==e['sha256']
    for r in m['rows']:
        p=Path(waves[r['sample_id']]['path']); b=p.read_bytes()
        assert len(b)==r['bytes']==waves[r['sample_id']]['bytes']
        assert sha(b)==r['sha256']==waves[r['sample_id']]['sha256']
        assert z.read(r['bundle_path'])==b
        assert b[:4]==b'RIFF' and b[8:12]==b'WAVE' and struct.unpack_from('<I',b,4)[0]+8==len(b)
        pos=12;fmt=[];data=[]
        while pos<len(b):
            tag=b[pos:pos+4]; n=struct.unpack_from('<I',b,pos+4)[0];part=b[pos+8:pos+8+n]
            assert len(part)==n
            if tag==b'fmt ':fmt.append(struct.unpack_from('<HHIIHH',part))
            if tag==b'data':data.append(part)
            pos+=8+n+(n&1)
        assert pos==len(b) and fmt==[(1,1,16000,32000,2,16)] and len(data)==1
        pcm=data[0];N=len(pcm)//2
        assert len(pcm)%2==0 and N==r['riff']['sample_count']==r['canonical_float32']['sample_count'] and 64000<=N<=128000
        samples=struct.unpack(f'<{N}h',pcm)
        f32=struct.pack(f'<{N}f',*(v/32768.0 for v in samples))
        assert len(f32)==4*N and sha(f32)==r['canonical_float32']['pcm_sha256']
        assert r['canonical_float32']['resampler']=='none' and r['canonical_float32']['sample_rate_hz']==16000
        out.append({'sample_id':r['sample_id'],'speaker_id':r['speaker_id'],'source_bytes':len(b),'source_sha256':sha(b),'N':N,'sample_rate':16000,'raw_f32_bytes':len(f32),'raw_f32_sha256':sha(f32)})
counts={s:sum(r['speaker_id']==s for r in out)for s in m['selection']['selected_speakers']}
assert set(counts.values())=={2}
o={'status':'PASS_ALL20_FULL_SOURCE_AND_INDEPENDENT_RAW_F32','rows':out,'speaker_counts':counts,'source_bytes':sum(x['source_bytes']for x in out),'canonical_bytes':sum(x['raw_f32_bytes']for x in out),'clock_and_N_preserved':True,'no_asr_or_scores':True,'source_script_sha256':sha(Path(__file__).read_bytes()),'manifest_raw_sha256':sha(raw),'zip_raw_sha256':sha(zraw)}
target=R/'.codex/scratch/root_future_dev20_full_pcm_qc_attempt01.json'
with target.open('xb')as f:f.write((json.dumps(o,sort_keys=True,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in o.items()if k!='rows'}))
