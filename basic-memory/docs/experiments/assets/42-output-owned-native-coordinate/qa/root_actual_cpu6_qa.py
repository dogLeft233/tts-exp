"""One actual-array scientific readback, reusing already qualified driver bodies."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
READBACK=HERE/'readback_attempt04'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=read(READBACK/'receipt.json')
m=read(HERE/'cpu6_input_manifest_attempt04.json')
admission=read(HERE/'root_cpu6_execution_admission_attempt04.json')
assert sha(READBACK/'receipt.json')=='892720030a6e78b109607bda0f447af7faeb9080283f8258a6a1bd0cd2dcf12b'
for name,key in [('materialize_native_cpu6_attempt04.py','worker_sha256'),('cpu6_input_manifest_attempt04.json','manifest_sha256'),('native_axis_ditto_cpu_probe_attempt04_return3.py','derived_helper_sha256')]:
    assert sha(READBACK/name)==admission[key]
assert r['status']=='MATERIALIZED_UNMODIFIED_NATIVE_FEATURES_ONLY' and r['torch_cuda_available'] is False and r['cuda_visible_devices']==''
assert len(r['rows'])==6 and r['wav2lip_mel_hop_samples']==200
clips={c['sample_id']:c for c in m['clips']}
canonical=Path(m['canonical_144_metadata']['path'])
scores={x['row_id']:x for x in map(json.loads,canonical.read_text(encoding='utf-8').splitlines())}
seen=set();out=[]
for row in r['rows']:
    sample,model=row['sample_id'],row['model_key'];key=(sample,model);assert key not in seen;seen.add(key)
    clip=clips[sample];N=clip['fixed_rms_driver']['sample_count']
    path=READBACK/Path(row['output_path']).name
    assert sha(path)==row['output_file_sha256']
    with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    f=data['features'];clock=data['native_time_s'];assert list(f.shape)==row['shape'] and str(f.dtype)==row['dtype'] and np.isfinite(f).all()
    rawsha=hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()
    f32sha=hashlib.sha256(np.asarray(f,dtype='<f4').tobytes()).hexdigest()
    assert rawsha==row['feature_array_bytes_sha256'] and f32sha==row['feature_float32_sha256']
    assert row['input_detail']['pcm_float32_sha256']==clip['fixed_rms_driver']['pcm_float32_sha256']
    assert row['input_detail']['file_sha256']==clip['fixed_rms_driver']['file_sha256'] and row['input_detail']['sample_count']==N
    if model=='wav2lip':
        assert f.dtype==np.float64 and f.shape==(N//200+1,80)
        expectedclock=np.arange(len(f))*200/16000
        mel=f.T;chunks=[];j=0
        while True:
            start=int(j*80/25)
            if start+16>mel.shape[1]:chunks.append(mel[:,-16:]);break
            chunks.append(mel[:,start:start+16]);j+=1
        rebuilt=np.asarray(chunks,dtype=np.float32)
        assert np.array_equal(rebuilt,data['wav2lip_chunks'])
        assert np.array_equal(np.arange(len(rebuilt))/25,data['wav2lip_chunk_native_time_s'])
        rendererframes=len(rebuilt);dimension=80
    elif model=='musetalk':
        assert f.dtype==np.float32 and f.shape==(N//640,50,384)
        expectedclock=np.arange(len(f))/25;rendererframes=len(f);dimension=19200
        assert row['frontend_detail']['detail']['device']=='cpu'
    else:
        assert model=='ditto' and f.dtype==np.float32 and f.shape==((N+639)//640,1024)
        expectedclock=np.arange(len(f))/25;rendererframes=len(f);dimension=1024
        detail=row['frontend_detail']['detail'];assert detail['device']=='cpu' and detail['onnx_providers']==['CPUExecutionProvider']
    assert clock.shape==(len(f),) and np.array_equal(clock,expectedclock)
    old=next(x for x in clip['canonical_rms_rows'] if x['model_id'].startswith(model))
    baseline=scores[old['row_id']]
    assert baseline['render']['frame_count']==rendererframes
    norms=np.linalg.norm(f.reshape(len(f),-1).astype(np.float64),axis=1)
    out.append({'sample_id':sample,'model':model,'file_sha256':sha(path),'shape':list(f.shape),'dtype':str(f.dtype),'dimension':dimension,
                'native_clock_hz':80 if model=='wav2lip' else 25,'renderer_frames':rendererframes,'canonical_baseline_frame_count_matches':True,
                'min_frame_norm':float(norms.min()),'zero_frame_norms':int((norms<=1e-12).sum()),'feature_min':float(f.min()),'feature_max':float(f.max()),
                'RMS':row['input_detail']['rms'],'pcm_sha256':row['input_detail']['pcm_float32_sha256']})
assert seen=={(s,model) for s in clips for model in ['wav2lip','musetalk','ditto']}
result={'schema':'root_actual_cpu6_feature_qa_v1','status':'PASS_ACTUAL_SIX_NATIVE_ARRAYS_PCM_CLOCK_AND_RAWMEL_CHUNKS',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),
        'receipt_sha256':sha(READBACK/'receipt.json'),'admission_sha256':sha(HERE/'root_cpu6_execution_admission_attempt04.json'),
        'rows':out,'scope':'Actual6NPZ bodies/arraybytes/shape/dtype/finite/model-specific clock, Wav chunks rebuilt from actual rawmel; source/helper/manifest pins and already qualified PCMdriver bindings. No encoder rerun or implementation tests.',
        'limits':'Features are CPU-derived baselines for future nativehooks; historicalrenderframecount matches do not prove historicalpixel/runtime equivalence. Same-route zero render controls planned.'}
q=HERE/'root_actual_cpu6_qa.json'
with q.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False,indent=2));print('QA_SHA',sha(q))
