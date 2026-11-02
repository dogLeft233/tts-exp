"""One CPU codec-observer comparison from six already qualified tracks."""
from pathlib import Path, PurePosixPath
from collections import defaultdict
import hashlib
import json
import numpy as np

A = Path(__file__).resolve().parent
RB = A/'output_owned_musetalk_readout_attempt02_readback/readout'
REMOTE = PurePosixPath('/home/wjj/tts-audio/tts-exp/.codex/scratch/native_attribute_response_probe_20261015_output_owned_musetalk_attempt01/readout_attempt02')
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
arms = ('native_height_zero','native_height_plus','native_height_minus')
lags = np.arange(-8,9)


def main():
    primary = read(A/'output_owned_geometry_effect_attempt01.json')
    manifest = read(RB/'readout_manifest.json')
    tracks = {}
    for row in manifest['rows']:
        r = read(RB.joinpath(*PurePosixPath(row['row_receipt_path']).relative_to(REMOTE).parts))
        m = r['mouth_motion_h264']
        p = RB.joinpath(*PurePosixPath(m['arrays_npz']).relative_to(REMOTE).parts)
        assert sha(p)==m['arrays_npz_sha256']
        with np.load(p, allow_pickle=False) as z:
            tracks[r['sample_id'],r['arm']] = (z['mouth_aperture_eye_normalized_float64'].copy(),z['tracking_valid_bool'].copy())
    cells = []
    old = read(A/'root_actual_response_qa.json')['scientific_cells']
    for c in primary['cells']:
        sample = c['sample_id']
        n = len(tracks[sample,arms[0]][0])
        common = np.arange(n)*640 < {'BAC009S0022W0123':51376,'BAC009S0024W0122':56561}[sample]
        for arm in arms:common &= tracks[sample,arm][1]
        endpoints, curves = {}, {}
        for arm in arms:
            y = tracks[sample,arm][0]
            z = (y-y[common].mean())/y[common].std()
            endpoints[arm], curves[arm] = {}, {}
            for scale, values in [('raw',y),('z',z)]:
                groups = defaultdict(list)
                for token in c['tokens']:
                    if token['analysis_status']!='SUPPORTED_ALL_3_ARMS_ALL_17_ABSOLUTE_LAGS':continue
                    core = np.asarray(token['retained_core_frame_indices'])
                    assert np.all(common[core[:,None]+lags[None,:]])
                    groups[token['class'],token['basephone']].append(np.asarray([values[core+lag].mean() for lag in lags]))
                by_class = {k:np.mean(np.stack([np.mean(np.stack(v),axis=0) for (klass,phone),v in groups.items() if klass==k]),axis=0) for k in ('high','low')}
                contrast = by_class['high']-by_class['low']
                cp = float(contrast[12]);hp = cp-float((contrast[8]+contrast[16])/2)
                endpoints[arm][scale] = {'Cp':cp,'Hp':hp}
                curves[arm][scale] = contrast.tolist()
        zero_ok = all(endpoints[arms[0]][s][k]>1e-9 for s in ('raw','z') for k in ('Cp','Hp'))
        ordered = all(endpoints[arms[1]][s][k]-endpoints[arms[0]][s][k]>1e-9 and endpoints[arms[0]][s][k]-endpoints[arms[2]][s][k]>1e-9 for s in ('raw','z') for k in ('Cp','Hp'))
        before = next(x for x in old if x['model']=='musetalk' and x['sample_id']==sample)
        cells.append({'sample_id':sample,'p':4,'common_valid_frames':int(common.sum()),'zero_control':zero_ok,
                      'new_owned_operator_h264_bidirectional_pass':zero_ok and ordered,
                      'new_owned_operator_h264_endpoints':endpoints,'new_owned_operator_h264_curves':curves,
                      'old_operator_h264_bidirectional_pass':before['frozen_response_pass'],
                      'old_operator_h264_endpoints':before['endpoints'],
                      'new_owned_operator_precodec_endpoints':c['fixed_p_endpoints']})
    out = A/'root_h264_functional_comparison.json'
    v = {'schema':'root_codec_observer_functional_comparison_v1','status':'COMPLETE_SIX_EXISTING_TRACKS_ONLY',
         'qa_source_sha256':sha(Path(__file__)),'verified_primary_qa_sha256':sha(A/'root_actual_output_owned_response_qa.json'),
         'method':'Same original S17 support, absolute [-8,8], fixed p4, raw and each arm own common-mask z; no detection or scoring replay.',
         'cells':cells,'interpretation':'Old-operator/precodec cell is unavailable because originals were deleted; compare new mapping under both available observers without rewriting the frozen primary.'}
    with out.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'sha256':sha(out),'cells':[{'sample':c['sample_id'],'old_h264_pass':c['old_operator_h264_bidirectional_pass'],'new_h264_pass':c['new_owned_operator_h264_bidirectional_pass'],'endpoints':c['new_owned_operator_h264_endpoints']} for c in cells]}))


if __name__=='__main__':main()
