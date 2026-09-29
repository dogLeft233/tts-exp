"""Rebuild formal contrasts from individual views, with separate grouping/bootstrap."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tts_level_visual_calibration_20260927/evaluation'


def main():
    read = lambda name: json.loads((OUT / name).read_text())
    p, scores, summary = read('protocol.json'), read('scores.json'), read('summary.json')
    assert summary['protocol_sha256'] == hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest()
    eligible = {r['id'] for r in p['rows'] if read('receipts/' + r['id'] + '.json')['eligible']}
    assert eligible == {r['id'] for r in scores} == set(summary['common_ids'])
    speakers = sorted({r['speaker'] for r in scores})
    cells = ['N_raw', 'N_LEVEL', 'T_raw', 'T_LEVEL']
    weights = {'N_LEVEL_minus_raw': [-1,1,0,0], 'T_LEVEL_minus_raw': [0,0,-1,1],
               'difference_in_differences': [1,-1,-1,1], 'T_minus_N_raw': [-1,0,1,0],
               'T_minus_N_LEVEL': [0,-1,0,1], **{key:np.eye(4)[i] for i,key in enumerate(cells)}}
    max_error = 0.
    checked = 0
    per_row = {}
    for metric in summary['statistics']:
        matrix = []
        for row in scores:
            values = []
            for key in cells:
                views = row['cells'][key]
                if metric == 'Q':
                    value = views['native']['M'] - views['frozen']['M']
                elif metric == 'reverse':
                    value = views['native']['M'] - views['reversed']['M']
                elif metric == 'frozen_M':
                    value = views['frozen']['M']
                else:
                    value = views['native'][metric]
                values.append(value)
            matrix.append(values)
        matrix = np.array(matrix)
        per_row[metric] = {}
        for label, coeff in weights.items():
            values = (matrix * np.asarray(coeff)[None,:]).sum(1)
            per_row[metric][label] = values
            grouped = np.array([np.mean([v for v,r in zip(values,scores) if r['speaker']==s]) for s in speakers])
            inds = np.random.Generator(np.random.PCG64(20260926)).integers(0,len(grouped),size=(20000,len(grouped)))
            boot = grouped[inds].mean(axis=1)
            expected = {'speaker_mean':grouped.mean(),'speaker_ci95':np.quantile(boot,[.025,.975]),'speaker_ci99':np.quantile(boot,[.005,.995])}
            got = summary['statistics'][metric][label]
            for name,value in expected.items():
                max_error = max(max_error,float(np.max(np.abs(np.asarray(got[name])-value))))
            assert got['n']==len(scores) and got['speakers']==len(speakers)
            for s,value in zip(speakers,grouped):
                max_error = max(max_error,abs(got['per_speaker'][s]-value))
            checked += 1
        # Difference-in-differences must also equal the change in the native T-N gap.
        gap_change=per_row[metric]['T_minus_N_LEVEL']-per_row[metric]['T_minus_N_raw']
        assert np.max(np.abs(gap_change-per_row[metric]['difference_in_differences']))<1e-12
    assert max_error<1e-12
    primary=all(summary['statistics'][metric]['N_LEVEL_minus_raw']['speaker_ci99'][0]>0 for metric in ['Q','M'])
    assert primary == summary['natural_content_improvement_99_conjunction']
    output={'status':'PASS','statistics_checked':checked,'maximum_error':max_error,'support_n':len(scores),'speakers':len(speakers),
            'method':'rebuild contrasts from native/frozen/reversed view cells with independent coefficient vectors and explicit per-speaker grouping; paired DiD algebra verified',
            'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'independent_statistics.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output),flush=True)


if __name__=='__main__':
    main()
