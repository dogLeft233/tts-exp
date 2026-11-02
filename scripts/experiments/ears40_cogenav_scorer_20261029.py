"""EARS40 CoGenAV adapter with frozen, explicit bounded source transformations.

Network, audio/visual preprocessing and paper Eq1 are unchanged. Controls are
actual PCM +/-400ms on the natural arm only; saved raw features remain auditable.
"""
import argparse
import json
from pathlib import Path
import sys

import ears40_extra_scorers_20261029 as common

ROOT=common.ROOT
RUN=common.RUN


def transformed_source(source):
    changes=[
        ('"lexical first sources, all N/L06/L17 draws"',
         '"all40 frozen EARS English speakers, exactly one N and one T per speaker"'),
        ('"three families N-L06,N-L17,L06-L17; paired same draw; source-level aggregation"',
         '"one native T-N pair per40 speakers; parent frozen protocol owns inference"'),
        ('"raw audio shifts +/-160ms must move peak by induced delay within40ms in >=80% pilot clip-shift cases; peak-vs-outerlags positive median; otherwise UNVALIDATED, no quality conclusion"',
         '"all40 N raw audio +/-400ms; >=80% relative peaks recover within40ms and >=80% Eq1 decrease; failure retained without stopping T scores"'),
        ('for shift in [-4, 4]:', 'for shift in ([-10, 10] if row["arm"] == "N" else []):'),
        ('sc, _ = lag_curve(sa, v)',
         'sc, _ = lag_curve(sa, v)\n            san = sa / np.maximum(np.linalg.norm(sa, axis=1, keepdims=True), 1e-12)\n            shifted_eq1 = float(np.maximum(0, (san * vn).sum(-1)).mean())'),
        ('"shift_frames": shift, "peak_lag_frames": peak,',
         '"shift_frames": shift, "shift_samples": shift*640, "eq1": shifted_eq1, "eq1_lower": shifted_eq1 < eq1, "peak_lag_frames": peak,'),
        ('audio_samples=len(audio), protocol_sha256=sha(protocol_path))',
         'audio_samples=len(audio), protocol_sha256=sha(protocol_path), input_audio_sha256=row["audio_sha256"], input_video_sha256=row["video_sha256"])'),
    ]
    for old,new in changes:
        if source.count(old)!=1:
            raise ValueError('Expected source anchor exactly once: '+old)
        source=source.replace(old,new)
    return source, [dict(before=old,after=new) for old,new in changes]


def controls_from_scores(scores):
    controls=[]
    for row in scores:
        expected=2 if row['arm']=='N' else 0
        if len(row['raw_shift_controls']) != expected:
            raise ValueError('Expected two natural controls and no TTS controls')
        for control in row['raw_shift_controls']:
            if control['shift_samples'] not in (-6400,6400):
                raise ValueError('Expected real PCM400ms perturbation')
            controls.append(dict(id=row['id'],native=row['eq1'],**control,
                                 lower=control['eq1']<row['eq1']))
    result=common.control_summary(controls)
    result['peak_within_one_frame']=sum(c['pass_within_one_frame'] for c in controls)
    result['relative_peak_status']=('PASS' if result['peak_within_one_frame']>=.8*len(controls)
        else 'CONTROL_FAILED') if controls else 'UNMEASURABLE'
    return result


def run(run_root):
    root=run_root/'cogenav';root.mkdir(exist_ok=True)
    rows=common.validate_rows(common.read(run_root/'inputs.json'))
    old=ROOT/'runs/cogenav_sync_full20_20261024_attempt01'
    for name in ['upstream','weights','autoavsr_detector.py','autoavsr_video_process.py','20words_mean_face.npy']:
        target=old/name;dest=root/name
        assert target.exists(), target
        if not dest.exists():dest.symlink_to(target.resolve(),target_is_directory=target.is_dir())
        else:assert dest.resolve()==target.resolve(), 'Unexpected asset target'
    manifest=[]
    for row in rows:
        for kind in ['audio','video']:
            assert common.sha(row[kind+'_path'])==row[kind+'_sha256'],row['id']
        manifest.append(dict(row,draw=0,qualified=True,score_float32=None,
            features=dict(duration_s=row['native_frames']/25)))
    manifest_path=root/'input_manifest.json'
    if manifest_path.exists():assert common.read(manifest_path)==dict(rows=manifest)
    else:common.write(manifest_path,dict(rows=manifest))
    original=Path(__file__).with_name('cogenav_sync_audit_20261024.py')
    source,changes=transformed_source(original.read_text(encoding='utf-8'))
    snapshot=root/'source_transformed_frozen.py'
    if snapshot.exists():assert snapshot.read_text(encoding='utf-8')==source
    else:snapshot.write_text(source,encoding='utf-8')
    common.write(root/'adapter_receipt.json',dict(script_sha256=common.sha(__file__),
        inherited_source=str(original),inherited_source_sha256=common.sha(original),
        source_transformed_sha256=common.sha(snapshot),changes=changes,
        parent_protocol_sha256=common.sha(run_root/'protocol.json'),
        parent_input_manifest_sha256=common.sha(run_root/'inputs.json'),
        assets={name:common.sha(root/name) for name in ['autoavsr_detector.py','autoavsr_video_process.py','20words_mean_face.npy']}))
    namespace=dict(__name__='ears_cogenav_frozen_adapter',__file__=str(snapshot))
    exec(compile(source,str(snapshot),'exec'),namespace)
    previous=sys.argv
    try:
        sys.argv=[str(snapshot),'--run',str(root),'--manifest',str(manifest_path),'--sources','40','--record-long-clips']
        namespace['main']()
    finally:sys.argv=previous
    scores=common.read(root/'scores.json')
    assert len(scores)==80
    controls=controls_from_scores(scores)
    common.assert_complete_controls(controls['rows'],rows)
    common.write(root/'known_shift_controls.json',controls)
    complete=common.read(root/'extraction_complete.json')
    common.write(root/'status.json',dict(status='COMPLETE',scored=len(scores),
        unmeasurable=complete['unmeasurable'],controls=controls['status'],relative_peak_control=controls['relative_peak_status']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUN)
    args=parser.parse_args();run(args.run.resolve())
