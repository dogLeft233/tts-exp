#!/usr/bin/env python3
"""Read-only identity/repeat render and SyncNet-track noise diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any
import numpy as np


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def command_bytes(argv: list[str]) -> bytes:
    p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    if p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {argv}; stderr={p.stderr.decode(errors='replace')[-2000:]}")
    return p.stdout


def probe(path: Path) -> dict[str,Any]:
    return json.loads(command_bytes(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))


def decoded_rgb(path: Path) -> tuple[np.ndarray,dict[str,Any]]:
    info=probe(path)
    streams=[s for s in info.get('streams',[]) if s.get('codec_type')=='video']
    if len(streams)!=1: raise RuntimeError(f"expected one video stream: {path}")
    st=streams[0]; h=int(st['height']); w=int(st['width'])
    raw=command_bytes(['ffmpeg','-v','error','-nostdin','-i',str(path),'-map','0:v:0','-vsync','0','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
    frame_bytes=h*w*3
    if len(raw)%frame_bytes: raise RuntimeError(f"decoded video byte count is not frame aligned: {path}")
    frames=np.frombuffer(raw,dtype=np.uint8).reshape(-1,h,w,3)
    return frames,{'width':w,'height':h,'decoded_frame_count':int(frames.shape[0]),'codec':st.get('codec_name'),
                   'r_frame_rate':st.get('r_frame_rate'),'avg_frame_rate':st.get('avg_frame_rate'),
                   'container_duration':info.get('format',{}).get('duration')}


def decoded_pcm16(path: Path) -> np.ndarray:
    raw=command_bytes(['ffmpeg','-v','error','-nostdin','-i',str(path),'-map','0:a:0','-vn','-ac','1','-ar','16000','-acodec','pcm_s16le','-f','s16le','pipe:1'])
    if len(raw)%2: raise RuntimeError(f"decoded PCM16 byte count is not sample aligned: {path}")
    return np.frombuffer(raw,dtype='<i2')


def array_delta(a: np.ndarray,b: np.ndarray) -> dict[str,Any]:
    if a.shape!=b.shape:
        return {'same_shape':False,'shape_a':list(a.shape),'shape_b':list(b.shape)}
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    result={'same_shape':True,'exact':bool(np.array_equal(a,b)),'element_count':int(a.size),
            'changed_element_count':int(np.count_nonzero(a!=b)),'max_abs_difference':float(d.max()) if d.size else 0.0,
            'mean_abs_difference':float(d.mean()) if d.size else 0.0,
            'rms_difference':float(np.sqrt(np.mean(d*d))) if d.size else 0.0}
    if a.ndim>=1 and a.shape[0]>0:
        changed=np.any(a!=b,axis=tuple(range(1,a.ndim))) if a.ndim>1 else a!=b
        result['changed_frame_or_row_count']=int(np.count_nonzero(changed))
    return result


def load_npz_matrices(path: Path) -> dict[int,np.ndarray]:
    out={}
    with np.load(path,allow_pickle=False) as data:
        for key in data.files:
            if key.startswith('track_') and key.endswith('_distance_by_query_lag'):
                tid=int(key.split('_')[1]); out[tid]=np.asarray(data[key],dtype=np.float32)
        if 'lag_offsets_frames' in data.files:
            lag=data['lag_offsets_frames'].astype(np.int32).tolist()
        else: lag=None
    return out,lag


def track_context(score: dict[str,Any], source_video: Path, matrix_id: str) -> list[dict[str,Any]]:
    active=Path(score['activesd_pickle_path'])
    data_dir=active.parents[2]
    reference=active.parent.name
    tracks_path=active.parent/'tracks.pckl'
    with tracks_path.open('rb') as f: track_records=pickle.load(f)
    matrices,lag=load_npz_matrices(Path(score['distance_npz_path']))
    crop_root=data_dir/'pycrop'/reference
    crop_paths=sorted(crop_root.glob('0*.avi'))
    source_audio=decoded_pcm16(source_video)
    result=[]
    count=max(len(track_records),len(crop_paths),len(matrices))
    for i in range(count):
        entry={'track_index':i,'present_in_track_pickle':i<len(track_records),
               'present_in_crop_files':i<len(crop_paths),'present_in_distance_npz':i in matrices}
        if i>=len(track_records) or i>=len(crop_paths) or i not in matrices:
            result.append(entry); continue
        record=track_records[i]
        raw_track=record.get('track',record) if isinstance(record,dict) else record
        frames=np.asarray(raw_track['frame'],dtype=np.int64).reshape(-1)
        bbox=np.asarray(raw_track['bbox'],dtype=np.float32)
        matrix=matrices[i]
        crop=crop_paths[i]
        crop_video,crop_video_info=decoded_rgb(crop)
        crop_pcm=decoded_pcm16(crop)
        start_frame=int(frames[0]); end_frame_exclusive=int(frames[-1])+1
        start_sample=start_frame*640; end_sample=end_frame_exclusive*640
        expected_slice=source_audio[start_sample:end_sample]
        query_frames=frames[:matrix.shape[0]].astype(np.int32)
        entry.update({'track_frame_count':int(frames.size),'track_frame_first':start_frame,'track_frame_last_inclusive':int(frames[-1]),
            'track_frame_end_exclusive':end_frame_exclusive,'track_fps_num':25,'track_fps_den':1,
            'track_start_timestamp_num':start_frame,'track_start_timestamp_den':25,
            'track_audio_start_sample_16k':start_sample,'track_audio_end_sample_exclusive_16k':end_sample,
            'bbox_shape':list(bbox.shape),'bbox_float32_sha256':hashlib.sha256(np.ascontiguousarray(bbox,dtype='<f4').tobytes()).hexdigest(),
            'query_count':int(matrix.shape[0]),'query_frame_first':int(query_frames[0]) if query_frames.size else None,
            'query_frame_last':int(query_frames[-1]) if query_frames.size else None,
            'query_frame_map_sha256':hashlib.sha256(np.ascontiguousarray(query_frames,dtype='<i4').tobytes()).hexdigest(),
            'query_timestamp_map':{'numerator_is_absolute_video_frame_index':True,'denominator':25},
            'audio_feature_window':{'mfcc_frame_start':'query_frame_index*4','mfcc_frame_count':20,
                                    'mfcc_hop_samples':160,'audio_sample_rate_hz':16000},
            'cropped_track_avi_path':str(crop),'cropped_track_avi_sha256':sha256(crop),
            'cropped_track_video':crop_video_info,'cropped_track_pcm16_sha256':hashlib.sha256(crop_pcm.astype('<i2',copy=False).tobytes()).hexdigest(),
            'cropped_track_pcm16_sample_count':int(crop_pcm.size),
            'expected_source_mux_pcm16_slice_sha256':hashlib.sha256(expected_slice.astype('<i2',copy=False).tobytes()).hexdigest(),
            'expected_source_mux_pcm16_slice_sample_count':int(expected_slice.size),
            'crop_audio_exact_expected_frame_slice':bool(np.array_equal(crop_pcm,expected_slice)),
            'lag_offsets_frames':lag,'distance_matrix_shape':list(matrix.shape),
            'distance_matrix_sha256':hashlib.sha256(np.ascontiguousarray(matrix,dtype='<f4').tobytes()).hexdigest()})
        result.append(entry)
    return result


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-root',default='/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/run')
    ap.add_argument('--output',default='/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/control/identity_repeat_noise_diagnostics.json')
    args=ap.parse_args(); root=Path(args.run_root); out=Path(args.output)
    if out.exists(): raise FileExistsError(f'refusing to overwrite {out}')
    manifest_path=root/'engineering_qc_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    rows={r['method_id']:r for r in manifest['rows']}
    a=rows['identity']; b=rows['identity_repeat']
    va,via=decoded_rgb(Path(a['generated_video_path'])); vb,vib=decoded_rgb(Path(b['generated_video_path']))
    ma,la=load_npz_matrices(Path(a['fresh_syncnet']['distance_npz_path']))
    mb,lb=load_npz_matrices(Path(b['fresh_syncnet']['distance_npz_path']))
    mat=[]
    for tid in sorted(set(ma)|set(mb)):
        if tid not in ma or tid not in mb: mat.append({'track_index':tid,'present_in_identity':tid in ma,'present_in_repeat':tid in mb}); continue
        mat.append({'track_index':tid,**array_delta(ma[tid],mb[tid]),'lag_axis_identity':la,'lag_axis_repeat':lb})
    ca=a['fresh_syncnet']['tracks']; cb=b['fresh_syncnet']['tracks']; curves=[]
    for i in range(max(len(ca),len(cb))):
        if i>=len(ca) or i>=len(cb): curves.append({'track_index':i,'present_in_identity':i<len(ca),'present_in_repeat':i<len(cb)}); continue
        av=np.asarray(ca[i]['lag_distance_mean_float32'],dtype=np.float32); bv=np.asarray(cb[i]['lag_distance_mean_float32'],dtype=np.float32)
        curves.append({'track_index':i,**array_delta(av,bv),
            'Cbest_identity':ca[i]['Cbest'],'Cbest_repeat':cb[i]['Cbest'],'delta_Cbest_identity_minus_repeat':float(ca[i]['Cbest']-cb[i]['Cbest']),
            'Canchor_identity':ca[i]['Canchor'],'Canchor_repeat':cb[i]['Canchor'],'delta_Canchor_identity_minus_repeat':float(ca[i]['Canchor']-cb[i]['Canchor']),
            'best_lag_frames_identity':ca[i]['best_lag_frames'],'best_lag_frames_repeat':cb[i]['best_lag_frames'],
            'official_av_offset_frames_identity':ca[i]['official_av_offset_frames'],'official_av_offset_frames_repeat':cb[i]['official_av_offset_frames']})
    report={'schema_version':'cross_tfg_identity_repeat_noise_diagnostics_v1','engineering_only':True,
      'candidate_selection_eligible':False,'generated_unix':time.time(),'manifest_path':str(manifest_path),
      'manifest_status':manifest['status'],'identity_repeat_seed_equal':a['seed']==b['seed'],
      'source_audio_pcm_sha256_equal':a['generator_audio_pcm_sha256']==b['generator_audio_pcm_sha256'],
      'identity_repeat_mux_audio_exact_natural_both':bool(a['mux_audio_readback']['matches_original_natural_pcm'] and b['mux_audio_readback']['matches_original_natural_pcm']),
      'generated_video':{'identity_path':a['generated_video_path'],'identity_file_sha256':a['generated_video_sha256'],
        'repeat_path':b['generated_video_path'],'repeat_file_sha256':b['generated_video_sha256'],
        'identity_decode':via,'repeat_decode':vib,'decoded_rgb_delta':array_delta(va,vb)},
      'syncnet_matrix_delta':mat,'syncnet_curve_and_metrics_delta':curves,
      'official_selected_track_context':{'identity':track_context(a['fresh_syncnet'],Path(a['muxed_video_path']),'identity'),
                                         'identity_repeat':track_context(b['fresh_syncnet'],Path(b['muxed_video_path']),'identity_repeat')},
      'note':'Same-seed identity rerenders are diagnostic noise only; differences do not invalidate rows or authorize candidate selection.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_suffix(out.suffix+'.tmp')
    temp.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,out)
    print(json.dumps({'status':'WRITTEN','output':str(out),'video_decode':report['generated_video']['decoded_rgb_delta'],
                      'matrix':mat,'curves':curves},ensure_ascii=False,sort_keys=True))
    return 0

if __name__=='__main__': raise SystemExit(main())