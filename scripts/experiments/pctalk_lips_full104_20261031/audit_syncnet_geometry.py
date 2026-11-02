"""Read official stdout, distances and tracks; no model inference or media changes."""
import json
import pickle
import re
import subprocess
import numpy as np
from prepare import RUN,write

def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=w*h
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union

def main():
    inputs=json.loads((RUN/'scoring_inputs.json').read_text())['unique_rows']
    scores={r['id']:r for r in json.loads((RUN/'syncnet/scores.json').read_text())}
    pre={r['image_sha256']:r for r in json.loads((RUN/'face_preflight.json').read_text())['portraits']}
    records=[]
    for row in inputs:
        item=row['id'];score=scores[item]
        curve=np.load(RUN/'syncnet'/(item+'_distances.npy')).astype(np.float64).mean(0)
        c=float(np.median(curve)-curve.min());assert abs(c-score['C'])<2e-5
        log=(RUN/'logs'/('syncnet_'+item+'.log')).read_text()
        stdout=[float(s) for s in re.findall(r'Confidence\s*:\s*([-+.\deE]+)',log)]
        assert len(stdout)==1 and abs(stdout[0]-score['C'])<.001
        tracks=pickle.loads((RUN/'syncnet/work/pywork'/item/'tracks.pckl').read_bytes())
        assert len(tracks)==1
        track=tracks[0]['track'];bbox=np.asarray(track['bbox']).mean(0)
        face=pre[row['image_sha256']];expected=face['faces'][face['selected_index']]['bbox']
        overlap=iou(bbox,expected)
        # Geometric audit of identity selection, not an additional scoring filter.
        assert overlap>.35,(item,overlap,bbox,expected)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(RUN/'syncnet/work/pyavi'/item/'video.avi')]))
        acodec=next(s['codec_name'] for s in probe['streams'] if s['codec_type']=='audio')
        records.append(dict(id=item,mean_syncnet_bbox=bbox.tolist(),input_main_face_bbox=expected,iou=float(overlap),
            track_frames=len(track['frame']),track_start=int(track['frame'][0]),track_end=int(track['frame'][-1]),
            native_frames=row['actual_video_frames'],offset=score['offset'],float64_C=c,official_stdout_C=stdout[0],
            C_error=abs(c-score['C']),official_internal_audio_codec=acodec))
    write(RUN/'syncnet_geometry_audit.json',dict(status='PASS',media=len(records),minimum_main_face_iou=min(r['iou'] for r in records),
        maximum_C_error=max(r['C_error'] for r in records),offsets=sorted({r['offset'] for r in records}),records=records,
        scope='Read-only raw array/stdout/main-face-track verification; unchanged score support and no offset correction'))
    print('SYNCNET_GEOMETRY_PASS',len(records),min(r['iou'] for r in records))

if __name__=='__main__':main()
