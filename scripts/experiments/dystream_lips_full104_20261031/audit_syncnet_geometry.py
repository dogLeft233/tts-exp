"""Verify native512 geometry and official SyncNet matrices/stdout, without shifts."""
import json,pickle,re,subprocess
import numpy as np
from prepare import RUN,write

def main():
    rows=json.loads((RUN/'scoring_inputs.json').read_text())['unique_rows']
    scores={r['id']:r for r in json.loads((RUN/'syncnet/scores.json').read_text())};records=[]
    for row in rows:
        item=row['id'];score=scores[item]
        curve=np.load(RUN/'syncnet'/(item+'_distances.npy')).astype(np.float64).mean(0)
        c=float(np.median(curve)-curve.min());assert abs(c-score['C'])<2e-5
        log=(RUN/'logs'/('syncnet_'+item+'.log')).read_text()
        stdout=[float(s) for s in re.findall(r'Confidence\s*:\s*([-+.\deE]+)',log)]
        assert len(stdout)==1 and abs(stdout[0]-score['C'])<.001
        tracks=pickle.loads((RUN/'syncnet/work/pywork'/item/'tracks.pckl').read_bytes());assert len(tracks)==1
        track=tracks[0]['track'];bbox=np.asarray(track['bbox']);assert np.isfinite(bbox).all()
        assert ((bbox[:,2]-bbox[:,0])>0).all() and ((bbox[:,3]-bbox[:,1])>0).all()
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(RUN/'syncnet/work/pyavi'/item/'video.avi')]))
        acodec=next(s['codec_name'] for s in probe['streams'] if s['codec_type']=='audio')
        records.append(dict(id=item,mean_syncnet_bbox=bbox.mean(0).tolist(),track_frames=len(track['frame']),
            track_start=int(track['frame'][0]),track_end=int(track['frame'][-1]),native_frames=row['actual_video_frames'],
            offset=score['offset'],float64_C=c,official_stdout_C=stdout[0],C_error=abs(c-score['C']),official_internal_audio_codec=acodec))
    write(RUN/'syncnet_geometry_audit.json',dict(status='PASS',media=len(records),maximum_C_error=max(r['C_error'] for r in records),
        offsets=sorted({r['offset'] for r in records}),records=records,
        scope='One generated face track, raw matrix/stdout and native track geometry; aligned512 output differs from input portrait geometry, so no source-coordinate IoU assertion. No offset correction.'))
    print('SYNCNET_GEOMETRY_PASS',len(records))

if __name__=='__main__':main()
