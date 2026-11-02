"""H.264 viewing copy only; the scored FFV1/PCM files remain authoritative."""
import json
import subprocess

from common import ROOT,DATA,label,save_json,sha


def main():
    assert json.loads((ROOT/'independent_results_qa.json').read_text())['passed']
    out=ROOT/'preview';out.mkdir(exist_ok=False)
    manifest=json.loads((DATA/'manifest.json').read_text())
    clip=next(c for c in manifest['eval_clips'] if c['split']=='test')
    test=next(s for s in manifest['splits'] if s['name']=='test')
    start=clip['source_start_frame']-test['supervision_start_frame']
    cmd=['ffmpeg','-v','error','-y'];filters=[]
    gains=[-6,0,6];i=0
    for g in gains:
        for h in gains:
            path=ROOT/'renders'/('train_'+label(g))/('test_'+label(h))/'original_pcm.mkv'
            cmd+=['-threads','1','-i',str(path)]
            filters.append(f'[{i}:v]trim=start_frame={start}:end_frame={start+250},setpts=PTS-STARTPTS,scale=384:384,pad=384:416:0:32:black,drawtext=text=\'train {g:+d} / input {h:+d} dB\':x=12:y=8:fontsize=18:fontcolor=white[v{i}]')
            i+=1
    audio=DATA/clip['score_audio_path'];cmd+=['-i',str(audio)]
    layout='|'.join(f'{(i%3)*384}_{(i//3)*416}' for i in range(9))
    filters.append(''.join(f'[v{i}]' for i in range(9))+f'xstack=inputs=9:layout={layout}[grid]')
    partial=out/'first_clip_3x3.mp4.partial';final=out/'first_clip_3x3.mp4'
    cmd+=['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[grid]','-map','9:a:0',
          '-t','10','-r','25','-c:v','libx264','-preset','fast','-crf','19','-threads','2','-pix_fmt','yuv420p',
          '-c:a','aac','-b:a','160k','-movflags','+faststart','-f','mp4',str(partial)]
    subprocess.run(cmd,check=True);partial.replace(final)
    save_json(out/'preview_receipt.json',{'file':final.name,'sha256':sha(final),'clip_id':clip['id'],
         'source_start_frame':clip['source_start_frame'],'rows_training_gain_db':gains,'columns_input_gain_db':gains,
         'audio':'original natural audio, AAC viewing copy','scored':False,'formal_media':'FFV1 plus exact original PCM16'})


if __name__=='__main__':main()
