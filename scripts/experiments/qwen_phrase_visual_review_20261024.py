"""Bounded descriptive review of three already-selected phrase cases. No models."""
import hashlib
import html
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/wjj/tts-audio/tts-exp')
SRC = ROOT / 'runs/qwen_phrase_local_20261024_attempt02'
OUT = ROOT / 'runs/qwen_phrase_visual_review_20261024_attempt02'
CASES = [('BAC009S0002W0122', 1, 'S0002_p2'),
         ('BAC009S0014W0121', 0, 'S0014_p1'),
         ('BAC009S0014W0121', 1, 'S0014_p2')]
PINYIN = {'BAC009S0002W0122': ['er','dui','lou','shi','cheng','jiao','yi','zhi','zuo','yong','zui','da','de','xian','gou'],
          'BAC009S0014W0121': ['zhe','he','lou','mian','jia','san','dian','ba','wan','yuan','mei','ping','fang','mi']}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    OUT.mkdir(exist_ok=False)
    write('review_scope.json', {
        'scope': CASES, 'all_draws': True, 'new_inference': False,
        'evidence': 'raw video frames + given-text FA + waveform measurements; no auditory perception',
        'known_score_direction': True, 'blinded': False,
        'mouth_roi': [175, 270, 319, 360],
        'midpoint_sheet': 'one native frame nearest each FA character midpoint',
        'event_sheet': 'every 40ms from FA character onset -160ms through +240ms',
        'timing_warning': 'FA characters are automatic approximations, not phoneme ground truth; raw video clock unshifted',
        'audio': 'phrase duration; 20ms RMS with 10ms hop; relative level to full utterance; no perceptual claims',
        'inputs': {n: sha(SRC/n) for n in ['alignment.json','input_manifest.json','local_metrics.json','presentation_receipts.json']}
    })
    alignment = {x['id']: x for x in json.loads((SRC/'alignment.json').read_text())['audio']}
    assets = {x['id']: x for x in json.loads((SRC/'input_manifest.json').read_text())['audio']}
    metrics = json.loads((SRC/'local_metrics.json').read_text())
    cache, pins, rows, sheets = {}, [], [], []
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)

    def load(key):
        if key not in cache:
            x = assets[key]
            assert sha(x['video_path']) == x['video_sha256']
            assert sha(x['source']) == x['sha256']
            cmd = ['ffmpeg','-v','error','-i',x['video_path'],'-map','0:v:0','-vsync','0',
                   '-vf','format=rgb24,crop=144:90:175:270','-f','rawvideo','-pix_fmt','rgb24','-']
            p = subprocess.run(cmd, capture_output=True, check=True)
            frames = np.frombuffer(p.stdout, np.uint8).reshape(-1,90,144,3)
            assert len(frames) == x['video_frame_count']
            audio, rate = sf.read(x['source'])
            assert rate == 16000 and audio.ndim == 1
            cache[key] = frames, audio, rate
            pins.append({'id': key, 'video': x['video_path'], 'video_sha256': x['video_sha256'],
                         'audio': x['source'], 'audio_sha256': x['sha256'], 'frames': len(frames),
                         'decode_command': cmd, 'exit': p.returncode})
        return cache[key]

    def panel(draw, frames, col, row, t, label):
        pts = np.arange(len(frames))/25
        fi = int(np.argmin(abs(pts-t)))
        tile = Image.fromarray(frames[fi])
        canvas.paste(tile, (120+col*150, 40+row*125))
        draw.text((120+col*150, 40+row*125+91), label, font=font, fill='black')
        draw.text((120+col*150, 40+row*125+107), f'{fi/25:.2f}s f{fi}', font=font, fill='black')
        return {'requested_s': t, 'frame': fi, 'pts_s': fi/25}

    for sid, pi, tag in CASES:
        group = [x for x in metrics if x['sample_id']==sid and x['phrase_index']==pi and x['variant']=='nominal']
        group.sort(key=lambda x: (['N','L06','L17'].index(x['model']), x['draw']))
        assert len(group)==7
        start, end = group[0]['character_span']
        canvas = Image.new('RGB',(120+(end-start)*150,40+7*125),'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((5,5),tag+' FA character MIDPOINTS (not phoneme alignment)',font=font,fill='black')
        sampling = []
        for ri, x in enumerate(group):
            key=x['id']; frames, audio, rate=load(key)
            ts=alignment[key]['alignment'][0]['timestamp']
            draw.text((5,40+ri*125),f"{x['model']} d{x['draw']}",font=font,fill='black')
            samples=[]
            for ci, charidx in enumerate(range(start,end)):
                t=sum(ts[charidx])/2000
                samples.append(panel(draw,frames,ci,ri,t,PINYIN[sid][charidx]))
            sampling.append({'id':key,'samples':samples})
            lo,hi=x['nominal_bounds']; wav=audio[round(lo*rate):round(hi*rate)]
            windows=np.lib.stride_tricks.sliding_window_view(wav,320)[::160]
            rms=np.sqrt(np.mean(windows**2,axis=1)); ref=np.sqrt(np.mean(audio**2))
            levels=20*np.log10(np.maximum(rms,1e-10)/ref)
            row={'id':key,'case':tag,'phrase':x['phrase'],'bounds':[lo,hi],
                 'duration_s':hi-lo,'char_boundaries_ms':ts[start:end],
                 'internal_FA_gap_ms':[ts[i+1][0]-ts[i][1] for i in range(start,end-1)],
                 'relative_rms_db_quantiles':np.quantile(levels,[.1,.5,.9]).tolist(),
                 'contrast':x['contrast'],'matched_distance':x['matched_distance'],
                 'median_background':x['median_background']}
            rows.append(row)
        name=tag+'_midpoints.png';canvas.save(OUT/name)
        sheets.append({'name':name,'sampling':sampling})
        events = [9,11] if sid.endswith('W0122') else ([3] if pi==0 else [7])
        for charidx in events:
            canvas=Image.new('RGB',(120+11*150,40+7*125),'white'); draw=ImageDraw.Draw(canvas)
            title=f'{tag} {PINYIN[sid][charidx]} around FA onset; native video frames, NO lag shift'
            draw.text((5,5),title,font=font,fill='black');sampling=[]
            for ri,x in enumerate(group):
                key=x['id'];frames,_,_=load(key)
                onset=alignment[key]['alignment'][0]['timestamp'][charidx][0]/1000
                draw.text((5,40+ri*125),f"{x['model']} d{x['draw']}",font=font,fill='black')
                vals=[panel(draw,frames,j,ri,onset+dt,f'{dt:+.2f}') for j,dt in enumerate(np.arange(-4,7)*.04)]
                sampling.append({'id':key,'FA_onset_s':onset,'samples':vals})
            name=tag+'_'+PINYIN[sid][charidx]+'_onset.png';canvas.save(OUT/name)
            sheets.append({'name':name,'sampling':sampling})
    write('measurements.json',rows);write('frame_sampling.json',sheets);write('input_receipt.json',pins)
    parts=['<!doctype html><meta charset="utf-8"><title>三处稳定短语的音视频核对</title>',
           '<style>body{font:16px sans-serif;margin:24px}video{width:250px}section{display:flex;gap:12px;flex-wrap:wrap}img{max-width:100%}</style>',
           '<h1>三处稳定短语：全部7次发声</h1><p>时间戳为自动字符对齐；未人工听辨。可调播放速度；逐帧图没有移动视频时间轴。</p>']
    for sid,pi,tag in CASES:
        group=[x for x in rows if x['case']==tag]
        parts.append('<h2>'+html.escape(group[0]['phrase'])+'</h2><section>')
        for x in group:
            media=f"../{SRC.name}/clips/{x['id']}_p{pi+1}"
            parts.append(f'<div><h3>{x["id"]}</h3><video controls preload="metadata" src="{media}.mp4"></video><br><audio controls src="{media}.wav"></audio></div>')
        parts.append('</section>')
        for s in sheets:
            if s['name'].startswith(tag):parts.append(f'<h3>{s["name"]}</h3><a href="{s["name"]}"><img src="{s["name"]}"></a>')
    (OUT/'index.html').write_text('\n'.join(parts),encoding='utf8')
    write('execution_receipt.json',{'unique_video_audio_inputs':len(pins),'phrase_audio_rows':len(rows),'sheets':len(sheets),
                                  'new_model_inference':False,'audio_direct_perception':'unavailable',
                                  'output_hashes':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'out':str(OUT),'inputs':len(pins),'rows':len(rows),'sheets':len(sheets)}))


if __name__=='__main__':
    main()
