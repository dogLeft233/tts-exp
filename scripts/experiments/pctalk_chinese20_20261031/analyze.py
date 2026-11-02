"""Independently reduce saved raw curves and compare source-level means."""
from pathlib import Path
import csv
import json
import pickle
import re
import subprocess
import numpy as np
from prepare import B, RUN, SOURCE, sha, write

def summarize(pairs):
    n=np.array([p['N'] for p in pairs]); t=np.array([p['T'] for p in pairs]); d=t-n
    rng=np.random.default_rng(20261030)
    boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
    return dict(sources=len(d),N=float(n.mean()),T=float(t.mean()),delta=float(d.mean()),
                ci95=np.quantile(boot,[.025,.975]).tolist(),T_higher=int((d>0).sum()))

def main():
    assert json.loads((RUN/'stages/syncnet_complete.json').read_text())['status']=='COMPLETE'
    manifest=json.loads((RUN/'scoring_inputs.json').read_text())
    scores={s['id']:s for s in json.loads((RUN/'syncnet/scores.json').read_text())}
    audit=[]
    for r in manifest['unique_rows']:
        s=scores[r['id']]
        z=np.load(RUN/'syncnet'/(r['id']+'_distances.npy')).astype(np.float64)
        curve=z.mean(0); k=int(curve.argmin()); c=float(np.median(curve)-curve[k])
        log=(RUN/'logs'/('syncnet_'+r['id']+'.log')).read_text()
        stdoutC=[float(x) for x in re.findall(r'Confidence\s*:\s*([-+.\deE]+)',log)]
        stdoutD=[float(x) for x in re.findall(r'Min dist\s*:\s*([-+.\deE]+)',log)]
        assert len(stdoutC)==len(stdoutD)==1,log[-3000:]
        assert abs(c-s['C'])<2e-5 and abs(curve[k]-s['D'])<2e-5
        assert abs(stdoutC[0]-s['C'])<.001 and abs(stdoutD[0]-s['D'])<.001
        assert s['offset']==15-k
        tracks=pickle.loads((RUN/'syncnet/work/pywork'/r['id']/'tracks.pckl').read_bytes())
        assert len(tracks)==1
        frames=np.asarray(tracks[0]['track']['frame'])
        box=np.asarray(tracks[0]['track']['bbox']).mean(0).tolist()
        internal=RUN/'syncnet/work/pyavi'/r['id']/'video.avi'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(internal)]))
        codec=next(x['codec_name'] for x in probe['streams'] if x['codec_type']=='audio')
        audit.append(dict(id=r['id'],float64_C=c,saved_C=s['C'],official_stdout_C=stdoutC[0],
                          maximum_error=abs(c-s['C']),offset=15-k,
                          track_frames=len(frames),track_start=int(frames[0]),track_end=int(frames[-1]),
                          original_generated_frames=r['actual_video_frames'],
                          mean_bbox=box,official_internal_audio_codec=codec))
    pins=json.loads((RUN/'source_pins.json').read_text())
    assert all(sha(SOURCE/p)==v for p,v in pins['source'].items())
    assert all(sha(SOURCE/'pctalk/checkpoints'/p)==v for p,v in pins['weights'].items())
    scorerpins=json.loads((RUN/'scorer_pins.json').read_text())
    assert all(sha(p)==v for p,v in scorerpins['files'].items())
    rows=[]
    for r in manifest['rows']:
        canonical=manifest['aliases'].get(r['id'],r['id'])
        rows.append(dict(**r,**{k:v for k,v in scores[canonical].items() if k!='id'},canonical_id=canonical))
    pairs=[]
    for source in sorted({r['source'] for r in rows}):
        rs=[r for r in rows if r['source']==source]
        ns=[r for r in rs if r['arm']=='N'];ts=[r for r in rs if r['arm']!='N']
        assert len(ns)==1 and len(ts)==3
        n=ns[0]['C'];t=float(np.mean([r['C'] for r in ts]))
        pairs.append(dict(source=source,speaker=ns[0]['speaker'],strict_reference=ns[0]['strict_reference'],
                          N=n,T=t,delta=t-n))
    groups={'same_reference12':[p for p in pairs if p['strict_reference']],
            'extended_reference8':[p for p in pairs if not p['strict_reference']],'all20':pairs}
    summary={name:summarize(ps) for name,ps in groups.items()}
    for name,s in summary.items():
        s['gap_vs_official_hdtf_I2V']={a:s[a]-9.37 for a in ['N','T']}
        s['gap_vs_official_mead_I2V']={a:s[a]-8.19 for a in ['N','T']}
    write(RUN/'summary.json',summary)
    write(RUN/'per_video_scores.json',rows)
    for name,records in [('paired_scores.csv',pairs),('per_video_scores.csv',[
        {k:r[k] for k in ['id','source','speaker','arm','strict_reference','C','D','B','offset','windows','canonical_id','audio_sha256','video_sha256']} for r in rows])]:
        with (RUN/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(RUN/'score_audit.json',dict(status='PASS',unique_scores=len(audit),logical_rows=len(rows),
        independent_fp64_max_error=max(x['maximum_error'] for x in audit),official_stdout_matched=True,
        source_and_weights_unchanged=True,scorer_source_weights_unchanged=True,items=audit))
    text='# PC-Talk 中文20：Sync-C 与官方I2V对照\n\n'
    text+='20来源、20N+60T逻辑媒体，三TTS先来源内平均；41唯一推理与评分。固定3.png、RMS−24、16k/25fps、person192/lip1/无EMC/seed42。\n\n'
    text+='|支持集|来源数|自然C|TTSC|T−N|95%来源bootstrap CI|T较高|\n|---|---:|---:|---:|---:|---|---:|\n'
    for name,s in summary.items():
        text+=f"|{name}|{s['sources']}|{s['N']:.3f}|{s['T']:.3f}|{s['delta']:+.3f}|[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]|{s['T_higher']}/{s['sources']}|\n"
    text+='\n本轮只完成SyncNet；未完成九评分器家族，没有Holm9显著性结论。区间为描述性边际区间。重复TTS draw不是独立合成重复。\n\n'
    s=summary['all20']
    text+=f"官方论文Table1 I2V HDTF LSE-C9.370、MEAD-Neutral8.190。本轮自然相差{s['N']-9.37:+.3f}/{s['N']-8.19:+.3f}，TTS相差{s['T']-9.37:+.3f}/{s['T']-8.19:+.3f}。官方基准是自然音频，TTS差距仅背景，不是同测试集复现误差。\n\n"
    text+='官方来源：https://arxiv.org/html/2503.14295v3#S4.SS1 。论文为HDTF/MEAD、10秒跨身份、多测试人物；本轮中文AISHELL短句、同固定脸、统一电平，数据/语言/脸/长度不同。公开代码无评分脚本，论文具体style/lip参数、crop与SyncNet权重SHA未齐全；CLI默认lip1与README示例lip.6加happy控制也有差异，均未经本轮按分调参。不能以绝对差直接认定评分错误或模型失效。\n\n'
    text+='评分核验：官方joonson/syncnet_python，FP32，syncnet_v2.model、vshift15/crop_scale.4；项目min_track25（上游100）。完整输入PCM重新无损封装，PTS/PCM逐值校验，视频原生尾部不足保留，不补帧/移音。原始31shift距离矩阵独立FP64归约与官方stdout一致，源/权重前后SHA不变。该核验排除这几类实现/绑定错误，不能验证作者完整协议或人类感知同步真值。\n'
    codecs=sorted({x['official_internal_audio_codec'] for x in audit})
    text+=f'\n内部处理补充：官方run_pipeline.py默认FFmpeg AVI转码后的音轨codec实测为{codecs}；本轮按官方流程保留，并非从入口到SyncNet模型始终无损。实际41唯一人脸轨迹的长度范围{min(x["track_frames"] for x in audit)}–{max(x["track_frames"] for x in audit)}帧；min_track100会改变短片纳入资格，不能把本轮min_track25结果称完整论文默认协议。\n'
    diagnostic=json.loads((RUN/'official_internal_codec_diagnostic.json').read_text())
    text+='\n预先选定smoke首对的只读音轨核查：自然和TTS内部音频相对原PCM的最大相关均位于+1105样点（+69.0625ms），不是逐值相同波形；首个自然片原生145帧而官方处理后轨迹147帧。该内部转码行为属于本轮保留的官方入口实现；SyncNet本身搜索±15帧。本轮没有进行PCM强制内部转码对照，不能量化其对C差值的贡献，也不能说69ms解释了论文分差。\n'
    assert all(x['internal_relative_audio_lag_samples']==1105 for x in diagnostic['items'])
    (RUN/'report.md').write_text(text)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
