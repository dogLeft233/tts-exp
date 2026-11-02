"""Create scientific figure and playable paired media from verified local evidence."""
from pathlib import Path
import argparse
import html
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    root=out/'evidence';summary=json.loads((root/'summary.json').read_text());pairs=json.loads((root/'paired_scores.json').read_text())
    rows=json.loads((root/'scoring_inputs.json').read_text())['rows'];links={}
    project=Path(__file__).resolve().parents[3]
    model_root=project/'results/pctalk'
    for r in rows:
        dest=model_root/('natural_raw' if r['arm']=='N' else 'tts_raw')/r['source']/(out.name+'_'+r['id']+'.mp4')
        dest.parent.mkdir(parents=True,exist_ok=True);src=root/'videos'/(r['id']+'.mp4')
        if dest.exists():
            assert hashlib.sha256(dest.read_bytes()).digest()==hashlib.sha256(src.read_bytes()).digest()
        else:shutil.copyfile(src,dest)
        links[r['id']]=str(dest)
    (out/'video_index.json').write_text(json.dumps(links,indent=2))
    fig,axs=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
    for ax,(metric,pp) in zip(axs.flat,pairs.items()):
        a=np.array([v['delta'] for v in pp]);strict=np.array([v['strict_quality'] for v in pp])
        ax.axhline(0,color='black',lw=.8)
        ax.bar(np.arange(len(a)),a,color=np.where(strict,'#167c80','#dd8c36'),width=.8)
        r=summary['subsets']['all21'][metric];q=summary['subsets']['strict_quality17'][metric]['holm9_p']
        delta_label=format(a.mean(),'+.3f' if metric=='SyncNet' else '+.4f')
        ax.set_title(f'{metric}  n={len(a)}\nmean={delta_label}; Holm9 p: strict={q:.4f}, all={r["holm9_p"]:.4f}',fontsize=9)
        ax.set_xlabel('Speaker order (fixed before scores)',fontsize=8);ax.set_ylabel('TTS - natural (native units)',fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle('PC-Talk / Chinese-LiPS pilot: 21 speakers, one paired utterance each\nTeal: strict quality; orange: original saturation/peak flag. Evaluator evidence, not perceptual truth.',fontsize=12)
    fig.savefig(out/'paired_differences.png',dpi=180);fig.savefig(out/'paired_differences.pdf');plt.close(fig)
    blocks=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>PC-Talk Chinese-LiPS21配对</title>',
        '<style>body{max-width:1200px;margin:2rem auto;font:16px system-ui}video{width:48%;max-width:560px}section{padding:1rem;border-top:1px solid #aaa}p{line-height:1.6}</style>',
        '<h1>PC-Talk：Chinese-LiPS 21人配对试跑</h1><p>左：自然音频；右：已有克隆TTS。按TTS−自然Sync-C差值排序，每人仅一对；17人严格质量子集，4人有原始饱和或TTS触顶标记。两臂使用同图、RMS−24，播放器为完整PCM音轨。分数不等同感知同步真值。</p>',
        '<p><a href="report.md">九评分器报告</a> · <a href="paired_scores.csv">逐来源分数CSV</a> · <a href="paired_differences.png">九指标差值图</a></p>']
    import os
    for r in sorted(pairs['SyncNet'],key=lambda x:x['delta'],reverse=True):
        source=r['source'];quality='严格质量' if r['strict_quality'] else '有源饱和/TTS触顶标记'
        blocks.append(f'<section><h2>{html.escape(source)} · ΔC={r["delta"]:+.3f}</h2><p>自然 {r["N"]:.3f}；TTS {r["T"]:.3f}；{quality}</p>')
        for arm in ['N','T']:
            rel=os.path.relpath(links[source+'_'+arm],out)
            blocks.append(f'<video controls preload="none" src="{html.escape(rel)}"></video>')
        blocks.append('</section>')
    (out/'index.html').write_text('\n'.join(blocks)+'</html>')
    print('PRESENTATION_COMPLETE',len(links))

if __name__=='__main__':main()
