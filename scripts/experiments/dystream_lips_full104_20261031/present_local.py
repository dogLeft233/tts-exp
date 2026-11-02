"""Final full-cohort speaker figure and locally playable104 pairs+one unpaired N."""
from pathlib import Path
import argparse,hashlib,html,json,os,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    root=out/'evidence';summary=json.loads((root/'summary.json').read_text());sp=json.loads((root/'speaker_scores.json').read_text())
    pairs=json.loads((root/'paired_scores.json').read_text());rows=json.loads((root/'scoring_inputs.json').read_text())['rows']
    project=Path(__file__).resolve().parents[3];model=project/'results/dystream';links={}
    for row in rows:
        src=root/'videos'/(row['id']+'.mp4');dest=model/('natural_raw' if row['arm']=='N' else 'tts_raw')/row['source']/(out.name+'_'+row['id']+'.mp4')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==row['video_sha256']
        else:shutil.copyfile(src,dest)
        links[row['id']]=str(dest)
    (out/'video_index.json').write_text(json.dumps(links,indent=2))
    fig,axes=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
    for ax,(metric,values) in zip(axes.flat,sp['all_pairs'].items()):
        strict={r['speaker'] for r in sp['strict_quality'][metric]};d=np.array([r['delta'] for r in values])
        ax.axhline(0,color='black',lw=.8);ax.bar(np.arange(len(d)),d,color=['#167c80' if r['speaker'] in strict else '#dd8c36' for r in values])
        q=summary['subsets']['all_pairs'][metric]['holm9_p'];qs=summary['subsets']['strict_quality'][metric]['holm9_p']
        fmt='.3f' if metric=='SyncNet' else '.4f'
        ax.set_title(f'{metric}: {len(d)} speakers\nmean={d.mean():+{fmt}}; Holm9 p strict={qs:.4f}, all={q:.4f}',fontsize=9)
        ax.set_xlabel('Speaker order',fontsize=8);ax.set_ylabel('TTS-natural (native units)',fontsize=8);ax.tick_params(labelsize=8)
    fig.suptitle('DyStream / Chinese-LiPS full104 pairs,21 speakers\nWithin-speaker utterance means, equal speaker weights. Teal: strict quality subset. Evaluator evidence.',fontsize=12)
    fig.savefig(out/'speaker_differences.png',dpi=180);fig.savefig(out/'speaker_differences.pdf');plt.close(fig)
    blocks=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>DyStream全量104对</title>',
        '<style>body{max-width:1200px;margin:2rem auto;font:16px system-ui}section{border-top:1px solid #aaa;padding:1rem}video{width:48%}input{padding:.6rem;width:70%}</style>',
        '<h1>DyStream：Chinese-LiPS全量104对＋1条单侧自然</h1><p>左自然、右TTS；按句级Sync-C分差降序，统计先说话人内取均值，104句不当104独立人。九评分器原始数据及完整说话人检验已复核。</p>',
        '<p><a href="report.md">完整报告</a> · <a href="paired_scores.csv">逐句九项CSV</a> · <a href="speaker_differences.png">说话人差值图</a></p>',
        '<input id="filter" placeholder="输入说话人ID或样本ID筛选"><script>addEventListener("DOMContentLoaded",()=>document.querySelector("#filter").oninput=e=>document.querySelectorAll("section").forEach(s=>s.hidden=!s.dataset.id.includes(e.target.value)))</script>']
    for pair in sorted(pairs['SyncNet'],key=lambda r:r['delta'],reverse=True):
        sid=pair['source'];quality='无已知饱和/触顶标记' if pair['strict_quality'] else '源饱和/TTS触顶标记，主要子集外'
        blocks.append(f'<section data-id="{sid}"><h2>{sid} · ΔC={pair["delta"]:+.3f}</h2><p>自然{pair["N"]:.3f}，TTS{pair["T"]:.3f}；{quality}；'+('PC-Talk原试跑句' if pair['pilot_seen'] else '其余配对句')+'</p>')
        for arm in ['N','T']:blocks.append(f'<video controls preload="none" src="{html.escape(os.path.relpath(links[sid+"_"+arm],out))}"></video>')
        blocks.append('</section>')
    for row in rows:
        if not row['paired_available']:
            blocks.append(f'<section data-id="{row["source"]}"><h2>{row["source"]} · TTS原有Engine411缺失</h2><p>单侧自然仅描述，不入104对推断。</p><video controls preload="none" src="{html.escape(os.path.relpath(links[row["id"]],out))}"></video></section>')
    (out/'index.html').write_text('\n'.join(blocks)+'</html>');print('PRESENTATION_COMPLETE',len(links))

if __name__=='__main__':main()
