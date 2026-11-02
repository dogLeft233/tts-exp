"""Replay official PEAVS metric at legacy Chinese batch20 using retained features."""
from pathlib import Path
import csv, json, os, shutil, subprocess, sys, time
from prepare import B, RUN, sha, write
from analyze import make_pairs, statistics

def main():
    primary=json.loads((RUN/'peavs/scores.json').read_text());work=Path(primary['work_dir'])
    target=RUN/'peavs/legacy20-metric';assert not target.exists()
    source=B/'assets/peavs/peavs-source/PEAVS'
    shutil.copytree(source,target,copy_function=lambda src,dst:os.symlink(src,dst),ignore=shutil.ignore_patterns('output_feats','results','inputs','__pycache__'))
    features=target/'av_features_extraction/output_feats';features.symlink_to(work/'av_features_extraction/output_feats')
    code=(source/'metric.py').read_text().replace("    os.system('rm -rf ./av_features_extraction/output_feats/')",'    pass # preserve canonical retained features')
    prefix='import torch; torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.60)\n'
    subprocess.run([sys.executable,'-c',prefix+code,'--batch_size','20'],cwd=target,check=True)
    scores=list(csv.DictReader((target/'results/results.csv').open()))
    assert len(scores)==80
    values={Path(r['filename']).stem:float(r['score']) for r in scores}
    inputs=json.loads((RUN/'scoring_inputs.json').read_text())['rows'];pairs,missing=make_pairs(inputs,values);assert not missing and len(pairs)==20
    subsets={}
    for name in ['all20','same_reference12','extended_reference8']:
        group=[r for r in pairs if name=='all20' or r['strict_reference']==(name=='same_reference12')]
        subsets[name]=statistics([r['N'] for r in group],[r['T'] for r in group])
    result=dict(status='PASS',scope='Supplementary batch sensitivity only; primary official64+16 scores unchanged',
        legacy_batch_size=20,batches='4x20 in unchanged global lexical filename order',new_feature_forwards=0,
        primary_json_sha256=sha(RUN/'peavs/scores.json'),feature_directory=str(work/'av_features_extraction/output_feats'),
        metric_source_sha256=sha(source/'metric.py'),subsets=subsets,paired_scores=pairs,raw_csv_sha256=sha(target/'results/results.csv'))
    write(RUN/'peavs_batch20_sensitivity.json',result)
    summary=json.loads((RUN/'summary.json').read_text())
    lines=['','## SyncReward 原生完整4.75秒子集','',
        '主表沿用官方短片内部补齐。下列只保留四条原始音频均覆盖4.75秒的完整来源，队列与主表不同，属于补充敏感性分析。','']
    sr=summary['syncreward_full_native_span_sensitivity']['statistics']
    lines.append(f"配对人数 {sr['n_sources']}；自然均值 {sr['N_mean']:.6f}，TTS均值 {sr['T_mean']:.6f}，差值 {sr['delta']:+.6f}，95% CI [{sr['ci95'][0]:+.6f}, {sr['ci95'][1]:+.6f}]。")
    lines += ['','## PEAVS 批次敏感性','',
        '主分析固定官方batch64+16；旧中文脚本曾按20条分批。使用同一批已保留的特征，仅重放官方metric为batch20的4批，没有新增特征前向，原主分数不改。','',
        '|分层|人数|主64+16差值|补充20×4差值|补充95% CI|','|---|---:|---:|---:|---|']
    for name,s in subsets.items():
        primary_delta=summary['subsets'][name]['PEAVS']['delta']
        lines.append(f"|{name}|{s['n_sources']}|{primary_delta:+.6f}|{s['delta']:+.6f}|[{s['ci95'][0]:+.6f}, {s['ci95'][1]:+.6f}]|")
    report=RUN/'report.md';text=report.read_text();assert '## PEAVS 批次敏感性' not in text
    report.write_text(text+'\n'.join(lines)+'\n')
    print(json.dumps(subsets,indent=2))

if __name__=='__main__':main()
