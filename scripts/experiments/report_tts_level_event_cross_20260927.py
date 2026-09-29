"""Format frozen event factorial outputs; no metric recomputation."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/tts_level_event_cross_20260927'

def read(p):
    return json.loads(Path(p).read_text())

def main():
    s=read(OUT/'summary.json');b=read(OUT/'bridge_summary.json');gate=read(OUT/'bridge_validation.json')
    independent=read(OUT/'independent_validation.json')
    assert independent['status']=='PASS' and not (OUT/'final.json').exists()
    def fmt(z):
        v=z['speaker_mean'];lo,hi=z['speaker_ci99']
        return f'{v:+.6f} [{lo:+.6f}, {hi:+.6f}]'
    lines=['# 电平控制后共同音素事件的状态与来源四格','',
       '固定历史32条/13 speaker/1411 queries、20647 donor assignments、k3、image3；CPU缓存探索。全部结果为speaker等权均值与逐比较99% CI，20k bootstrap seed20260926，无FWER声明。',
       '', '## 结论', '',
       'FIXED后raw margin的T−N差仍为正，且比同支持RAW缩小。其残余在算术上主要表现为共同负池距离的T−N差；正确event正距离差的CI跨0。与此同时，固定自然另一侧时，单独换成TTS音频或TTS生成视觉均降低margin/rank；来源交互仍为正，并主要表现在正距离交互。两者描述的是不同问题：前者是同臂T−N残余，后者是N/T来源的配对依赖，不能合成中介比例。',
       '',
       'unit residual margin也为正，但FIXED−RAW变化的CI跨0，且unit残余的正、负距离差各自CI均跨0；因此不能把raw的距离定位原样推广到所有几何。raw residual rank的CI跨0，unit residual rank为正；LEVEL raw/unit margin也保留，但不等同于FIXED的新确认。完整阴性和反向simple effects见下表及summary.json。',
       '', '## 支持与桥接', '',
       '旧三图3/6/9全部query和历史统计通过重放；image3单列。旧三图+0.436 margin与+2.250 interaction不直接充当新单图基线。新joint guard20会删23query，本run依批准原样保留legacy-event支持；它不等于71条官方Sync-C支持。',
       '',f'桥接最大误差：旧query {gate["old_query_max"]:.3g}，旧全部统计 {gate["old_statistic_max"]:.3g}，新RAW image3 {gate["new_raw_image3_query_max"]:.3g}。',
       '', '| RAW桥接 | raw margin对角TT−NN | raw margin来源交互 |',
       '|:--|--:|--:|']
    for title,k in [('旧三图','historical_three_images'),('image3','image3_RAW')]:
        lines.append(f'| {title} | {fmt(b[k]["raw_margin_diagonal"])} | {fmt(b[k]["raw_margin_interaction"])} |')
    lines+=['','## 同来源配对残余与操作路径','','RAW/FIXED 状态路径 G=FR−RR，E=RF−RR，I=FF−FR−RF+RR，total=FF−RR。表中为TT−NN响应差；RR/FR/RF/FF状态中第一位表示视频生成状态，第二位为评价音频状态。LEVEL仅两侧同状态。','',
            '| geometry / metric | RAW gap | FIXED gap | LEVEL gap | G gap响应 | E gap响应 | I gap响应 | total gap响应 |',
            '|:--|--:|--:|--:|--:|--:|--:|--:|']
    for g in ['raw','unit']:
        for m in ['positive','negative','margin','rank']:
            keys=[f'{g}/{m}/source/{h}/diagonal' for h in ['RR','FF','LL']]+[f'{g}/{m}/state_gap/{n}' for n in ['G','E','I','total']]
            lines.append('| '+g+' / '+m+' | '+' | '.join(fmt(s[k]) for k in keys)+' |')
    lines+=['','## N/T来源四格','','此处替换的是N/T来源，与上表电平状态操作是不同因子。qXY表示V_X与A_Y，正/负音频均随其cell音频状态。两个固定自然侧simple effects和来源交互如下；反向simple effects及全部cells在summary.json。',
            '', '| geometry / metric / 状态 | V_T−V_N 固定 A_N | A_T−A_N 固定 V_N | 来源交互 |',
            '|:--|--:|--:|--:|']
    for g in ['raw','unit']:
        for m in ['positive','negative','margin','rank']:
            for h in ['RR','FF','LL','FR','RF']:
                keys=[f'{g}/{m}/source/{h}/{n}' for n in ['visual_at_N_audio','audio_at_N_visual','interaction']]
                lines.append('| '+g+' / '+m+' / '+h+' | '+' | '.join(fmt(s[k]) for k in keys)+' |')
    lines+=['','## 每臂完整电平路径','','| geometry / metric / 来源cell | G | E | I | total |','|:--|--:|--:|--:|--:|']
    for g in ['raw','unit']:
        for m in ['positive','negative','margin','rank']:
            for a in ['NN','NT','TN','TT']:
                lines.append('| '+g+' / '+m+' / '+a+' | '+' | '.join(fmt(s[f'{g}/{m}/state/{a}/{n}']) for n in ['G','E','I','total'])+' |')
    lines+=['','## 完整产物与科学边界','',
       '- protocol.json、seal.json、inputs.json与indices.json在评分前绑定；review.json为独立先验审阅。bridge_queries.npz保存旧三图全query重放，query_metrics.npz保存全部20cell×1411×2geometry×4metric；不落盘完整donor距离。',
       '- summary.json含全部600个逐比较端点、95/99CI、每speaker值及clip加权描述；主判读按协议用99CI。clip_effects.npz保留每clip全部对比。所有跨零结果原样保留。',
       '- positive与negative在同几何中对margin算术闭合，不将raw/unit跨尺度比较成份额，不将99CI含0称等效。',
       '- RMS操作不是听感响度匹配；LEVEL按对中点、FIXED由cal独立目标，不能混称同一操作。负donor全部采用对应音频state。',
       '- annotation坐标不是真实物理同步；无波形/视频warp、forward、GPU或自然音轨替换。MFA残差、上下文、发音及谱形时序仍混合；历史32条、单图，不是新speaker确认或真实嘴型证据。',
       '- 本结果不直接分解71条官方Sync-C残余，不宣称生理中介、因果百分比或TTS专属同实例适配。',
       '- 独立数值复核PASS：225760个query数值误差0、clip聚合误差0；600端点/15000个统计值最大误差1.77636e−15；effect误差≤2.66454e−15。747输入hash与审阅→桥接→评分顺序通过。外部checker及receipt绑定见independent_validation.json。',
       '- 实际生产执行23.76秒，最大RSS78048KiB（76.22MiB）、2CPU affinity；GPU/forward为0。新增产物与磁盘门见final.json。', '']
    p=OUT/'report.md';p.write_text('\n'.join(lines))
    print(hashlib.sha256(p.read_bytes()).hexdigest())

if __name__=='__main__':main()
