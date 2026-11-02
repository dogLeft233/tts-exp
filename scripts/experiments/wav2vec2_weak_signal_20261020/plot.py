"""Scientific static summary, showing attenuation separately from feature change."""
import os,sys,json,argparse
from pathlib import Path


def main(root):
    os.environ['MPLCONFIGDIR']=str(root/'matplotlib-cache')
    sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/plotdeps')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    font=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    j=json.loads((root/'analysis.json').read_text())
    stages=['conv0_linear','conv0_groupnorm','conv0_activation','conv1_activation','conv2_activation',
            'conv3_activation','conv4_activation','conv5_activation','conv6_activation','projection_norm',
            'projection_linear','positional_embedding','transformer_input_norm']+[
            f'transformer_{i:02d}' for i in range(1,13)]
    # Compact path omits only positional embedding and duplicate endpoint.
    selected=['conv0_groupnorm','conv6_activation','projection_linear']+[f'transformer_{i:02d}' for i in range(1,13)]
    labels=['首层GN','卷积7','投影']+[str(i) for i in range(1,13)]
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    ax=axes[0,0];x=np.arange(3);width=.26
    for k,s in enumerate(['train','val','test']):
        vals=[next(r['median_attenuation_pct'] for r in j['first_GN_prevariance'] if r['split']==s and r['stratum']==q)
              for q in ['low20','middle60','high20']]
        bars=ax.bar(x+(k-1)*width,vals,width=width,label=s)
        ax.bar_label(bars,fmt='%.3f%%',padding=3,fontsize=8)
    ax.set_xticks(x,['归一化前方差最低20%','中间60%','最高20%']);ax.set_ylabel('GN去除偏置后的幅度下降 (%)')
    ax.set_ylim(0,.25);ax.legend();ax.set_title('首层：低方差通道的相对抑制更强（中位数）')
    for ax,key,title in [(axes[0,1],'relative_l2','表征确实变化：末层相对L2约0.96%–1.45%')]:
        for s in ['train','val','test']:
            vals=[100*next(r[key] for r in j['layer'] if (r['split'],r['stage'],r['stratum'])==(s,st,'all')) for st in selected]
            ax.plot(range(len(selected)),vals,'o-',ms=3,label=s)
        ax.set_xticks(range(len(selected)),labels,rotation=35);ax.set_ylabel('相对 L2 变化 (%)');ax.set_xlabel('卷积前端 → Transformer 层数');ax.legend();ax.set_title(title)
    ax=axes[1,0]
    for s in ['train','val','test']:
        vals=[]
        for st in selected:
            lo=next(r for r in j['layer'] if (r['split'],r['stage'],r['stratum'])==(s,st,'low20'))
            hi=next(r for r in j['layer'] if (r['split'],r['stage'],r['stratum'])==(s,st,'high20'))
            vals.append(100*(lo['norm_ratio']/hi['norm_ratio']-1))
        ax.plot(range(len(selected)),vals,'o-',ms=3,label=s)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(len(selected)),labels,rotation=35)
    ax.set_ylabel('低能量时段相对高能量时段的幅度变化 (%)');ax.set_title('低能量时段：部分层压低，部分层反向；非逐层降噪');ax.legend()
    ax=axes[1,1]
    for s in ['train','val','test']:
        vals=[]
        for st in selected:
            lo=next(r for r in j['channel_centered'] if (r['split'],r['stage'],r['stratum'])==(s,st,'low20'))
            hi=next(r for r in j['channel_centered'] if (r['split'],r['stage'],r['stratum'])==(s,st,'high20'))
            vals.append(100*(lo['centered_norm_ratio']/hi['centered_norm_ratio']-1))
        ax.plot(range(len(selected)),vals,'o-',ms=3,label=s)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(len(selected)),labels,rotation=35)
    ax.set_ylabel('弱输出波动通道相对强通道的变化 (%)')
    ax.set_title('按各层输出波动重新定义弱通道：没有一致压低');ax.legend()
    fig.suptitle('May实际音频 · wav2vec2降低输入尺度：存在轻微通道选择性，未证明噪声被滤除',fontsize=15)
    fig.savefig(root/'weak-signal-layer-summary.png',dpi=180);plt.close(fig)
    print(str(root/'weak-signal-layer-summary.png'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
