"""Static summary of frozen-encoder noise-induced direction changes."""
import argparse,sys,os,json
from pathlib import Path


def main(out):
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib-cache')
    sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/plotdeps')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    font=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    summary=json.loads((out/'summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    comparisons=[('whole_scale_half','整体减半，信噪比不变\nB/A：0.5(s+n) 对 s+n'),
                 ('only_noise_half','只降低噪声\nC/A：s+0.5n 对 s+n'),
                 ('restore_noise_at_half_content','低内容尺度下补回噪声\nD/B：0.5s+n 对 0.5(s+n)')]
    for ax,(key,title) in zip(axes,comparisons):
        for offset,kind,label,color in [(-.18,'white','白噪声','#4f79a7'),(.18,'pink','粉红噪声','#e79540')]:
            rows=[next(r for r in summary if (r['split'],r['kind'],r['contrast'])==(split,kind,key)) for split in ['train','val','test']]
            values=np.array([100*(r['median_unit_response_ratio']-1) for r in rows])
            lower=np.array([100*(r['median_unit_response_ratio']-r['min_ratio']) for r in rows])
            upper=np.array([100*(r['max_ratio']-r['median_unit_response_ratio']) for r in rows])
            ax.bar(np.arange(3)+offset,values,width=.34,color=color,label=label,
                   yerr=np.stack([lower,upper]),capsize=3)
            for k,v in enumerate(values):
                endpoint=v+upper[k] if v>=0 else v-lower[k]
                ax.annotate(f'{v:+.3f}%',(k+offset,endpoint),xytext=(0,8 if v>=0 else -12),
                    textcoords='offset points',ha='center',fontsize=8)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),['train','val','test']);ax.set_title(title,fontsize=11)
        ax.margins(y=.4);ax.legend(loc='best')
    axes[0].set_ylabel('噪声引起的末层方向变化：相对对照的变化 (%)')
    fig.suptitle('May已知噪声小试 · 固定编码器 / 原始添加噪声SNR 30 dB / 0训练',fontsize=15)
    fig.supxlabel('柱为3个噪声种子的中位数，误差线为最小–最大；各面板纵轴不同；不代表Sync-C或降噪质量',fontsize=10)
    fig.savefig(out/'known-noise-summary.png',dpi=180);plt.close(fig)
    # Across the selected layer path, contrast same-SNR lower whole scale.
    contrasts=json.loads((out/'contrasts.json').read_text())
    stages=['first_gn','cnn7','projection','transformer4','transformer8','transformer9','last']
    labels=['首层GN','卷积7','投影','T4','T8','T9','T12']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,kind,name in zip(axes,['white','pink'],['白噪声','粉红噪声']):
        for split in ['train','val','test']:
            values=[]
            for stage in stages:
                rows=[r['unit_response_ratio'] for r in contrasts if (r['split'],r['kind'],r['stage'],r['stratum'],r['contrast'])==
                      (split,kind,stage,'all','whole_scale_half')]
                values.append(100*(np.median(rows)-1))
            ax.plot(range(7),values,'o-',label=split)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(7),labels);ax.legend();ax.set_title(name)
        ax.set_ylabel('整体减半后噪声方向响应相对变化 (%)')
    fig.suptitle('信噪比不变：降低整体尺度是否减少已知噪声的编码影响？',fontsize=14)
    fig.savefig(out/'known-noise-layer-response.png',dpi=180);plt.close(fig)
    print(str(out/'known-noise-summary.png'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
