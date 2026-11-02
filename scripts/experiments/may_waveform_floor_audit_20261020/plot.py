"""Static scientific localization plots; no inference or training."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path


def main(a):
    root=a.output
    os.environ['MPLCONFIGDIR']=str(root/'matplotlib-cache')
    sys.path.insert(0,str(a.plotdeps))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    font=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    j=json.loads((root/'results.json').read_text());splits=['train','val','test']
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    ax=axes[0,0];x=np.arange(3)
    for offset,db,label,color in [(-.18,0,'原音量','#4f79a7'),(.18,-6,'降6 dB','#e79540')]:
        vals=[100*next(r for r in j['spectral'] if r['split']==s)[
            'original_lower_clip' if db==0 else 'low_lower_clip']/next(r for r in j['spectral'] if r['split']==s)['cells'] for s in splits]
        bars=ax.bar(x+offset,vals,width=.36,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f%%',padding=3,fontsize=9)
    ax.set_xticks(x,splits);ax.set_ylabel('最终 mel 触底单元占比 (%)');ax.legend();ax.set_ylim(0,max(vals)*1.3)
    ax.set_title('完整 split（含上下文）：AVE 输出固定为 −4')
    ax=axes[0,1]
    freq=list(csv.DictReader((root/'spectral_by_frequency.csv').open(encoding='utf-8-sig')))
    for s in splits:
        rows=[r for r in freq if r['split']==s]
        ax.plot([float(r['center_hz']) for r in rows],
                [100*int(r['newly_lower_clipped'])/int(r['frames']) for r in rows],label=s)
    ax.set_xlabel('mel 滤波器中心频率 (Hz)');ax.set_ylabel('该频带新增触底帧 (%)');ax.legend()
    ax.set_title('哪些频带新增丢失强弱差异')
    ax=axes[1,0]
    times=list(csv.DictReader((root/'spectral_by_time.csv').open(encoding='utf-8-sig')))
    rows=[r for r in times if r['split']=='train']
    ax.plot([float(r['source_seconds']) for r in rows],[int(r['newly_lower_clipped']) for r in rows],lw=.7)
    ax.set_xlabel('May 原录音时间 (秒)');ax.set_ylabel('新增触底频带数 / 80');ax.set_ylim(0,80)
    ax.set_title('训练音频全时轴：新增触底的位置')
    ax=axes[1,1];ax.axis('off')
    restored=max(r['restored_max_abs_error'] for r in j['waveform'] if r['path']=='historical_DOUBLE' and r['gain_db']==-6)
    text=('实际历史浮点音频：\n  3 个 split，降 6 dB 后新增零点均为 0\n'
          f'  按原倍数恢复，最大波形误差 {restored:.2e}\n\n'
          '最近归一化后 ×0.5 的输入：\n  历史输入 SHA 精确匹配；再 ×2 逐位恢复\n\n'
          '频谱触底 ≠ 波形静音 ≠ 已证明降噪\n本轮仅 CPU 前端核查，0 训练 / 0 模型前向')
    ax.text(.02,.93,text,va='top',linespacing=1.7,fontsize=11)
    fig.suptitle('May 实际音频核查：波形未新增零点，AVE 弱频谱有新增截断',fontsize=16)
    fig.savefig(root/'audit-summary.png',dpi=180);plt.close(fig)
    example=next(r for r in j['examples'] if r['split']=='train');b,e=example['start_mel_frame'],example['end_mel_frame']
    amp=np.load(root/'arrays/train_gain_0_mel_amplitude.npy')[:,b:e]
    low=np.load(root/'arrays/train_gain_m6_mel_amplitude.npy')[:,b:e]
    mask=np.load(root/'arrays/train_newly_lower_clipped.npy')[:,b:e]
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
    extent=[example['source_start_seconds'],example['source_end_seconds'],0,80]
    for ax,m,title in zip(axes[:2],[amp,low],['原音量：截断前 mel dB（包含固定 −20）','降 6 dB：截断前 mel dB（包含固定 −20）']):
        im=ax.imshow(20*np.log10(np.maximum(m,1e-12))-20,origin='lower',aspect='auto',extent=extent,vmin=-120,vmax=-40,cmap='magma')
        ax.set_title(title);ax.set_ylabel('mel 频带');fig.colorbar(im,ax=ax,label='dB（非声压级）')
    axes[2].imshow(mask,origin='lower',aspect='auto',extent=extent,vmin=0,vmax=1,cmap='Blues')
    axes[2].set_title('蓝色：原来高于最终下限，降低音量后被固定为 −4 的单元')
    axes[2].set_ylabel('mel 频带');axes[2].set_xlabel('May 原录音时间 (秒)')
    fig.suptitle('训练段 4 秒定位示例：按新增触底单元最多选取，不代表全体或噪声标签',fontsize=13)
    fig.savefig(root/'train-floor-localization.png',dpi=180);plt.close(fig)
    print(str(root/'audit-summary.png'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--plotdeps',type=Path,required=True)
    main(p.parse_args())
