"""Show the full fixed grid and prespecified interactions, with no test selection."""
from pathlib import Path
import sys,os,json
from protocol import OUT,sha,write
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib-cache');sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/plotdeps')
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
def main():
    root=OUT/'analysis';assert json.loads((root/'qa.json').read_text())['passed'];p=json.loads((OUT/'protocol.json').read_text());parent=Path(p['parent']);cells=json.loads((root/'cells.json').read_text());effects=json.loads((root/'effects.json').read_text())
    get=lambda mid,t,q:next(c for c in cells if (c['encoder'],c['train_input_scale'],c['inference_input_scale'])==(mid,t,q))
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for row,mid in enumerate(['hubert_base','wav2vec2_base']):
        for col,(key,label) in enumerate([('sync_c','Sync-C (proxy, higher)'),('distance_d0','D0 at original clock (lower)'),('mouth_bbox_MSE','Mouth-box MSE (lower)')]):
            ax=axes[row,col];z=np.array([[get(mid,t,q)[key] for q in [.5,1,2]] for t in [.5,1]])
            im=ax.imshow(z,cmap='viridis',aspect='equal');fig.colorbar(im,ax=ax,shrink=.7)
            ax.set(xticks=range(3),xticklabels=['0.5','1','2'],yticks=range(2),yticklabels=['0.5','1'],xlabel='Inference scale AFTER z-score',ylabel='Training scale AFTER z-score',title=mid+'\n'+label)
            for i in range(2):
                for j in range(3):ax.text(j,i,f'{z[i,j]:.3f}' if col<2 else f'{z[i,j]:.5f}',ha='center',va='center',color='white',bbox=dict(facecolor='black',alpha=.3,edgecolor='none',pad=2))
    fig.suptitle('May | fixed encoder, same initial/12000 updates/EMA12 | original scoring PCM');fig.savefig(root/'scale_matrices.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,mid in zip(axes,['hubert_base','wav2vec2_base']):
        for t,label in [(.5,'new train scale0.5'),(1,'imported train scale1')]:
            folder=OUT/'training'/(mid+('_scale050' if t==.5 else '_scale100'));curve=json.loads((folder/'validation_curve.json').read_text());vals=[next(c['mean']['mouth_bbox_MSE'] for c in row['validation'] if c['normalize']==1) for row in curve]
            ax.plot([r['step'] for r in curve],vals,label=label)
        ax.set(title=mid,xlabel='Successful updates',ylabel='Validation mouth-box MSE at scale1',yscale='log');ax.legend();ax.grid(alpha=.2)
    fig.savefig(root/'learning_curves.png',dpi=150);plt.close(fig)
    canvas=Image.new('RGB',(3*384,4*426),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for row,(mid,t) in enumerate([(m,t) for m in ['hubert_base','wav2vec2_base'] for t in [.5,1]]):
        for col,q in enumerate([.5,1,2]):
            f=parent/'remote/renders'/(mid+'_on')/'on_g+0/test_00/frame125.png' if t==q==1 else OUT/'remote/renders'/(mid+('_scale050' if t==.5 else '_scale100'))/f's{q:.2f}/test_00/frame125.png'
            image=Image.open(f).convert('RGB');canvas.paste(image.resize((384,384)),(col*384,row*426+42));draw.text((col*384+4,row*426+3),f'{mid} T:{t:g} I:{q:g}',fill='black',font=font)
    canvas.save(root/'matched_frame_grid.png')
    lines=['# May归一化后低尺度训练与高尺度驱动对照','','同native z-score后再乘0.5/1/2，约对应RMS0.5/1/2；相对标准化内部尺度，不是原录音gain，也不是RMS0.1策略。两新低尺度模型各12000成功FP32更新/0skip，两标准模型和两个标准cell原件复用。原PCM16评分。单May/seed/短预算，不选最佳尺度或宣称感知改善。','','| encoder | train scale | infer scale | C | D0 | mouth MSE | offsets |','|---|---:|---:|---:|---:|---:|---|']
    for c in sorted(cells,key=lambda c:(c['encoder'],c['train_input_scale'],c['inference_input_scale'])):lines.append(f'| {c["encoder"]} | {c["train_input_scale"]:g} | {c["inference_input_scale"]:g} | {c["sync_c"]:.3f} | {c["distance_d0"]:.3f} | {c["mouth_bbox_MSE"]:.6f} | {c["av_offsets"]} |')
    lines+=['','## 预定配对差异','','正ΔC仅表示评分置信度变大；正ΔD0/ΔMSE为对应误差增加。','','| encoder | contrast | ΔC | ΔD0 | Δmouth MSE |','|---|---|---:|---:|---:|']
    for c in effects:lines.append(f'| {c["encoder"]} | {c["contrast"]} | {c["sync_c"]:+.3f} | {c["distance_d0"]:+.3f} | {c["mouth_bbox_MSE"]:+.6f} |')
    lines+=['','报告55的gain是在native标准化之前施加，norm开启后被基本抵消；本次在标准化之后直接缩放，不能把两种干预混作同一个dose。旧AVE60k与当前表征/预算不同，不作纯尺度因果比较。训练组只有一个随机seed，即使实际采样相同，也不能排除优化轨迹差异；SyncNet和像素误差不等于人类感知。','','所有新5000帧/20矩阵的RGB/原PCM/ROI/crop/selectedfloat/时钟和float64距离重建见qa.json；另4旧矩阵按完整父SHA与原QA导入。真实图检和BM发布仍需本chat完成。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');write(root/'candidate_receipt.json',dict(passed=True,numerical_delivery_ready=True,visual_and_memory_pending=True,report_sha256=sha(root/'report.md')))
if __name__=='__main__':main()
