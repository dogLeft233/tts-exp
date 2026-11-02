"""Display every frozen train-gain x driver-gain cell; do not select a best volume."""
from pathlib import Path
import json,sys,os
from protocol import OUT,sha,write
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib-cache');sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/plotdeps')
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
def main():
 root=OUT/'analysis';qa=json.loads((root/'qa.json').read_text());assert qa['passed'];p=json.loads((OUT/'protocol.json').read_text());parent=Path(p['parent']);cells=json.loads((root/'cells.json').read_text());effects=json.loads((root/'training_gain_effects.json').read_text())
 get=lambda mid,n,tg,qg:next(c for c in cells if (c['encoder'],c['train_normalize'],c['test_normalize'],c['train_gain_db'],c['gain_db'])==(mid,n,n,tg,qg))
 fig,axes=plt.subplots(4,3,figsize=(13,14),layout='constrained');metrics=[('sync_c','Sync-C (proxy, higher)'),('distance_d0','D0 at original clock (lower)'),('mouth_bbox_MSE','Mouth-box MSE (lower)')]
 for row,(mid,n) in enumerate([(m,n) for m in ['hubert_base','wav2vec2_base'] for n in [False,True]]):
  for col,(metric,label) in enumerate(metrics):
   ax=axes[row,col];matrix=np.array([[get(mid,n,tg,qg)[metric] for qg in [-6,0,6]] for tg in [-6,0,6]])
   common=[get(mid,nn,tg,qg)[metric] for nn in [False,True] for tg in [-6,0,6] for qg in [-6,0,6]]
   im=ax.imshow(matrix,cmap='viridis',vmin=min(common),vmax=max(common),aspect='equal');fig.colorbar(im,ax=ax,shrink=.8)
   ax.set(xticks=[0,1,2],xticklabels=['-6','0','+6'],yticks=[0,1,2],yticklabels=['-6','0','+6'],xlabel='Inference driver gain (dB)',ylabel='Training gain (dB)',title=f'{mid} | norm {"on" if n else "off"}\n{label}')
   for i in range(3):
    for j in range(3):ax.text(j,i,f'{matrix[i,j]:.3f}' if col<2 else f'{matrix[i,j]:.5f}',ha='center',va='center',color='white',bbox=dict(facecolor='black',alpha=.32,edgecolor='none',pad=2))
 fig.suptitle('May same-encoder gain factorial | matched training/inference normalization\nFixed 12000 successful updates / EMA12 | original scoring PCM | same clips',fontsize=14);fig.savefig(root/'gain_matrices.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');colors={-6:'#1767b2',0:'#3c9765',6:'#c86624'}
 for ax,mid in zip(axes,['hubert_base','wav2vec2_base']):
  for n in [False,True]:
   for gain in [-6,0,6]:
    arm=mid+('_on' if n else '_off')
    folder=parent/'training'/arm if gain==0 else OUT/'training'/(arm+('_trainm6' if gain<0 else '_trainp6'))
    curve=json.loads((folder/'validation_curve.json').read_text());ys=[next(c['mean']['mouth_bbox_MSE'] for c in r['validation'] if c['normalize']==n) for r in curve]
    ax.plot([r['step'] for r in curve],ys,'-' if n else '--',color=colors[gain],label=f'{gain:+d} dB / norm {"on" if n else "off"}',lw=1.5)
  ax.set(title=mid,xlabel='Successful updates',ylabel='Validation mouth-box MSE (fixed 16 frames)',yscale='log');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Same initialization, actual frame/ray/LR ledger; zero-gain arms are exact report54 imports');fig.savefig(root/'learning_curves.png',dpi=160);plt.close(fig)
 canvas=Image.new('RGB',(6*384,4*426),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 for row,(mid,n) in enumerate([(m,n) for m in ['hubert_base','wav2vec2_base'] for n in [False,True]]):
  for col,(tg,qg) in enumerate([(-6,0),(0,0),(6,0),(0,-6),(0,6),(6,6)]):
   arm=mid+('_on' if n else '_off');base=parent/'remote/renders'/arm if tg==0 else Path(p['new_media_placements'][arm+('_trainm6' if tg<0 else '_trainp6')])
   im=Image.open(base/f'{"on" if n else "off"}_g{qg:+d}/test_00/frame125.png').convert('RGB');canvas.paste(im.resize((384,384)),(col*384,row*426+42));draw.text((col*384+5,row*426+3),f'{mid} N:{int(n)} T:{tg:+d} I:{qg:+d}',font=font,fill='black')
 canvas.save(root/'matched_frame_grid.png')
 lines=['# May训练音量×推理音量与输入归一化先导','', '固定每个encoder权重，只切native输入z-score；−6/0/+6训练gain与−6/0/+6推理driver gain完整交叉。2encoder×2train norm×3train gain×2inference norm×3driver gain=72格，144个10秒clip/矩阵。0gain四组来自报告54原件，8新组各12000成功FP32更新，不做零gain重放。','', '本次问题是训练音量和推理音量的作用是否被归一化改变；跨encoder或旧AVE60k差异仍不能归因归一化。归一化开启不是RMS0.1，而是同HF输入零均值/单位方差。单May、单seed、短预算，两clip相关，无人类感知端点；训练volume组的微小差别可能包含CUDA优化轨迹差异，不能作总体因果结论。','', '## 主矩阵：训练与推理归一化一致','', '| encoder | norm | train dB | driver dB | C | Dmin | D0 | B | mouth MSE | offsets |','|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
 for mid in ['hubert_base','wav2vec2_base']:
  for n in [False,True]:
   for tg in [-6,0,6]:
    for qg in [-6,0,6]:
     c=get(mid,n,tg,qg);lines.append(f'| {mid} | {n} | {tg:+d} | {qg:+d} | {c["sync_c"]:.3f} | {c["sync_d"]:.3f} | {c["distance_d0"]:.3f} | {c["background_median"]:.3f} | {c["mouth_bbox_MSE"]:.6f} | {c["av_offsets"]} |')
 lines+=['','## 固定推理0dB的训练音量差异','', '相对相同norm的训练0dB模型。ΔC正仅代表评价器置信度增加；ΔD0或mouth MSE正代表对应误差增加。','', '| encoder | norm | train dB−0 | ΔC | ΔD0 | Δmouth MSE |','|---|---|---:|---:|---:|---:|']
 for c in effects:
  if c['train_normalize']==c['test_normalize'] and c['test_gain_db']==0:lines.append(f'| {c["encoder"]} | {c["train_normalize"]} | {c["training_gain_db"]:+d} | {c["sync_c"]:+.3f} | {c["distance_d0"]:+.3f} | {c["mouth_bbox_MSE"]:+.6f} |')
 lines+=['','## 增益敏感性','', '| encoder | norm | fixed train dB | C range across driver gains | mouth MSE range |','|---|---|---:|---:|---:|']
 for mid in ['hubert_base','wav2vec2_base']:
  for n in [False,True]:
   for tg in [-6,0,6]:
    rows=[get(mid,n,tg,qg) for qg in [-6,0,6]];lines.append(f'| {mid} | {n} | {tg:+d} | {max(r["sync_c"] for r in rows)-min(r["sync_c"] for r in rows):.6f} | {max(r["mouth_bbox_MSE"] for r in rows)-min(r["mouth_bbox_MSE"] for r in rows):.8f} |')
 lines+=['','## 控制、复用和边界','', '训练与推理norm失配的全部72格在cells.csv；训练gain−0差异在training_gain_effects.csv，固定两个gain时的归一化效应在normalization_effects.csv。所有条件等权两clip，不按test选择训练音量、剂量或checkpoint。','', '每组共同完整随机initial、12000成功更新、原AdamW/60k LR前缀、EMA12；65536 rays/FP16/RMS0.1的Patrick实验不在此协议。原May AVE60k九格见报告50，不能与此处改变了特征/预算的数字作纯归一化因果差。','', f'独立新消费QA：{qa["new_video_frames"]}帧RGB/PCM/ROI/selected float/量化界，{qa["new_matrices"]}矩阵，{qa["new_distance_cells"]}距离，float64重算最大误差{qa["max_new_distance_error"]:.3g}；另报告54原48矩阵/12000帧保持原独立QA并全成员SHA导入。','', '大文件跨F/E两个持久卷保存，各arm逐成员新鲜SHA核验。为了空间，只有旧报告50云整包9分片/643成员实际独立Linux fresh读回通过后才迁移固定18个旧render副本，原SHA/归档member/locator留存；旧PNG/JSON/score/最终cp保持。新产物尚不声称云备份。','', '实际图检后才发布最终报告；图中固定帧仅工程观看，不是盲评。训练gain效应与纯推理gain响应分开；没有多训练seed，不能把归一化后仍有模型差异自动当gain机制。','', '证据入口：protocol.json、parent_full_member_import_review.json、feature_independent_review.json、qualification_independent_review.json、training_independent_review.json、postprocess_complete.json、media_clock_review.json、analysis/qa.json及跨卷最终manifest。']
 (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');write(root/'report_candidate_receipt.json',dict(passed=True,visual_pending=True,report_sha256=sha(root/'report.md')))
 print('plots and report created; actual view_image required')
if __name__=='__main__':main()
