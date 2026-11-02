"""Scientific plots and condition-specific report; never infer perceptual harm."""
import os
from pathlib import Path
from protocol import OUT,sha,write
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib-cache')
import sys
sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/plotdeps')
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 a=OUT/'analysis';qa=json.loads((a/'qa.json').read_text());assert qa['passed']
 p=json.loads((OUT/'protocol.json').read_text());cells=json.loads((a/'cells.json').read_text());contrasts=json.loads((a/'contrasts.json').read_text());history=json.loads((a/'historical_same_clips.json').read_text())
 fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
 for axis,spec in zip(axes,p['models']):
  for norm,color in [(True,'#1767b2'),(False,'#cf6d22')]:
   arm=spec['id']+('_on' if norm else '_off');curve=json.loads((OUT/'training'/arm/'validation_curve.json').read_text())
   for test,style in [(True,'-'),(False,'--')]:
    xs=[r['step'] for r in curve];ys=[next(v['mean']['mouth_bbox_MSE'] for v in r['validation'] if v['normalize']==test) for r in curve]
    axis.plot(xs,ys,style,color=color,label=f'train {"on" if norm else "off"} / infer {"on" if test else "off"}',lw=1.7)
  axis.set(title=spec['id'],xlabel='Successful updates',ylabel='Validation mouth-box MSE (16 fixed frames)',yscale='log');axis.grid(alpha=.2);axis.legend(fontsize=8)
 fig.suptitle('May paired normalization pilot — same encoder weights and generator initialization');fig.savefig(a/'learning_curves.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(15,7.2),layout='constrained')
 colors={(True,True):'#1767b2',(False,False):'#cf6d22',(True,False):'#68a5cc',(False,True):'#dfa878'}
 for ri,spec in enumerate(p['models']):
  for ci,(metric,label) in enumerate([('sync_c','Sync-C (proxy, higher)'),('distance_d0','D0 (original clock)'),('background_median','Background median'),('mouth_bbox_MSE','Mouth-box MSE')]):
   ax=axes[ri,ci]
   for train,test in colors:
    rows=sorted([c for c in cells if c['encoder']==spec['id'] and c['train_normalize']==train and c['test_normalize']==test],key=lambda c:c['gain_db'])
    ax.plot([c['gain_db'] for c in rows],[c[metric] for c in rows],'-o' if train==test else '--o',color=colors[(train,test)],label=f'{"on" if train else "off"} / {"on" if test else "off"}',lw=1.6,ms=4)
   ax.set(title=spec['id']+'\n'+label,xlabel='Driver gain (dB)',xticks=[-6,0,6]);ax.grid(alpha=.2)
   if ri==0 and ci==0:ax.legend(title='train / inference',fontsize=8)
 fig.suptitle('Fixed final EMA12, original scoring PCM, two prespecified clips; no perceptual endpoint');fig.savefig(a/'crossed_conditions.png',dpi=160);plt.close(fig)
 from PIL import Image,ImageDraw,ImageFont
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22);canvas=Image.new('RGB',(4*512,2*560),'white');draw=ImageDraw.Draw(canvas)
 for row,spec in enumerate(p['models']):
  for col,(train,test) in enumerate([(True,True),(True,False),(False,True),(False,False)]):
   path=OUT/'remote/renders'/(spec['id']+('_on' if train else '_off'))/f'{"on" if test else "off"}_g+0/test_00/frame125.png'
   im=Image.open(path).convert('RGB');canvas.paste(im,(col*512,row*560+48));draw.text((col*512+10,row*560+7),f'{spec["id"]}  T:{"on" if train else "off"} I:{"on" if test else "off"}',font=font,fill='black')
 canvas.save(a/'matched_frame_grid.png')
 lines=['# May 同权重编码器归一化训练与推理先导','', '四臂各12000次实际成功FP32更新，0跳步；同完整随机生成器initial、同帧/ray/LR，独立核验通过。每编码器的开/关只改输入归一化；全部内部norm和预训练权重不变。','', '这是一个已暴露英语身份、一个seed、两段相关10秒clip的短程先导。不能判定总体“有害/有益”、完整收敛或人类感知改善。未恢复Patrick。','', '## 固定0dB的四格','', '| 编码器 | train norm | inference norm | C | Dmin | D0 | B | mouth MSE |','|---|---|---|---:|---:|---:|---:|---:|']
 for c in cells:
  if c['gain_db']==0:lines.append(f'| {c["encoder"]} | {c["train_normalize"]} | {c["test_normalize"]} | {c["sync_c"]:.3f} | {c["sync_d"]:.3f} | {c["distance_d0"]:.3f} | {c["background_median"]:.3f} | {c["mouth_bbox_MSE"]:.6f} |')
 lines+=['','## 偏移与full边界敏感性','', '以下保留每个clip的最佳offset，不将最大置信度等同原时钟同步；full C只作敏感性分析，不替代guard20主端点。','', '| encoder | train norm | inference norm | driver dB | offsets (two clips) | guard20 C | full C |','|---|---|---|---:|---|---:|---:|']
 for c in cells:lines.append(f'| {c["encoder"]} | {c["train_normalize"]} | {c["test_normalize"]} | {c["gain_db"]} | {c["av_offsets"]} | {c["sync_c"]:.3f} | {c["full_sync_c"]:.3f} |')
 lines+=['','## 条件内差异','', '正ΔC只代表SyncNet置信度提高；正ΔD0或mouth MSE代表对应误差增加。on/on−off/off同时改变训练和推理处理，是匹配流程联合差异，不能叫单独训练效应。','', '| 编码器 | 0dB contrast | ΔC | ΔD0 | Δmouth MSE |','|---|---|---:|---:|---:|']
 for c in contrasts:
  if c['gain_db']==0:lines.append(f'| {c["encoder"]} | {c["contrast"]} | {c["sync_c"]:+.3f} | {c["distance_d0"]:+.3f} | {c["mouth_bbox_MSE"]:+.6f} |')
 lines+=['','## 与旧May结果比较','', '以下重新聚合旧报告50相同test_00/test_03，避免用旧四clip均值对比新两clip。旧AVE为60000更新/不同特征与生成器入口，新两编码器为12000更新；因此只作描述性历史参照，不能把差值归因输入归一化。报告53原为几何消融，本次才新增下游训练与视频；方向变化本身不是改善。','', '| 旧AVE train dB | test dB | 两同clip C | D0 |','|---:|---:|---:|---:|']
 for c in history:lines.append(f'| {c["training_gain_db"]} | {c["test_gain_db"]} | {c["sync_c"]:.3f} | {c["distance_d0"]:.3f} |')
 lines+=['','## 协议、控制和证据边界','', '沿用May4150/550/1100帧划分；归一化以实际HF零均值/单位方差公式施于完整支持段后20秒chunk，末层50Hz配对成25fps；不做RMS0.1、不删除内部LN/GN。两编码器相同768维AudioNet入口，每个norm对内架构/initial严格相同。','', '最终固定EMA12；三档驱动±6/0与两种推理norm形成24格，48矩阵，固定原自然PCM16与source-owned GT ROI。主guard20、full次；没有按test选断点、片段或最佳gain。CUDA原子梯度不承诺不同训练逐位复现，只有一个seed，不能排除训练随机性或预算依赖；重复控制只测推理重复性。','', f'独立QA：{qa["video_frames"]}个实际视频帧、RGB/原PCM/裁剪hash全核验；{qa["matrices"]}矩阵、{qa["distance_cells"]}距离，float64重建最大误差{qa["max_distance_error"]:.3g}。全state实际optimizer step=12000、EMA12与trace计数/采样/LR通过。','', '最初shell CRLF/路径部署错误发生在model构造前，0训练更新；原failure和尝试保留。修正只涉及运输/运行环境及资格重载顺序，没有训练重放或数学调参。','', '图像网格是固定帧的工程观看，非盲评。图表仍需本chat实际view_image检查，BM最终发布和最终manifest读回后才收口。','', '## 复现与文件','', 'protocol.json、qualification_independent_review.json、training_independent_review.json、analysis/qa.json、各job execution/source pins及F成员manifest为证据入口。cells.csv、contrasts.csv、historical_same_clips.csv含完整±6/0结果；learning_curves.png和crossed_conditions.png、matched_frame_grid.png供审阅。']
 (a/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 write(a/'report_receipt.json',dict(numerical_report_created=True,report_sha256=sha(a/'report.md'),visual_review_pending=True,memory_publication_pending=True))
 print('report and plots created, visual review required')
if __name__=='__main__':main()
