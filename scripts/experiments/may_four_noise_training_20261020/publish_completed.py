"""Publish only after complete numerical QA; reports all paired signs."""
from pathlib import Path
import json,re
from loguru import logger
logger.remove()
from basic_memory.mcp.tools import read_note,write_note,edit_note
from basic_memory.cli.commands.command_utils import run_with_cleanup
OUT=Path('F:/Datasets/tts-exp/may_four_noise_training_20261020_v1');REPO=Path('E:/Documents/tts-exp')
NAMES={'none':'无添加噪声','single_speech':'背景讲话','babble4':'四人babble','modulated_ssn':'起伏语音谱形','clicks':'合成点击'}
async def main():
 for ident in ['tts-exp/instructions/startup-router','tts-exp/research/研究主线与证据地图','tts-exp/instructions/实验指令','tts-exp/schemas/experiment']:
  await read_note(identifier=ident,project='tts-exp',output_format='json')
 assert not (OUT/'publication_receipt.json').exists()
 get=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
 p=get('protocol.json');qa=get('analysis/qa.json');assert qa['passed'] and get('training_independent_review.json')['passed'] and get('backup_terminal_v2.json')['passed']
 cells=get('analysis/cells.json');effects=get('analysis/effects.json');assert len(cells)==36
 full=[];interactions=[];combinations=[]
 for c in cells:full.append(f"| {NAMES[c['kind']]} | {c['snr_db'] if c['snr_db'] is not None else '—'} | {c['train_scale']} | {c['inference_scale']} | {c['sync_c']:.3f} | {c['distance_d0']:.3f} | {c['mouth_bbox_MSE']:.7f} | {c['av_offsets']} |")
 for kind in ['single_speech','babble4','modulated_ssn','clicks']:
  for q in [30,90]:
   find=lambda contrast:next(r for r in effects if r['kind']==kind and r['snr_db']==q and r['inference_scale']==1. and r['contrast']==contrast)
   harm=find('noise_harm_at_standard_train');low=find('low_train_effect_with_noise');interaction=find('noise_by_training_scale_interaction')
   interactions.append(f"| {NAMES[kind]} | {q} | {harm['sync_c']:+.3f} | {low['sync_c']:+.3f} | {interaction['sync_c']:+.3f} | {interaction['distance_d0']:+.3f} | {interaction['mouth_bbox_MSE']:+.7f} |")
   if harm['sync_c']<0 and interaction['sync_c']>0:combinations.append(NAMES[kind]+str(q)+'dB')
 number=max(int(f.name[:2]) for f in (REPO/'basic-memory/docs/experiments').glob('*.md') if re.match(r'^\d\d-',f.name))+1
 title=f'{number:02d}-may-four-noise-training'
 body=f'''# {title}

✅ CONCLUDED / SUCCEEDED：18个新模型各12000成功FP32更新，0skip，36个固定自然音频测试条件/72clip与矩阵，完整训练状态与媒体独立数值核验通过。单May/seed42/每类第一噪声实例；不作总体或感知结论。

## 主要读数

主推理尺度固定1，无添加合成噪声、原自然PCM16评分。以下每项为两个预定相关10秒clip平均，非独立训练重复；正ΔC有利、负ΔD0/MSE有利。没有按测试结果挑选噪声、强度、checkpoint或尺度。

| 噪声 | SNR dB | 加噪对标准训练ΔC | 加噪时低−标准训练ΔC | 噪声×低训练交互ΔC | 交互ΔD0 | 交互Δmouth MSE |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(interactions)+f'''

按C点估计，“标准训练下加噪为负、低训练交互为正”的条件有{len(combinations)}/8：{', '.join(combinations) if combinations else '无'}。这是未加阈值的符号描述，不是显著性、充分机制证明或原May真实噪声因果确认；D0和嘴部误差须分别看，个别C改善可能来自背景距离变化。所有clip数值和full/guard视图完整保留。

## 冻结设计与配对

用户授权四类噪声全部做。单录音背景讲话、四人babble、独立起伏语音谱形、稀疏合成点击，均用报告65预定第一实例seed20261020；名义整split SNR30dB普通添加档、90dB极弱档，各尺度1/0.5，另无添加噪声两尺度，共18个新模型。无添加组仍有原May未知本底噪声，不能叫声源分离干净语音。

完整May原train168秒先做float32 native z=(x−mean)/sqrt(var+1e−7)，加入固定n30×10^((30−q)/20)，再整体乘1或0.5。没有对混合物再标准化、clipping、PCM量化或时钟变化。同wav2vec2-base revision0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8/权重SHA3249fe98bfc62fcbc26067f724716a6ec49d12c4728a2af1df659013905dff21，encoder冻结；历史20秒chunk/50Hz两帧到25fps、768输入AudioNet路径。30dB和90dB不是已知训练有害标签。

May4150训练帧、550val、1100test；18组都从同一完整随机初始state开始，包括基线，不复用旧训练成品。seed42/4096rays/12000成功FP32更新、原AdamW和60k学习率前缀、EMA0.95每1000更新，固定最终step12000/EMA12。逐step源帧/实际ray hash/LR序列在线和独立完整一致；0skip，无AMP scaler。每2000完整保存模型、梯度、optimizer、scheduler、EMA、metadata、RNG。18组各两步独立roundtrip资格不作正式warmstart。

主问题是同固定推理条件下的差分之差：(noisy_low−noisy_standard)−(clean_low−clean_standard)。它扣除无添加噪声时已有的低尺度训练效应；不能只看noisy_low对clean_standard的对角差。所有组测试driver无新添加噪声，scale1主、0.5次；两预定10秒clip、固定原自然PCM16评分、相同GT ROI和时钟，guard20主/full次。固定最终预算，未按test早停或追加。

## 全部36条件

Sync-C三位小数；D0越小越好，mouth MSE为归一化RGB原值像素误差，不等于人类感知。

| 噪声 | 添加SNR | 训练尺度 | 推理尺度 | Sync-C | D0 | mouth MSE | offsets |
|---|---:|---:|---:|---:|---:|---:|---|
'''+ '\n'.join(full)+f'''

逐clip、训练/验证曲线、完整精度差值、背景距离、Dmin、full视图均在持久根；不对这两个相关clip做总体显著性推断。没有人工盲评或跨身份结论。

## 独立核验与恢复

22完整特征条件/172encoder输入chunk独立重建；18个train首完整chunk与65输入/末层hash桥接，6无添加完整特征与56逐值一致。3个CPU控制测试通过。Ubuntu新持久数据根24294源文件freshSHA通过，实际provider加载像素/姿态/音频与源文件一致。18份最终模型/梯度/optimizer/EMA全有限、step/调度器/实际frame-ray-LR完全配对，0skip。

全部72矩阵{qa['distance_cells']}个距离独立float64重建，最大差{qa['max_distance_error']:.6g}；{qa['frames']}帧RGB/源时钟/原PCM/GT ROI/crop hash检查，selectedfloat原像素误差复算通过。数值QA没有声称人工视觉/感知优劣。

首次部署在0GPU更新时发现先前TTS缓存缺92个原始末尾非本轮监督图像，按原manifest从F精确补齐后全清单重验；原失败与repair_protocol保留，没有修改旧缓存或放松阈值。首只读备份因本地/远端deployment_receipt JSON格式字节不同而停止，v2将远端原字节放remote_evidence；无科学job重放。所有真实训练/资格/渲染/评分job仅单次派发，首次真实失败停止策略未被绕过。

## 产物与限制

F持久根`{OUT.as_posix()}`，协议SHA eb52901d489eeac7bb77d3be4a3a251c99e4c40e18a5f182721a1110a94c0d76；Ubuntu原件`/home/wjj/tts-audio/may-four-noise-training-20261020-v1`；E媒体`E:/Documents/tts-exp/runs/may_four_noise_training_20261020_v1/renders`。源码scripts/experiments/may_four_noise_training_20261020；完整receipt/trace/fullstates/analysis和跨F/E SHA清单保留。没有声称云备份；旧Patrick和其他暂停试验未恢复。

单训练seed42、单固定干扰实例/类型、May单身份、两个相关测试clip、12000步短预算，不能排除细小输入改变优化轨迹带来的个体差异，也不证明充分收敛。即便某人工噪声存在损害及交互，也只支持该条件的实验干扰，不能自动还原原May未知噪声成因。

## Observations
- [result] 18个等初始化/实际采样与更新配对的May模型完成，36测试条件/72矩阵独立QA通过。
- [decision] 用噪声×训练尺度交互扣除基线尺度效应，所有反向条件与两固定剂量均保留。
- [limitation] 单May/seed/干扰实例与短预算，未确认原May噪声机制或感知同步。

## Relations
- concludes [[May四类已知噪声与训练尺度配对对照]]
- follows [[65-may-four-noise-scale-scan]]
- follows [[56-may-postnorm-scale]]
'''
 for retry in range(20):
  r=await write_note(title=title,content=body,directory='docs/experiments',project='tts-exp',note_type='experiment',overwrite=False,metadata=dict(status='concluded',execution_status='SUCCEEDED',scientific_status='SINGLE_SEED_TRAINING_INTERACTION_PILOT',repository='dogLeft233/tts-exp',code_paths=['scripts/experiments/may_four_noise_training_20261020']),output_format='json')
  if r.get('action')=='created':break
  if r.get('action')!='conflict':raise RuntimeError(r)
  old=title;number+=1;title=f'{number:02d}-may-four-noise-training';body=body.replace('# '+old+'\n','# '+title+'\n',1)
 else:raise RuntimeError('publication conflict loop')
 ident=r.get('permalink') or r['file_path'];q=await read_note(identifier=ident,project='tts-exp',output_format='json');assert '216000' in q['content'] or '12000' in q['content']
 spec='tts-exp/experiments/may-四类已知噪声与训练尺度配对对照'
 completed='# May四类已知噪声与训练尺度配对对照\n\n✅ CONCLUDED / SUCCEEDED。正式报告[['+title+']]；18新模型×12000成功FP32更新、0skip、36自然测试条件/72矩阵独立数值QA通过。完整设计/恢复/所有反向条件和限制以正式报告与F根为准。单May/seed42/每类首固定噪声实例，不作原May去噪原因或总体/感知结论。\n\n## Relations\n- concluded_in [['+title+']]\n- follows [[65-may-four-noise-scale-scan]]\n'
 r2=await write_note(title='May四类已知噪声与训练尺度配对对照',content=completed,directory='Experiments',project='tts-exp',note_type='experiment',overwrite=True,metadata=dict(status='concluded',execution_status='SUCCEEDED',scientific_status='SINGLE_SEED_TRAINING_INTERACTION_PILOT',repository='dogLeft233/tts-exp'),output_format='json')
 await read_note(identifier=spec,project='tts-exp',output_format='json')
 await edit_note(identifier='tts-exp/research/研究主线与证据地图',operation='append',content='\n## May四类干扰训练尺度交互：已完成\n\n[['+title+']]：18新模型×12000成功FP32更新/0skip、36自然测试条件/72矩阵与18000帧独立QA通过；完整噪声损害/低尺度效应/差分之差及所有反向条件均报告。单May/seed42/每类首实例，不把人工干扰条件外推原May未知噪声或感知收益。F根may_four_noise_training_20261020_v1。\n',project='tts-exp',output_format='json')
 idx=await read_note(identifier='tts-exp/docs/experiments/readme',project='tts-exp',output_format='json');content=idx['content'];old=re.search(r'现有01–\d+，07尚未占用',content).group()
 await edit_note(identifier='tts-exp/docs/experiments/readme',operation='find_replace',find_text=old,content=f'现有01–{number:02d}，07尚未占用',project='tts-exp',output_format='json')
 rows=[x for x in content.splitlines() if re.match(r'\| \d\d \|',x)];last=max(rows,key=lambda x:int(x.split('|')[1].strip()))
 added=f'| {number:02d} | [May四类噪声训练交互]({title}.md) | 18新模型等初始化/实际采样/12000成功FP32更新，36格，单seed交互先导 | ✅ |'
 await edit_note(identifier='tts-exp/docs/experiments/readme',operation='find_replace',find_text=last,content=last+'\n'+added,project='tts-exp',output_format='json')
 for i in [ident,'tts-exp/docs/experiments/readme','tts-exp/research/研究主线与证据地图']:
  q=await read_note(identifier=i,project='tts-exp',output_format='json');assert title in q['content']
 (OUT/'report-source.md').write_text(body,encoding='utf-8');receipt=dict(passed=True,title=title,identifier=ident,report_path=str(REPO/'basic-memory/docs/experiments'/(title+'.md')));(OUT/'publication_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False),flush=True)
run_with_cleanup(main())
