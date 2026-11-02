"""Publish only reviewed, actual small-pilot evidence. No experiment execution."""
from pathlib import Path
from datetime import datetime,timezone
import json,re,hashlib,shutil
from loguru import logger
logger.remove()
from basic_memory.mcp.tools import read_note,write_note
from basic_memory.cli.commands.command_utils import run_with_cleanup
OUT=Path('F:/Datasets/tts-exp/may_encoder_normalization_pilot_20261019')
REPORT='54-may-encoder-normalization-pilot'
EX='tts-exp/experiments/may同权重编码器归一化训练与推理小规模对照'
TASK='tts-exp/tasks/运行may同权重编码器归一化训练推理对照'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def obs(body,text):return re.sub(r'## Observations\n[\s\S]*?(?=\n## Relations|\Z)','## Observations\n'+text+'\n',body,count=1)
async def main():
 for ident in ['tts-exp/instructions/startup-router','tts-exp/research/研究主线与证据地图','tts-exp/instructions/实验指令','tts-exp/instructions/任务指令','tts-exp/schemas/experiment','tts-exp/schemas/task']:
  n=await read_note(identifier=ident,project='tts-exp',output_format='json');assert n.get('content')
 a=OUT/'analysis';assert json.loads((a/'qa.json').read_text())['passed'];assert json.loads((OUT/'training_independent_review.json').read_text())['passed']
 assert json.loads((a/'visual_review.json').read_text())['passed']
 assert json.loads((OUT/'media_clock_review.json').read_text())['passed']
 cells=json.loads((a/'cells.json').read_text());p=json.loads((OUT/'protocol.json').read_text());headline=[]
 for spec in p['models']:
  mid=spec['id'];on=next(c for c in cells if c['encoder']==mid and c['train_normalize'] and c['test_normalize'] and c['gain_db']==0);off=next(c for c in cells if c['encoder']==mid and not c['train_normalize'] and not c['test_normalize'] and c['gain_db']==0)
  headline.append(f'{mid}匹配on/on−off/off：ΔC{on["sync_c"]-off["sync_c"]:+.3f}、ΔD0{on["distance_d0"]-off["distance_d0"]:+.3f}，mouth MSE相对{(on["mouth_bbox_MSE"]/off["mouth_bbox_MSE"]-1)*100:+.2f}%')
 result='；'.join(headline)+'。这是训练/推理联合流程差异，条件内训练、推理和失配拆分见完整表。'
 summary='✅ 小规模实验完成：四臂各12000实际成功FP32更新、0跳步，24格/48矩阵/12000真实视频帧及原PCM/ROI/像素/距离独立QA通过。'+result+' 单May/单seed/短预算，不作总体或感知结论；Patrick仍暂停。'
 publication=OUT/'final_publication_v1';publication.mkdir(exist_ok=False)
 assets=Path('basic-memory/docs/experiments/assets')/REPORT;assets.mkdir(parents=True,exist_ok=False)
 images=[]
 for name in ['learning_curves.png','crossed_conditions.png','matched_frame_grid.png']:
  shutil.copyfile(a/name,assets/name);assert sha(assets/name)==sha(a/name);images.append(dict(path=str(assets/name),sha256=sha(a/name)))
 body='# '+REPORT+'\n\n'+summary+'\n\n'+(a/'report.md').read_text(encoding='utf-8').split('\n',1)[1]
 body+='\n\n'+ '\n\n'.join(f'![{name}](assets/{REPORT}/{name})' for name in ['learning_curves.png','crossed_conditions.png','matched_frame_grid.png'])
 body+='\n\n## Observations\n- [status] concluded；工程先导，固定预算全部完成。\n- [result] '+result+'\n- [limitation] 同权重模块对照；跨encoder/旧AVE仅描述参照；短预算和一个训练seed不排除训练随机性，不等于完整稳定性或感知获益。\n\n## Relations\n- follows [[53-encoder-input-normalization-ablation]]\n- relates_to [[50-synctalk-may-gain-training-pilot]]\n- supports [[May同权重编码器归一化训练与推理小规模对照]]\n- relates_to [[研究主线与证据地图]]\n'
 report_path=Path('basic-memory/docs/experiments')/(REPORT+'.md');assert not report_path.exists(),'do not replace an existing numbered report'
 written=await write_note(title=REPORT,content=body,directory='docs/experiments',project='tts-exp',note_type='experiment',metadata={'status':'concluded','execution_status':'SUCCEEDED','scientific_status':'ENGINEERING_PILOT_ONLY','outputs':str(OUT),'inputs':'May source SHA and frozen split protocol','model':'same pinned HuBERT-base and wav2vec2-base, native input norm on/off; frozen encoders','code_paths':['scripts/experiments/may_encoder_norm_pilot_20261019']},tags=['may','encoder','normalization','training','pilot'],overwrite=False,output_format='json')
 assert written['action']=='created',written
 rb=await read_note(identifier=written['permalink'],project='tts-exp',output_format='json');assert rb['content'].strip()==body.strip()
 records=[written];(publication/'report_readback.json').write_text(json.dumps(rb,ensure_ascii=False,indent=2),encoding='utf-8')
 for ident in [EX,TASK,'tts-exp/research/研究主线与证据地图','tts-exp/tasks/task-board','tts-exp/docs/experiments/readme']:
  old=await read_note(identifier=ident,project='tts-exp',output_format='json');fm=old['frontmatter'];text=old['content'];(publication/(str(len(records))+'_before.json')).write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
  if ident==EX:
   frozen=text[text.index('用户授权用May'):text.index('## Observations')]
   frozen=frozen.replace('冻结方案时尚无科学执行；现在实际状态见顶部执行头。候选代码不算结果。','该段为执行前冻结设计；结果以正式报告和真实产物为准。')
   text='# '+old['title']+'\n\n'+summary+'\n\n正式报告[['+REPORT+']]；F证据'+str(OUT)+'。\n\n## 冻结设计（执行前记录）\n'+frozen+'\n## Observations\n- [status] concluded\n- [result] '+result+'\n- [limitation] 只单May/单seed/12000更新，不外推RMS0.1/FP16 Patrick数值失败，不作感知结论。\n\n## Relations\n- concluded_in [['+REPORT+']]\n- follows [[53-encoder-input-normalization-ablation]]\n- relates_to [[50-synctalk-may-gain-training-pilot]]\n'
   fm.update(status='concluded',execution_status='SUCCEEDED',scientific_status='ENGINEERING_PILOT_ONLY')
  elif ident==TASK:
   text='# '+old['title']+'\n\n'+summary+'\n\nRequirements逐条验证：准入见qualification_independent_review.json；四臂全state/trace见training_independent_review.json；24格/两clip和RGB/PCM/ROI/matrix见analysis/qa.json及visual_review.json；最终报告[['+REPORT+']]。\n\n'+text[text.index('## Requirements'):]
   text=obs(text,'- [status] done\n- [result] 固定四臂/48000实际更新与24格/48矩阵完成，F全成员与独立QA、实际看图和报告发布通过。\n- [close_date] October 19, 2026\n- [limitation] 此任务完成不代表Patrick coarse10/new27或原24full完成。')
   assert text.count('- [status]')==1 and '- [status] done' in text
   fm.update(status='done',current_step=3,completed='October 19, 2026')
  elif ident.endswith('研究主线与证据地图'):
   title='## May同权重归一化的下游小规模检验';index=text.index(title);end=text.find('\n## ',index+len(title));tail=text[end:] if end>=0 else ''
   text=text[:index]+title+'\n\n[['+REPORT+']] / [[May同权重编码器归一化训练与推理小规模对照]]：'+summary+' 不把跨encoder旧AVE60000差异归因归一化；条件内两轴和原PCM/时钟保持，旧RMS/恢复probe及Patrick暂停不改。\n'+tail
  elif ident.endswith('task-board'):
   text='\n'.join(line for line in text.splitlines() if '[[运行May同权重编码器归一化训练推理对照]]' not in line)+'\n'
  else:
   assert '54-may-encoder-normalization-pilot' not in text
   text=text.replace('现有01–53','现有01–54')
   row='| 54 | [May同权重归一化训练/推理](54-may-encoder-normalization-pilot.md) | 四臂各12000成功更新，24格/48矩阵；同编码器输入norm开关及两轴交叉，单May/单seed短程先导 | ✅ |\n'
   text=text.replace('\n## 专题/总结文档', '\n'+row+'\n## 专题/总结文档',1)
  r=await write_note(title=old['title'],content=text,directory=str(Path(old['file_path']).parent),project='tts-exp',note_type=fm.get('type','note'),metadata={k:v for k,v in fm.items() if k not in ['title','type','permalink','tags']},tags=fm.get('tags',[]),overwrite=True,output_format='json')
  after=await read_note(identifier=r['permalink'],project='tts-exp',output_format='json');assert after['content'].strip()==text.strip();records.append(r)
  (publication/(str(len(records))+'_readback.json')).write_text(json.dumps(after,ensure_ascii=False,indent=2),encoding='utf-8')
 receipt=dict(passed=True,published_utc=datetime.now(timezone.utc).isoformat(),report=written['permalink'],notes=records,image_mirrors=images,task_done=True,original_patrick_paused=True,original_large_objective_not_done=True)
 (OUT/'final_memory_publication_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))
run_with_cleanup(main())
