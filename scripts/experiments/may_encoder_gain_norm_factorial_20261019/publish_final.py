"""Publish the completed, reviewed fixed factorial only; never execute science here."""
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import json,re,hashlib,shutil
from loguru import logger
logger.remove()
from basic_memory.mcp.tools import read_note,write_note
from basic_memory.cli.commands.command_utils import run_with_cleanup
OUT=Path('F:/Datasets/tts-exp/may_encoder_gain_norm_factorial_20261019');REPORT='55-may-gain-normalization-factorial'
EX='May训练音量与推理音量的同权重归一化交叉对照';TASK='运行May训练推理音量与归一化交叉对照'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
async def save(old,text,metadata=None):
 fm=old['frontmatter'];fm.update(metadata or {})
 r=await write_note(title=old['title'],content=text,directory=str(Path(old['file_path']).parent),project='tts-exp',note_type=fm.get('type','note'),metadata={k:v for k,v in fm.items() if k not in ['title','type','permalink','tags']},tags=fm.get('tags',[]),overwrite=True,output_format='json')
 n=await read_note(identifier=r['permalink'],project='tts-exp',output_format='json');assert n['content'].strip()==text.strip();return r,n
async def main():
 for i in ['tts-exp/instructions/startup-router','tts-exp/research/研究主线与证据地图','tts-exp/instructions/实验指令','tts-exp/instructions/任务指令','tts-exp/schemas/experiment','tts-exp/schemas/task']:await read_note(identifier=i,project='tts-exp',output_format='json')
 a=OUT/'analysis';qa=json.loads((a/'qa.json').read_text());assert qa['passed'] and qa['total_cells']==72 and qa['total_matrices']==144
 for path in [a/'visual_review.json',OUT/'training_independent_review.json',OUT/'media_clock_review.json',OUT/'parent_render_retirement_receipt.json']:assert json.loads(path.read_text())['passed']
 cells=json.loads((a/'cells.json').read_text());now=datetime.now(timezone.utc).astimezone(ZoneInfo('Asia/Shanghai'));date=f'{now:%B} {now.day}, {now.year}'
 summary='✅ 已完成：八新模型各12000成功FP32更新、0skip，复用四个已资格训练0dB模型；72条件/144矩阵，新增24000帧与96矩阵全数值/像素/PCM/ROI/时钟独立QA及实际图检通过。固定同encoder权重切norm，单May/单seed/短预算，不作总体或感知结论；Patrick仍暂停。'
 p=OUT/'final_publication_v1';p.mkdir(exist_ok=False);assets=Path('basic-memory/docs/experiments/assets')/REPORT;assert not (Path('basic-memory/docs/experiments')/(REPORT+'.md')).exists();assets.mkdir(parents=True,exist_ok=False)
 image_records=[]
 for name in ['gain_matrices.png','learning_curves.png','matched_frame_grid.png']:
  shutil.copyfile(a/name,assets/name);assert sha(a/name)==sha(assets/name);image_records.append(dict(path=str(assets/name),sha256=sha(a/name)))
 body='# '+REPORT+'\n\n'+summary+'\n\n'+(a/'report.md').read_text(encoding='utf-8').split('\n',1)[1]
 body+='\n\n'+'\n\n'.join(f'![{n}](assets/{REPORT}/{n})' for n in ['gain_matrices.png','learning_curves.png','matched_frame_grid.png'])
 body+='\n\n## Observations\n- [status] concluded\n- [result] 完整72格训练gain×推理gain×norm交叉已执行并独立核验，0gain原件不重放。\n- [limitation] 单训练seed的不同gain模型存在优化轨迹随机性；编码器内模块对照不外推整体编码器优劣、感知或旧Patrick RMS因果。\n\n## Relations\n- follows [[54-may-encoder-normalization-pilot]]\n- relates_to [[50-synctalk-may-gain-training-pilot]]\n- supports [['+EX+']]\n- relates_to [[研究主线与证据地图]]\n'
 r=await write_note(title=REPORT,content=body,directory='docs/experiments',project='tts-exp',note_type='experiment',metadata=dict(status='concluded',execution_status='SUCCEEDED',scientific_status='ENGINEERING_PILOT_ONLY',inputs='same May, report54 exact random initial and encoder pins',outputs=str(OUT),model='same encoder native norm x training gain x driver gain',code_paths=['scripts/experiments/may_encoder_gain_norm_factorial_20261019']),tags=['may','gain','normalization','factorial'],overwrite=False,output_format='json');assert r['action']=='created';records=[r]
 n=await read_note(identifier=r['permalink'],project='tts-exp',output_format='json');assert n['content'].strip()==body.strip();(p/'report_readback.json').write_text(json.dumps(n,ensure_ascii=False,indent=2),encoding='utf-8')
 scope=json.loads((OUT/'scope_publication_receipt.json').read_text())
 for record in scope['notes']:
  old=await read_note(identifier=record['permalink'],project='tts-exp',output_format='json');text=old['content'];meta={}
  if old['title']==EX:
   frozen=text.split('## 冻结设计',1)[1].split('## Observations',1)[0]
   text='# '+EX+'\n\n'+summary+'\n\n正式报告[['+REPORT+']]；证据根'+str(OUT)+'。\n\n## 执行前冻结设计\n'+frozen+'\n## Observations\n- [status] concluded\n- [result] 固定因素矩阵完整且独立QA/图检/发布通过。\n- [limitation] 不外推RMS0.1或完整训练/感知。\n\n## Relations\n- concluded_in [['+REPORT+']]\n- follows [[54-may-encoder-normalization-pilot]]\n';meta=dict(status='concluded',execution_status='SUCCEEDED',scientific_status='ENGINEERING_PILOT_ONLY')
  elif old['title']==TASK:
   text='# '+TASK+'\n\n'+summary+'\n\n逐项证据：来源/缓存/入口见parent_full_member_import_review和feature/qualification独立review；八新模型见training_independent_review；完整72格见analysis/qa、visual_review和正式报告[['+REPORT+']]。\n\n'+text[text.index('## Requirements'):]
   text=re.sub(r'## Observations\n[\s\S]*?(?=\n## Relations|\Z)','## Observations\n- [status] done\n- [result] 全部固定Scenario完成，跨卷全成员读回、独立QA与实际图检通过。\n- [close_date] '+date+'\n- [limitation] 此Task完成不代表旧Patrick或原24full完成。\n',text,count=1);meta=dict(status='done',current_step=3,completed=now.isoformat());assert text.count('- [status]')==1
  elif old['title']=='Task Board':text='\n'.join(l for l in text.splitlines() if '[['+TASK+']]' not in l)+'\n'
  else:
   title='## May训练音量与归一化的交互';start=text.index(title);end=text.find('\n## ',start+len(title));tail=text[end:] if end>=0 else '';text=text[:start]+title+'\n\n[['+REPORT+']] / [['+EX+']]：'+summary+' 训练gain与推理gain效应、norm失配、原时钟D0和嘴部误差分别报告；不按test选最佳档位。旧报告50的18大render、36裁剪缓存和87个step2000..58000中间断点副本现以同SHA云包和四个恢复locator定位，原结果/矩阵/embeddings/三个最终cp不改；报告54和本次所有cp仍本地保持。\n'+tail
  r,n=await save(old,text,meta);records.append(r);(p/(str(len(records))+'_readback.json')).write_text(json.dumps(n,ensure_ascii=False,indent=2),encoding='utf-8')
 old=await read_note(identifier='tts-exp/docs/experiments/readme',project='tts-exp',output_format='json');text=old['content'];assert REPORT not in text;text=text.replace('现有01–54','现有01–55');text=text.replace('\n## 专题/总结文档','\n| 55 | [May训练/推理音量×归一化](55-may-gain-normalization-factorial.md) | 8新×12000更新+4旧0gain；72格/144矩阵同权重交叉，小预算单seed | ✅ |\n\n## 专题/总结文档',1);r,n=await save(old,text);records.append(r)
 result=dict(passed=True,published_client_time=now.isoformat(),report=records[0]['permalink'],notes=records,image_mirrors=image_records,task_done=True,old_patrick_still_paused=True,new_science_cloud_backup_not_claimed=True)
 (OUT/'final_memory_publication_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
run_with_cleanup(main())
