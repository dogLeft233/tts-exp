"""Create reviewable local results only from actual qualified coarse10/new27 artifacts."""
import csv,hashlib,itertools,json,math
from pathlib import Path
BASE=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017');OUT=BASE/'trials/patrick_ave_rms_coarse_v1';S=OUT/'score_27_v1';D=S/'delivery_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def run():
 review=read(D/'independent_review.json');assert review['passed'] and review['cells']==27 and review['checked_matrices']==162
 data=read(S/'all27_preview.json');ready=read(OUT/'coarse_ready_for_27_grid.json');assert data['passed'] and ready['passed']
 keys=lambda c:(c['training_gain_db'],c['driver_gain_db'],c['evaluation_gain_db'])
 indexed={keys(c):c for c in data['cells']};assert len(indexed)==27
 oldpath=BASE/'trials/fp16_other_identities_trial_v1/early_score_preview_v2/all27_preview.json'
 assert sha(oldpath)=='0c727888b1fbeb09e573ab26bc7129d453aa03b45de345b460d66973ac6fac1a'
 old=read(oldpath);previous={keys(c):c for c in old['cells']};assert set(previous)==set(indexed)
 comparison=[]
 for key,new in indexed.items():
  before=previous[key];b={c['clip_id']:c for c in before['clips']};assert set(b)=={c['clip_id'] for c in new['clips']}
  for c in new['clips']:
   for support in ['guard20','full']:
    for metric in ['sync_c','sync_d','distance_d0','background_median']:
     x=b[c['clip_id']][support][metric];y=c[support][metric]
     comparison.append(dict(training_gain_db=key[0],driver_gain_db=key[1],scorer_gain_db=key[2],clip_id=c['clip_id'],support=support,metric=metric,old_no_RMS=x,new_RMS=y,new_minus_old=y-x,old_offset=b[c['clip_id']][support]['av_offset'],new_offset=c[support]['av_offset']))
 with (D/'RMS_minus_original_per_clip.csv').open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
 # Re-read all original traces; superseded intermediate checkpoint bytes need not still exist.
 totals={};trace_rows=[]
 for gain in [-6,0,6]:
  count=0;updates=0
  for epoch in range(1,11):
   p=OUT/'training/PatrickLeahy/seed42'/f'gain_{gain:+d}'/'coarse'/f'ep{epoch:04d}/attempt01'
   receipt=read(p/'epoch_receipt.json');raw=(p/'attempts.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==receipt['trace_sha256']
   rows=[json.loads(x) for x in raw.splitlines()];assert len(rows)==6000
   for i,row in enumerate(rows):
    assert row['global_step']==(epoch-1)*6000+i+1 and math.isfinite(row['loss']);updates+=int(row['successful_update']);count+=1
    assert updates==row['successful_updates']
    trace_rows.append(dict(gain_db=gain,epoch=epoch,global_step=row['global_step'],loss=row['loss'],successful_update=row['successful_update'],amp_skip=row['amp_skip'],scale_after=row['scale_after']))
  assert count==60000 and updates==ready['phase']['actual_updates_by_gain'][str(gain)];totals[gain]=dict(attempts=count,successful_updates=updates,amp_skips=count-updates)
 with (D/'training_loss_updates.csv').open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(trace_rows[0]));w.writeheader();w.writerows(trace_rows)
 def table(metric,support,scorer):
  lines=['| 训练增益 / 驱动增益 | −6 dB | 0 dB | +6 dB |','|---|---:|---:|---:|']
  for train in [-6,0,6]:lines.append(f'| {train:+d} dB | '+' | '.join(f"{indexed[train,driver,scorer]['mean_'+support][metric]:.3f}" for driver in [-6,0,6])+' |')
  return '\n'.join(lines)
 fixed=[]
 for gain in [-6,0,6]:
  c=indexed[gain,0,0]['mean_guard20'];b=previous[gain,0,0]['mean_guard20']
  fixed.append(f"| {gain:+d} | {b['sync_c']:.3f} | {c['sync_c']:.3f} | {c['sync_c']-b['sync_c']:+.3f} | {b['sync_d']:.3f} | {c['sync_d']:.3f} |")
 contract=read(OUT/'contract.json');inputqa=read(OUT/'independent_normalized_input_review.json')
 body=f'''# Patrick：同AVE逐录音RMS归一化 coarse10 三档评测

PatrickLeahy / seed42，三档均完成各60000尝试，固定epoch10 EMA60，9个新FFV1/PCM视频、27格、6片等权，162矩阵独立复核通过。训练更新控制状态：{ready['phase']['status']}。仅单身份、单seed的实际模型描述；不将分数变化称为感知改善。

## 固定驱动0 / 评分0主端点

| 训练dB | 原C | RMS C | ΔC（新减旧） | 原D | RMS D |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(fixed)+f'''

## 方法与解释边界

同一官方AVE与权重，先−6/0/+6dB，再逐完整录音float64 RMS缩放到固定0.1（−20dBFS），float32进入原mel；无去均值、epsilon、clip、limiter或时钟改动。三个gain均分别重提AVE和完整训练，不复用0档训练state。独立公式与原PCM/时钟/完整特征形状核验通过。三档归一化后的实际AVE是否逐位一致，以OUT/features/PatrickLeahy/ave_rms_qualification.json为准。

原数据、共同完整随机初始化、FP16/defaultGradScaler/guard、65536rays/maxsteps16、loss/AdamW/EMA与采样政策保持；不由旧coarse/fine模型warmstart。实际尝试/成功/skip：{totals}。原未归一化实验的CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES始终保留；旧新模型预算混杂及单身份限制意味着并列表不能直接归因为归一化的总体因果或感知收益。

评分音轨不跟着AVE归一化：0严格原PCM16，±6用float64原PCM数值单位，SyncNetFP32/TF32off、原source ROI、原时钟。guard20为主，full为边界敏感性；driver/scorer三轴不混合。没有按test挑checkpoint、条件或延长预算。

## guard20完整27格
'''
 for metric in ['sync_c','sync_d']:
  for scorer in [-6,0,6]:body+=f'\n### {metric}，评分 {scorer:+d} dB\n\n'+table(metric,'guard20',scorer)+'\n'
 body+=f'''
## 独立核验与持久入口

执行端每个视频1500帧FFV1 RGB/原PCM逐值读回，真实9固定帧重复资格；源ROI/原六clip保持。最终另用按列NumPy float64公式重建全部162矩阵，max matrix error={review['matrix_max_abs_error']:.12g}，endpoint error={review['endpoint_max_abs_error']:.12g}，原门3e−5。27格六clip等权、27组PNG/head hash和9原PCM复核通过；全部9媒体已独立Linux cloud整包读回。最终CPU复核未重新恢复全部lossless电影或重读cloud payload，证据范围保留。

表格scores_27_cells.csv、距离曲线mean_distance_curves.csv、训练180000行training_loss_updates.csv、逐clip新减旧RMS_minus_original_per_clip.csv、图patrick_27_grid.png；视频H264预览在各renders/*/local/preview.mp4，lossless按cloud_storage_locator.json显式恢复。

此报告本机数据与数值QA完成后仍需一次图像目视和Basic Memory持久发布。旧fine14/Chris/Cathy继续暂停，原24完整模型目标未完成。
'''
 (D/'report.md').write_text(body,encoding='utf-8')
 entries=[]
 for p in sorted(D.rglob('*')):
  if p.is_file():entries.append(dict(path=p.relative_to(D).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
 write(D/'delivery_manifest.json',entries)
 for row in entries:assert sha(D/row['path'])==row['sha256']
 write(D/'delivery_candidate_receipt.json',dict(passed=True,numerical_QA_passed=True,persistent_local_files_verified=True,new27_sha256=sha(S/'all27_preview.json'),report_sha256=sha(D/'report.md'),review_sha256=sha(D/'independent_review.json'),training_control_status=ready['phase']['status'],visual_chart_review_pending=True,Basic_Memory_publication_pending=True,task_done=False))
 print('RMS_REPORT_CANDIDATE_READY_FOR_VISUAL_AND_MEMORY_FINALIZATION',flush=True)
if __name__=='__main__':run()
