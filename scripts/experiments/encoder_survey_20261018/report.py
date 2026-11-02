"""Create a concise evidence report from completed scientific artifacts only."""
from pathlib import Path
import json,shutil,hashlib,numpy as np
from prepare import ROOT,sha
from run import BASE

def main():
 j=json.loads((BASE/'analysis/results.json').read_text(encoding='utf8'))
 qa=json.loads((BASE/'analysis/qa.json').read_text(encoding='utf8'))
 assert j['status']=='COMPLETE' and qa['status']=='PASS_DATA_ONLY'
 p=json.loads((BASE/'protocol.json').read_text(encoding='utf8'))
 out=ROOT/'runs/encoder_survey_20261018';out.mkdir(exist_ok=False)
 files=['results.json','table.csv','qa.json','encoder_commonality.png','encoder_commonality.svg','encoder_commonality.pdf',
  'final_layer_commonality.png','final_layer_commonality.svg','final_layer_commonality.pdf']
 for name in files:shutil.copyfile(BASE/'analysis'/name,out/name)
 shutil.copyfile(BASE/'protocol.json',out/'protocol.json')
 for name in ['batch_status.json','cached_reference_extension.json','prefetch_receipt.json','native_mask_repair.json','recovery_status.json']:
  if (BASE/name).exists():shutil.copyfile(BASE/name,out/name)
 sources=[]
 for f in Path(__file__).parent.glob('*.py'):
  dst=BASE/'execution_sources'/f.name;shutil.copyfile(f,dst);sources.append({'file':f.name,'sha256':sha(f)})
 member=[]
 for folder in [BASE/'input',BASE/'models',BASE/'failures',BASE/'execution_sources',BASE/'analysis',BASE/'weights']:
  for f in sorted(folder.rglob('*')):
   if f.is_file() and not (folder.name=='weights' and f.suffix!='.json'):
    member.append({'relative':f.relative_to(BASE).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)})
 manifest={'artifact_root':str(BASE),'models':j['completed_models'],'new_encoders':len(p['models']),
  'new_forward_count':sum(json.loads((BASE/'models'/m/'receipt.json').read_text())['forward_count'] for m in j['completed_models']),
  'prior_failed_forward_count_inferred_from_frozen_call_path':2,
  'source_files':sources,'scientific_members':member,'scientific_member_bytes':sum(r['bytes'] for r in member),
  'no_new_TFG_or_SyncNet_or_training':True,'cloud_uploaded':False}
 (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 rows=[r for r in j['results'] if r['domain']=='full_phone' and r['layer_fraction']==1]
 def get(mid,ep,domain='full_phone'):
  return next(r for r in j['results'] if r['model']==mid and r['domain']==domain and r['layer_fraction']==1 and r['endpoint']==ep)
 cov={k:sum(x['common_phones'] for x in v) for k,v in j['coverage'].items()}
 lines=['---','type: experiment','status: concluded','---','# 52 — 主流语音编码器的 TTS 与增益方向共性','',
 '## 问题与执行范围','',
 '用户要求把已有的编码方向共性分析扩展到主流编码器，并由ROOT直接执行。完成6个语音内容编码器家族的11个公开checkpoint，另在完全相同音素支持上重用Wav2Lip原生编码参照；不是穷举全部音频模型。',
 f'本次正式新编码前向 **{manifest["new_forward_count"]}** 次（11×[20句×6臂+1个RAW重复]），最终11/11完整结束、0缺模型。W2v-BERT首尝试另有2次在首句重复检查前完成的前向，因原生padding零向量停止；按实际attention mask修复后单独完成121前向，其他10模型没有重跑。总调用1333次，失败原件及精确调用路径推定依据保留。0新TFG/SyncNet前向、0训练。Wav2Lip参照只读取已验证缓存，新增前向为0。','',
 '## 设计与识别边界','',
 '- 固定同一CAL20：10位男性、每人2句，同文本自然/Qwen TTS，历史已暴露探索队列；新增模型数不等于新增独立说话人数。',
 '- 所有编码器读取完全相同的原16k float32 PCM。TTS_RAW比较两侧原电平；TTS_FIXED双方RMS均0.0376838172675746；纯±3dB以自然FIXED为基线，严格保持其内容和原时钟。',
 '- TTS和自然语音按已有MFA同音素分别取原生帧，不插值或重定时。该比较不是两个发声实例拥有相同局部时钟；没有独立自然重录，所以不能将TTS差异全部识别为合成特有因素。',
 '- 保留官方原生前处理；wave模型按config的卷积感受野中心确定时间，Whisper为20ms编码步长/centered STFT，W2v-BERT为25ms fbank、10ms hop、两帧stack的17.5ms首中心。Whisper仅保留原时长内帧，30秒原生上下文仍保留。',
 f'- 跨模型主表使用12条路径共同的 **{cov["full_phone"]}对匹配音素**；中央50%敏感性为 **{cov["inner50"]}对**，20/20句、10/10人保留。所有层原生支持结果另存；跨模型共同支持表固定0/1/2/middle/last层，末层统一展示，不选最佳层。',
 '- 每帧单位化→音素内平均再单位化→目标减来源。音素、句、说话人依次等权；10人45对余弦不是45个独立实验。LOSO参考轴只由其他9人定义，另计算各音素改变量对该轴的局部余弦。',
 '- 新编码器方向分辨率门：单位向量变化RMS≥1e-5且大于重复差异10倍。低于门时不解释共同方向余弦；该门不表示严格零响应。连续幅度照常报告，接近门的结果保留其数值边界。Wav2Lip缓存未新重复，不虚构重复值。',
 '- 不同模型维数和坐标不同；只比较每个模型内部的几何统计，不将跨模型向量直接求cos。无最佳层、最佳剂量选择或总体显著性主张。','',
 '## 末层 TTS 方向：共同支持','',
 '| 编码器 | RAW跨人cos | FIXED跨人cos | FIXED留一人cos | FIXED局部音素cos |',
 '|---|---:|---:|---:|---:|']
 for mid in j['completed_models']:
  a=get(mid,'TTS_RAW');b=get(mid,'TTS_FIXED')
  lines.append(f'| {mid} | {a["pairwise_cos_mean"]:.3f} | {b["pairwise_cos_mean"]:.3f} | {b["loso_cos_mean"]:.3f} | {b["loso_local_phone_cos_mean"]:.3f} |')
 lines+=['','“跨人cos”比较每人的平均改变量；“局部音素cos”先比较每个音素改变量与其他人的共同轴，再依次句/人等权。二者不能互换：平均轴共性不意味着每个音素受到同一种变化。','',
 '## 末层纯增益方向：同内容、同原时钟','',
 '| 编码器 | 官方wave归一化 | +3dB方向变化RMS | +3dB跨人cos | −3dB反向跨人cos |',
 '|---|---|---:|---:|---:|']
 for mid in j['completed_models']:
  a=get(mid,'GAIN_P3');b=get(mid,'GAIN_M3_SIGNED');norm=a['do_normalize']
  if mid.startswith('whisper'):norm='原生log-mel'
  elif mid=='w2vbert2':norm='原生mel-CMVN'
  else:norm='开启' if norm else '关闭'
  def c(r):return f'{r["pairwise_cos_mean"]:.3f}' if r['direction_resolved'] else '低于方向分辨率'
  lines.append(f'| {mid} | {norm} | {a["mean_chord_rms_native"]:.3e} | {c(a)} | {c(b)} |')
 lines+=['','## 本次得到的具体结论','',
 '1. **固定RMS后，TTS的平均方向共性在11个预训练编码器中仍保留。** 末层跨人cos为0.165–0.506；例如中文HuBERT0.382、wav2vec2 base0.317、Whisper base0.478、XLS-R0.506、W2v-BERT0.503。相同663音素支持下，Wav2Lip从RAW0.329降至FIXED0.015。旧Wav2Lip结果得到同支持复现，同时其电平混淆程度应限定该接收编码器。',
 '2. **纯增益的共享方向和变化量级依赖具体原生路径。** Wav2Lip、WavLM base-plus和Whisper base/small的+3dB方向变化RMS为0.0721/0.0350/0.0275/0.0311，跨人cos0.884/0.479/0.613/0.577；若干wave-normalized/CMVN路径的变化在1e-6–5e-5量级。HuBERT large、XLS-R、data2vec和WavLM large的部分值虽过预定1e-5门，仍须按连续的小幅度解释，不能与1e-2量级响应视为相同强度。WavLM base-plus关闭wave归一化、large开启；模型权重也不同，本实验不将两者差异全部单因归给归一化。',
 '3. **平均方向比局部变化更一致。** 11个预训练编码器FIXED的句均LOSO为0.327–0.698，而各音素对LOSO轴的平均cos仅0.068–0.268。在这些表示中存在可复用的平均分量，但把它当作所有音素的同一局部改变量会丢失大量结构。',
 '4. **原始TTS方向与增益轴高度重合具有编码器边界。** Wav2Lip的RAW总体轴cos0.933；WavLM base-plus、Whisper base/small分别0.146/0.177/0.105，统一RMS后为−0.184/0.081/0.022。下表完整列出预定分辨率门通过的末层路径，近门小信号须结合上表变化幅度阅读。','',
 '| 编码器 | RAW TTS轴 vs +3dB轴 | FIXED TTS轴 vs +3dB轴 |','|---|---:|---:|']
 for mid in j['completed_models']:
  a=get(mid,'GAIN_P3')
  if not a['direction_resolved']:continue
  pair=[r for r in j['gain_TTS_comparisons'] if r['model']==mid and r['domain']=='full_phone' and r['layer']==a['layer']]
  vals={r['TTS_endpoint']:r['grand_mean_cos'] for r in pair}
  if vals:lines.append(f'| {mid} | {vals["TTS_RAW"]:.3f} | {vals["TTS_FIXED"]:.3f} |')
 lines+=['','纯增益变化RMS在各模型真实帧支持上计算；其共同方向统计仍在共同音素支持上计算。“−3dB反向”指FIXED−MINUS3，统一为向更高电平的方向。完整输入前处理相对L2、各层半音素结果、范数响应和TTS对增益轴关系见实际receipt及结果JSON。','',
 '## 图与完整结果','',
 '![共同支持的分层几何](assets/52-encoder-direction-commonality/encoder_commonality.png)','',
 '![末层共性](assets/52-encoder-direction-commonality/final_layer_commonality.png)','',
 '同一encoder的固定层位置全部展示；原生全部层结果为`models/<id>/summary_native_support.json`，不要把其中不同帧支持的数值混入共同支持主表。','',
 '## 验证、成本与复现','',
 f'- 11个官方revision/权重body经实际SHA核对；可用LFS SHA均匹配，实际encoder没有missing/mismatched参数。首句RAW重复控制记录在每模型receipt。',
 f'- 一次数据-only消费者核验 **{qa["files"]}个实际NPZ**、**{qa["common_support_stat_checks"]}组共同支持统计**，单位方向范数最大误差{qa["max_direction_norm_error"]:.3e}，跨人/LOSO/局部音素统计重建最大误差{qa["max_stat_error"]:.3e}；0模型前向。',
 '- 本机RTX4060、FP32、TF32关闭、batch1、最多50%PyTorch allocator，逐模型退出释放资源；未占Ubuntu其他会话的训练GPU。前四模型使用原producer，其后增加low_cpu_mem_usage以减小加载内存，forward数学与科学协议未改，两个实际源码正文均封存。',
 '- W2v-BERT attention mask修复只排除原生无效padding位置，原时钟/剂量/科学分辨率门不变；首句零向量控制失败的原receipt/log保留在failures/w2vbert2_attempt01，修复版执行正文另存run_native_mask.py。旧batch_status的rc1不被篡改，最终成功见recovery_status及新模型receipt。',
 '- 公共模型权重成功后作为可重下载缓存回收，保留固定revision、配置和SHA。空间通过回收已验证的冗余May传输分片/打包副本、pip下载缓存和NTFS透明压缩解决；全部643 May科学成员重新SHA通过且保留，已独立云读回的分片保留。见[[编码器扩展的执行环境与资产存储]]。',
 f'- 持久实际根`{BASE.as_posix()}`；仓库小型镜像`runs/encoder_survey_20261018/`含protocol/results/table/QA/图及成员索引。科学成员{len(member)}个、{manifest["scientific_member_bytes"]} bytes；本轮新产物未声明云同步。',
 '- 代码`scripts/experiments/encoder_survey_20261018/{prepare,run,cache_reference,analyze,verify,plot,report}.py`；精确执行正文在持久根execution_sources。原输出create-only，勿盲目重跑覆盖；重分析使用已有原始NPZ、新输出目录，协议源SHA以实际receipt为准。','',
 '## 意义与限制','',
 '本实验把“编码方向存在差异”“变化具有跨人共性”“能够迁移生成收益”分开。前两者可以在同一语料上随编码器、原生归一化和层深发生不同变化；本次没有新的生成干预，因此不新增Sync-C或感知质量收益结论。Wav2Lip既有共性和留出生成结论仍限定其模型与合同，不自动外推为通用编码轴。','',
 'CAL20只有10位已暴露男性，不能把模型覆盖扩展称作跨人口/语言确认；全部使用一个Qwen来源，没有独立自然重录或多个TTS来源因果分离。LibriSpeech音频未进入本次队列，模型自身预训练来源并不等于当前队列已通过污染审计。','',
 '下一有价值的生成验证只应针对具有可测、可复制局部响应且与接收TFG相接的路径；当前暂停的通用增强头不因几何结果自动恢复。','',
 '## Observations','',
 '- [status] concluded；11新编码器/1331正式前向，另2次失败前向，及1个缓存原生参照，共同支持几何核验完成。',
 '- [decision] 保留每个编码器原生前处理，低于方向分辨率不解释cos；平均方向与局部音素方向分开。',
 '- [limitation] 方向共性不是跨模型同一坐标，也不等于生成或感知收益。','',
 '## Relations','',
 '- follows [[TTS 编码方向共性与电平混淆的配对音素补算]]',
 '- follows [[纯增益编码方向的跨说话人共性补算]]',
 '- relates_to [[增益共同方向与残差的留出说话人生成验证]]',
 '- relates_to [[主流语音编码器的方向共性扩展]]',
 '- relates_to [[编码器扩展的执行环境与资产存储]]',
 '- relates_to [[研究主线与证据地图]]']
 # Pin actual official checkpoints, rather than treating family names as versions.
 models_table=['','## 官方checkpoint与实际权重','',
  '| 路径 | 固定版本 | 权重SHA前16位（完整值见model_source） |','|---|---|---|']
 for spec in p['models']:
  source=json.loads((BASE/'models'/spec['id']/'model_source.json').read_text(encoding='utf8'))
  weights=next(f for f in source['files'] if f['file'].endswith(('.bin','.safetensors')))
  models_table.append(f'| {spec["id"]} | [{spec["repo"]}@{source["revision"][:12]}](https://huggingface.co/{spec["repo"]}/tree/{source["revision"]}) | {weights["sha256"][:16]} |')
 insert=lines.index('## 验证、成本与复现');lines[insert:insert]=models_table+['']
 text='\n'.join(lines)+'\n';(out/'report_body.md').write_text(text,encoding='utf8')
 asset=ROOT/'basic-memory/docs/experiments/assets/52-encoder-direction-commonality';asset.mkdir(exist_ok=False)
 for name in files:
  if name.endswith(('.png','.svg','.pdf')):shutil.copyfile(out/name,asset/name)
 print(json.dumps({'report_body':str(out/'report_body.md'),'manifest_sha256':sha(out/'artifact_manifest.json'),
  'new_forwards':manifest['new_forward_count'],'models_total':len(j['completed_models']),'common_phone_count':cov}))
if __name__=='__main__':main()
