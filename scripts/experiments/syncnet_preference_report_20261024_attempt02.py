"""Report-only recovery after missing matplotlib; no extraction or fitting."""
from pathlib import Path
import json,hashlib,datetime,sys,time
from PIL import Image,ImageDraw
R=Path('/home/wjj/tts-audio/tts-exp/runs/syncnet_preference_probe_20261024_attempt01')
def read(n):return json.loads((R/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    started=time.time();full=read('full_results.json');strict=read('qualified_results.json');gate=read('gate.json');im=read('input_manifest.json')
    pins={n:sha(R/n) for n in ['full_results.json','qualified_results.json','full_oof.json','qualified_oof.json','hubert_pooled.npz','protocol.json']}
    receipt={'stage':'REPORT_ONLY_RECOVERY_AFTER_MISSING_MATPLOTLIB','original_analysis_wrapper_exit':1,'no_refit':True,'no_new_inference':True,'scientific_input_pins':pins,'source_sha256':sha(Path(__file__)),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':sys.argv}
    dst=R/'report_attempt02';dst.mkdir(exist_ok=False)
    lines=['# 归一化三臂音频的原生 SyncNet 偏好探针','', '**未通过预设推进门。** 在20个已有来源上，加入冻结HuBERT表示的M3没有超过既有六声学特征和支持数基线M1。这里只检验固定表示与线性探针，不能据此认定不存在可解释机制。','',f"主比较MAE改善（M1−M3，正值有利）为 **{full['mae_improvement']:+.3f}**；按来源bootstrap描述性95%区间 [{full['descriptive_CI95'][0]:+.3f},{full['descriptive_CI95'][1]:+.3f}]。本实验训练了小型reward probe，生成器零训练。",'', '## 全量组外预测','', '| 模型 | 主MAE | 符号准确率 | gap绝对值≥0.1准确率 | gap绝对值≥0.3准确率 |','|---|---:|---:|---:|---:|']
    for m in ['M0','M1','M2','M3']:
        s=full['summaries'][m];lines.append(f"| {m} | {s['primary']['mae']:.3f} | {s['primary']['accuracy']:.3f} | {s['threshold_.1']['accuracy']:.3f} | {s['threshold_.3']['accuracy']:.3f} |")
    lines+=['','M0=仅arm身份的训练折估计比较器；M1=arm+六声学特征+log1p官方有效距离行数；M2=arm+HuBERT PCA8+log时长+log1p支持；M3=M1+HuBERT PCA8。主比较事先指定M3对M1。主MAE按来源内三个family等权、再按20来源等权。', '', '## 三个主比较及同模型重复','', '| Family | pair数 | M0 MAE | M1 MAE | M2 MAE | M3 MAE | M1准确率 | M3准确率 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for f in ['N-L06','N-L17','L17-L06','L06-within','L17-within']:
        ss=[full['summaries'][m]['families'][f] for m in ['M0','M1','M2','M3']]
        lines.append(f"| {f} | {ss[0]['pairs']} | "+' | '.join(f"{s['mae']:.3f}" for s in ss)+f" | {ss[1]['accuracy']:.3f} | {ss[3]['accuracy']:.3f} |")
    lines+=['','## 队列与共同资格敏感性','', '| 支持集 | M1 MAE | M3 MAE | 改善M1−M3 |','|---|---:|---:|---:|']
    for c in ['old','new']:
        a=full['summaries']['M1']['cohorts'][c]['mae'];b=full['summaries']['M3']['cohorts'][c]['mae'];lines.append(f'| {c}10来源 | {a:.3f} | {b:.3f} | {a-b:+.3f} |')
    a=strict['summaries']['M1']['primary']['mae'];b=strict['summaries']['M3']['primary']['mae'];lines.append(f'| 共同资格20来源 | {a:.3f} | {b:.3f} | {a-b:+.3f} |')
    lines+=['',f"全量保留300pair（其中主比较180），共同资格为{strict['pair_count']}pair。已知S0011/S0016的L06 draw3无EOS：敏感性同时去两arm同draw，保留N及全部20来源，再完整nested重拟合。没有根据分差筛选。",f"主集gap绝对值≥0.1准确率改变={gate['accuracy_improvement']:+.3f}，old/new的MAE改善均为负，资格改善也为负；四项推进门均未通过。按协议停止本轮复杂模型/SAE，无层/模型/超参扩搜索。",'', '## 逐来源','', '| 来源 | M1 MAE | M3 MAE | 改善 |','|---|---:|---:|---:|']
    ids=sorted(full['summaries']['M1']['primary']['sources']);vals=[]
    for sid in ids:
        a=full['summaries']['M1']['primary']['sources'][sid]['mae'];b=full['summaries']['M3']['primary']['sources'][sid]['mae'];vals.append(a-b);lines.append(f'| {sid} | {a:.3f} | {b:.3f} | {a-b:+.3f} |')
    lines+=['', '![来源级MAE改善](source_mae_improvement_attempt02.png)','', '## Material Passport','',f"- **材料**：已有N20、L06/L17各60，共140唯一canonical WAV；20来源/20speaker，去空格文本20种，无跨来源同文本或speaker重合。N明确来自role N，不是codec R。每speaker/text连通组留一，全部draw同折；没有新的确认集。",f"- **音量及格式**：冻结目标RMS={im['normalization_target']}，实测[{im['actual_rms_range'][0]:.10f},{im['actual_rms_range'][1]:.10f}]；全为mono16k PCM16；全长原canonical不对齐时长、不剪尾，两条长尾均保留。",f"- **标签**：每clip首track官方float32的31lag数组恢复C/D/B，C=median(mean distance)−min(mean distance)；与stdout三位小数桥接最大{im['bridge_max_error']:.10f}<0.000501。输入音频、视频、score与距离数组SHA重新核验。距离/偏移/C/D/B均不作预测输入，预声明支持行数除外。", '- **冻结表示**：现有本地HuBERT base-ls960最终隐藏层，eval/no_grad FP32，自己的音频标准化如encoder_metadata.json；attention mask真实样本数及卷积有效帧数逐条核验；时间mean/std池化（1536维），没有截尾或层搜索。PCA8是训练折内表示压缩，并非音素概念或可解释特征。', '- **拟合**：每outer留一组；inner GroupKFold5在训练unique clips重新fit标准化/PCA，full deterministic SVD、CPU单线程；alpha仅1/10/100，最小主family/source宏MAE，平局强正则。Ridge无intercept，预测r(A)−r(B)。每source/family总权重1，三pair各1/3；自然音频没有复制成独立样本。', '- **模型产物**：140×1536 pooled、逐clip真实样本/有效frame数，40个fold的模型/处理器pickle与独立数字NPZ（train indices、scaler、PCA、ridge coef/alpha），两支持集OOF逐pair表与全部指标。源顺序和特征顺序在environment_snapshot.json。', '- **测试与记录**：pair设计、共同资格、手算宏指标、零预测准确率、反对称性、预测输入无target；拟合中逐fold训练/测试来源不交及交换AB变号；首foldheldout hidden变更不影响训练fit的scaler/PCA及训练transform。3个pytest兼容测试直接断言执行通过（环境无pytest，不安装）；所有执行阶段的开始/PID/命令/代码SHA均记录。', '- **工程状态**：原audit/extract成功；完整科学拟合/OOF/40fold保存成功后，原analyze包装因缺matplotlib退出1，原失败记录保留。此report-only attempt02使用已有PIL，无重新提取/拟合，所有科学输入SHA在receipt固定并结束重验。', '', '## 解释边界','', 'native各自(audio, generated video)评分，因此audio-only probe预测固定现有TFG链下的原生SyncNet排序；不能隔离评价器纯偏好与生成端响应，不能判断人类优劣、嘴型质量或因果机制。', '全部20来源先前已查看，本实验属于探索性组外预测。来源bootstrap有共享训练折，只给描述性区间，不是独立确认推断。未过门只限制这套固定HuBERT/PCA/线性探针的增量预测价值，不否定可能存在其它机制。']
    report='\n'.join(lines)+'\n';(R/'report.md').write_text(report);(dst/'report.md').write_text(report)
    # Simple static chart in English: axis labels render with existing fonts.
    img=Image.new('RGB',(1200,520),'white');d=ImageDraw.Draw(img);d.text((30,15),'Exploratory OOF MAE improvement: M1 - M3 (positive favors HuBERT)',fill='black')
    x0,x1,y0,y1=75,1175,60,410;span=max(abs(min(vals)),abs(max(vals)))*1.2;zero=(y0+y1)/2
    d.line((x0,zero,x1,zero),fill='black',width=2)
    for k in range(-2,3):
        v=k*span/2;y=zero-v/span*(y1-y0)/2;d.text((10,y-5),f'{v:+.2f}',fill='black')
    for i,(sid,v) in enumerate(zip(ids,vals)):
        x=x0+i*(x1-x0)/20;y=zero-v/span*(y1-y0)/2;d.rectangle((x+5,min(y,zero),x+40,max(y,zero)),fill='#247fba' if v>=0 else '#bd5638');d.text((x+5,y1+10),sid[6:11],fill='black')
    img.save(R/'source_mae_improvement_attempt02.png')
    assert all(sha(R/n)==h for n,h in pins.items())
    receipt.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.time()-started,exitcode=0,scientific_pins_unchanged=True)
    (dst/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
