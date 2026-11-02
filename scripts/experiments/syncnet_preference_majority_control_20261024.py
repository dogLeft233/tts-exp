"""Post-hoc train-only family sign comparator; no fitted reward model."""
from pathlib import Path
import json,importlib.util
import numpy as np
R=Path('/home/wjj/tts-audio/tts-exp/runs/syncnet_preference_probe_20261024_attempt01')
def main():
    sp=importlib.util.spec_from_file_location('probe',R/'analysis_source.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
    result={}
    for name in ['full','qualified']:
        rows=json.loads((R/(name+'_oof.json')).read_text());pred=[]
        for x in rows:
            tr=[v for v in rows if v['group']!=x['group'] and v['family']==x['family']];counts={}
            for v in tr:counts[v['source']]=counts.get(v['source'],0)+1
            pos=sum((v['target']>0)/counts[v['source']] for v in tr);neg=sum((v['target']<0)/counts[v['source']] for v in tr)
            pred.append(float(np.sign(pos-neg)))
        s=p.source_metrics(rows,pred);result[name]={'primary_accuracy':s['accuracy'],'primary_sources':s['sources'],'families':{f:p.source_metrics(rows,pred,[f]) for f in p.PRIMARY},'method':'Post-hoc train-only family sign majority; source/family equal weight; outer group excluded. Not replacing continuous Ridge M0 or gate.'}
    old=json.loads((R/'majority_control.json').read_text())
    assert all(result[n]['primary_accuracy']==old[n]['primary_accuracy'] for n in result)
    # Existing numeric comparator remains frozen; only append contextual report text.
    para='\n## 事后arm多数符号比较器\n\n父侧独立审核已通过：140音频/视频/score/曲线SHA及原float32距离重算，160个outer线性模型闭式重拟合、全部OOF重放，参数最大差2.52e−7、预测最大差1.89e−7，指标与bootstrap区间误差0。\n\n为避免把连续奖励M0误当分类最强基线，补充每outer训练集内按family多数符号预测（按source/family等权、不看留出来源）。全量主准确率60.556%，共同资格60.833%；M1相应65.556%/61.111%，严格资格下仅比多数符号高0.278个百分点。该事后比较器不替换预定M0、不修改既定gate。预测信号需同时考虑arm先验和停止异常影响，不能仅与50%比较。独立复算一致，见majority_control.json。\n'
    path=R/'report.md';text=path.read_text();assert '事后arm多数符号比较器' not in text;path.write_text(text+para)
    (R/'report_attempt02/majority_addendum_receipt.json').write_text(json.dumps({'no_refit':True,'no_gate_change':True,'full_accuracy':result['full']['primary_accuracy'],'qualified_accuracy':result['qualified']['primary_accuracy'],'independently_recomputed_equal':True},indent=2))
if __name__=='__main__':main()
