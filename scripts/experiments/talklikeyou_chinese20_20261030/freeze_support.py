from pathlib import Path
import json
r=Path('/home/data2/wangjiajun/av-evaluation/jobs/talklikeyou_chinese20_rms24_20261030_attempt01')
rows=json.loads((r/'inputs.json').read_text())['rows']
lookup={x['source']:[y for y in rows if y['source']==x['source']] for x in rows}
eligible=[x for x in rows if min(x['samples']//640,x['expected_frames'])>=125]
sources=[s for s,rs in lookup.items() if all(x in eligible for x in rs)]
full=[s for s,rs in lookup.items() if all(x['samples']>=76000 and x['expected_frames']>=119 for x in rs)]
x=dict(status='PREDICTED_FROM_FROZEN_INPUTS_BEFORE_SCORING',classifier_eligible_media=len(eligible),classifier_complete_sources=sources,classifier_primary_sources=[s for s in sources if lookup[s][0]['strict_reference']],syncreward_full_native_sources=full,policy='Final score eligibility also checks actually decoded video frame counts. No selection by results.')
p=r/'predeclared_support.json';assert not p.exists();p.write_text(json.dumps(x,indent=2));print(json.dumps(x))
