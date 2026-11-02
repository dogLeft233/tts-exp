"""Finish the actual new27 CPU review/report; leave visual and BM finalization explicit."""
import json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_v1')
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def run():
 receipt=OUT/'delivery_handoff_receipt.json';assert not receipt.exists()
 state=dict(pid=os.getpid(),status='waiting_for_actual_new27',passed=False)
 write(receipt,state)
 while not (OUT/'score_27_v1/all27_preview.json').exists():
  assert not (OUT/'controller_failure.json').exists(),'training failed; preserve current state'
  assert not (OUT/'score_handoff_failure.json').exists(),'score handoff failed; preserve current state'
  if (OUT/'score_27_v1/controller_receipt.json').exists():assert json.loads((OUT/'score_27_v1/controller_receipt.json').read_text())['status']!='failed'
  time.sleep(30)
 summary=json.loads((OUT/'score_27_v1/all27_preview.json').read_text());assert summary['passed'] and len(summary['cells'])==27
 for script,stage in [('final_review_patrick_rms_v1.py','independent_162_matrix_review'),('plot_patrick_rms_v1.py','plot_actual_grid')]:
  state['status']=stage;write(receipt,state)
  with (OUT/(script+'.stdout.log')).open('xb') as log,(OUT/(script+'.stderr.log')).open('xb') as err:
   subprocess.run([sys.executable,str(HERE/script)],stdout=log,stderr=err,check=True)
 d=OUT/'score_27_v1/delivery_v1/reproduction_sources';d.mkdir()
 for p in HERE.glob('*.py'):shutil.copyfile(p,d/p.name)
 state['status']='report_and_comparison_from_actual_scores';write(receipt,state)
 with (OUT/'report_patrick_rms_v1.stdout.log').open('xb') as log,(OUT/'report_patrick_rms_v1.stderr.log').open('xb') as err:
  subprocess.run([sys.executable,str(HERE/'report_patrick_rms_v1.py')],stdout=log,stderr=err,check=True)
 candidate=json.loads((OUT/'score_27_v1/delivery_v1/delivery_candidate_receipt.json').read_text());assert candidate['passed']
 state.update(status='numerical_delivery_ready_pending_visual_and_memory',passed=True,visual_review_pending=True,memory_publication_pending=True);write(receipt,state)
if __name__=='__main__':
 try:run()
 except BaseException:write(OUT/'delivery_handoff_failure.json',dict(pid=os.getpid(),no_automatic_replay=True,traceback=traceback.format_exc()));raise
