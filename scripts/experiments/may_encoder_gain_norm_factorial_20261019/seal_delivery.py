"""Seal actual cross-volume members and exact imported report54 science, without deletion."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,subprocess
from protocol import OUT,REMOTE,sha,write
def main():
 p=json.loads((OUT/'protocol.json').read_text());parent=Path(p['parent'])
 for path in [OUT/'analysis/qa.json',OUT/'analysis/visual_review.json',OUT/'analysis/report_interpretation_readback.json',OUT/'training_independent_review.json',OUT/'final_memory_publication_receipt.json',OUT/'final_semantic_publication_receipt.json',OUT/'entry_documents_final_readback.json',OUT/'media_clock_review.json']:assert json.loads(path.read_text())['passed']
 for row in json.loads((OUT/'analysis/visual_review.json').read_text())['images']:assert sha(OUT/'analysis'/row['file'])==row['sha256']
 assert sha(OUT/'analysis/report.md')==json.loads((OUT/'analysis/report_interpretation_readback.json').read_text())['report_sha256']
 pins=json.loads((OUT/'execution_source_pins.json').read_text())
 for name in ['train.py','runtime.py','evaluate.py']:assert sha(Path(__file__).parent/name)==pins['source_sha256'][name]
 assert sha(OUT/'protocol.json')=='4fdf3fa68595be8032839e1a5cfff1c5839b51c650440e619113055755e51518'
 assert json.loads((OUT/'analysis/qa.json').read_text())['total_matrices']==144
 oldseal=json.loads((parent/'final_delivery_receipt.json').read_text());assert sha(parent/'final_delivery_receipt.json')==p['parent_delivery_receipt_sha256']
 oldmanifest=json.loads((parent/'delivery_manifest.json').read_text());assert sha(parent/'delivery_manifest.json')==oldseal['manifest_sha256']
 for row in oldmanifest['members']:
  f=parent/row['path'];assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
 write(OUT/'final_imported_parent_readback.json',dict(passed=True,members=len(oldmanifest['members']),parent_root=str(parent),parent_manifest_sha256=sha(parent/'delivery_manifest.json'),new_scientific_jobs=0))
 code="""import subprocess,json
s=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
rows=[r for r in s.splitlines() if '/dev/shm/may-encoder-gain-norm-factorial-20261019/' in r and any(x in r for x in ['/train.py','/evaluate.py','job_wrapper.py'])]
print(json.dumps(dict(active_scientific_processes=rows)))
"""
 ssh=['ssh','-b','192.168.50.1','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','wjj@192.168.50.2'];r=subprocess.run(ssh+['python3','-'],input=code.encode(),capture_output=True,check=True);state=json.loads(r.stdout);assert not state['active_scientific_processes']
 paused=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_repeat_v1/user_pause_v1/paused.json');assert sha(paused)=='545c965703b38a42058fc5c3a65a4e662722470a306a37ca7ea9f97e1a8ec090'
 write(OUT/'postrun_process_review.json',dict(passed=True,**state,old_patrick_pause_sha256=sha(paused)))
 placement=OUT/'media_placement_handoff_v1.json'
 if placement.exists():
  handoff=json.loads(placement.read_text(encoding='utf-8-sig'));assert handoff['passed'] and not handoff['scientific_protocol_changed']
  logical=Path(handoff['logical_path']);physical=Path(handoff['physical_path']);assert logical.resolve()==physical.resolve() and physical.resolve().is_relative_to(Path('E:/Datasets/tts-exp/may_encoder_gain_norm_factorial_20261019_media').resolve())
  assert sha(logical/'f_readback_manifest.json')==sha(physical/'f_readback_manifest.json')
  write(OUT/'media_placement_final_readback.json',dict(passed=True,logical_path=str(logical),physical_path=str(physical),member_manifest_sha256=sha(physical/'f_readback_manifest.json'),placement_handoff_sha256=sha(placement)))
 capsule=OUT/'execution_sources_final';capsule.mkdir(exist_ok=False)
 for f in Path(__file__).parent.glob('*.py'):shutil.copyfile(f,capsule/f.name)
 roots={'main':OUT}
 for arm,folder in p['new_media_placements'].items():
  folder=Path(folder)
  if not folder.resolve().is_relative_to(OUT.resolve()):
   assert folder.resolve().is_relative_to(Path('E:/Datasets/tts-exp/may_encoder_gain_norm_factorial_20261019_media').resolve());roots['media_'+arm]=folder.resolve()
 rows=[]
 for key,root in roots.items():
  for f in sorted(root.rglob('*')):
   if not f.is_file() or 'matplotlib-cache' in f.parts or f.name in ['delivery_manifest.json','final_delivery_receipt.json']:continue
   if key=='main' and not f.resolve().is_relative_to(OUT.resolve()):continue
   rows.append(dict(root=key,path=f.relative_to(root).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
 write(OUT/'delivery_manifest.json',dict(roots={k:str(v) for k,v in roots.items()},members=rows,total_new_persistent_bytes=sum(x['bytes'] for x in rows),imports=dict(root=str(parent),manifest_sha256=sha(parent/'delivery_manifest.json'),members=len(oldmanifest['members'])),new_science_cloud_backup_claimed=False))
 for row in rows:
  f=roots[row['root']]/row['path'];assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
 write(OUT/'final_delivery_receipt.json',dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),roots=len(roots),members=len(rows),new_persistent_bytes=sum(x['bytes'] for x in rows),manifest_sha256=sha(OUT/'delivery_manifest.json'),report_sha256=sha(OUT/'analysis/report.md'),visual_review_sha256=sha(OUT/'analysis/visual_review.json'),memory_publication_sha256=sha(OUT/'final_memory_publication_receipt.json'),imported_parent_manifest_sha256=sha(parent/'delivery_manifest.json'),new_training_models=8,imported_training_models=4,new_updates=96000,total_updates_including_imports=144000,cells=72,total_score_matrices=144,new_score_matrices=96,old_zero_gain_replayed=False,no_active_scientific_jobs=True,old_patrick_still_paused=True,single_seed_pilot=True,perceptual_claim=False,new_science_cloud_backup_claimed=False))
 print(json.dumps(json.loads((OUT/'final_delivery_receipt.json').read_text())))
if __name__=='__main__':main()
