"""Shape/serialization budget only; reads no waveform, image, feature or score payload."""
from pathlib import Path
import ctypes
import ctypes.util
import hashlib
import json
import math
import shutil
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/grid_shape_correlation_evaluation_20260927'
OUT=ROOT/'runs/grid_shape_eval_audit_20260927/resource_review'
K=1024; M=K*K
roundblock=lambda n: math.ceil(n/4096)*4096
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
encoded=lambda x:len((json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
FLOAT=-1.7976931348623157e308
INT=18446744073709551615
H='f'*64
views=['BASE','TX_-8','TX_8','TY_-8','TY_8','ROT_-5','ROT_5','SCALE_0.9','SCALE_1.1']
# Longest finite binary64 serialization (24 bytes) and deliberately excessive
# uint64 integer representations. All actual bounded fields fit these tokens.
qc={'valid_fraction':FLOAT,'missing_run':INT,'reference_angles_deg':[FLOAT]*3,
    'pose_change_p95_deg':[FLOAT]*2,'stable_residual_median':FLOAT,
    'stable_residual_p95':FLOAT,'aperture_sd':FLOAT,
    'checks':{k:False for k in ['coverage','missing','reference_pose','relative_pose','similarity','motion']},'passed':False}
landmark={'pixel_hashes':[H]*75,'full_landmark_hashes':[H]*75,
          'face_counts':[2]*75,'invalid_reasons':['not_exactly_one_face_and_matrix']*75,
          'id':'s26','domain':'FIXED','view':'SCALE_1.1','qc':qc,
          'box_from_first_frame':[INT]*4,'npz_sha256':H,'protocol_sha256':H,
          'resources':{k:INT for k in ['allocated_bytes','free_bytes','tmp_bytes','peak_rss_bytes','child_rss_bytes','memavailable_bytes']}}
assert encoded(landmark)<20*K
stat={'n':16,'mean':FLOAT,'ci95':[FLOAT]*2,'ci99':[FLOAT]*2,'computable':False}
domain={'BASE_QC_pass':False,'passed':False,'native_sd':FLOAT,'max_spatial_centered_rmse':FLOAT,
        'spatial':{v:{'r':FLOAT,'centered_rmse':FLOAT} for v in views[1:]}}
corr={a+'/'+b:{'RAW':FLOAT,'FIXED':FLOAT,'delta':FLOAT} for a in views for b in views}
record={'id':'s26','indices':list(range(14,55)),'support_pass':False,
        'domains':{d:domain for d in ['REAL','RAW','FIXED']},
        'correlations':{'0':corr,'-3':corr},'all_required_computable':False,'measurement_pass':False}
geometry={'rows':[record]*16,'eligible_denominator':16,'fixed_input_denominator':16,
          'measurement_passed_sources':16,'measurement_required':16,'measurement_gate_pass':False,
          'all_required_computable':False,'summary':{str(l):{a+'/'+b:stat for a in views for b in views} for l in [0,-3]},
          'primary':'lag0 BASE/BASE','robustness_conclusion':'ROBUST_IMPROVEMENT',
          'bootstrap_draws':100000,'bootstrap_indices_hash':H,'seed':20260927,
          'no_output_QC_sample_removal':False,'FWER_simultaneous_coverage_claim':False}
assert encoded(geometry)<2*M
cell={'C':FLOAT,'D':FLOAT,'offset':INT,'best_lag':INT,'curve':[FLOAT]*31,'query_start':20,'query_stop':49}
sync_result={'rows':[{'id':'s26','cells':{'RAW':cell,'FIXED':cell},'deltas':{k:FLOAT for k in ['C','D','offset']}}]*16,
             'eligible_denominator':16,'summary':{k:stat for k in ['C','D','offset']},
             'support':'same eligible labels, different temporal windows from geometry','primary_rescue_forbidden':False}
assert encoded(sync_result)<128*K
# The four NPY payloads have fixed simple-dtype headers <=128 bytes each.
# ZIP uses default zlib compression with raw DEFLATE; compressBound for the same
# default parameters includes its wrapper and is an upper bound. 4096 additional
# bytes cover the four ZIP local/central/ZIP64 headers and filenames.
libz=ctypes.CDLL(ctypes.util.find_library('z'))
libz.compressBound.argtypes=[ctypes.c_ulong];libz.compressBound.restype=ctypes.c_ulong
payloads=[75*13*3*4,75,75*4*4*8,75*8]
npz_bound=sum(libz.compressBound(n+128) for n in payloads)+4096
assert npz_bound<32*K

# Only stat paths: do not open any observed scientific data or QC values.
inventory=[]
for f in sorted(RUN.rglob('*')):
    if f.is_file():
        s=f.stat();inventory.append([str(f.relative_to(RUN)),s.st_size,s.st_blocks*512])
paths={r[0] for r in inventory}
completed=[x for x in paths if x.startswith('landmarks/') and x.endswith('.json') and x[:-5]+'.npz' in paths]
counts={'views':len(completed),'A':sum(x.startswith('sync/') and x.endswith('/A.npy') for x in paths),
        'V':sum(x.startswith('sync/') and x.endswith('_V.npy') for x in paths),
        'generation':sum(x.startswith('generation/') and x.endswith('.json') for x in paths),
        'syncmeta':sum(x.startswith('sync/') and x.endswith('/metadata.json') for x in paths),
        'retention':sum(x.startswith('retention/') and x.endswith('.json') for x in paths)}
allocated=sum(x[2] for x in inventory)
remaining={'landmarks':(432-counts['views'])*(32+20)*K,
           'A':(16-counts['A'])*roundblock(69*1024*4+128),
           'V':(32-counts['V'])*roundblock(71*1024*4+128),
           'generation':(32-counts['generation'])*16*K,
           'syncmeta':(16-counts['syncmeta'])*20*K,
           'retention':(32-counts['retention'])*4*K,
           'distances':32*roundblock(69*31*8+128),
           'geometry_result':2*M,'sync_result':128*K,'score_lock':512*K,
           'future_nonscientific_enforced_cap':1*M}
assert all(v>=0 for v in remaining.values())
bound=allocated+sum(remaining.values())
assert bound<=37*M,(bound/ M,counts)
OUT.mkdir(exist_ok=True)
inventory_path=OUT/'stable_size_inventory.json'
inventory_path.write_text(json.dumps(inventory,separators=(',',':'))+'\n')
receipt={'status':'PASS_CONDITIONAL_ENGINEERING_ENFORCEMENT','time':time.time(),
         'protocol_sha256':sha(RUN/'protocol.json'),'worker_sha256':sha(ROOT/'scripts/experiments/grid_shape_correlation_evaluation_20260927.py'),
         'auditor_sha256':sha(__file__),'inventory_sha256':sha(inventory_path),
         'counts':counts,'allocated_bytes':allocated,'remaining_upper_bounds':remaining,
         'final_upper_bound_bytes':bound,'final_upper_bound_MiB':bound/M,'approved_cap_bytes':37*M,
         'serialization_worst_case_bytes':{'landmark_json':encoded(landmark),'geometry_result_json':encoded(geometry),
                                            'sync_result_json':encoded(sync_result),'landmark_npz':npz_bound},
         'future_nonscientific_limit_bytes':M,'other_reserved_bytes_unchanged':8*M,
         'disk_floor_bytes_unchanged':5*2**30,'free_bytes':shutil.disk_usage(RUN).free,
         'scope':'size-only live inventory and synthetic worst-case serialization; no effects or model',
         'conditions':['snapshot taken after controls block2 exit and before next stage',
                       'future nonscientific allocated increment <=1MiB enforced including logs, code/protocol versions, runtime, report/final; fail closed',
                       'scientific arrays/results are complete and never truncated',
                       'all listed per-product allocated upper bounds checked before persistence',
                       '37MiB run cap, 8MiB other reserve, 5GiB floor stay enforced; no old deletion'],
         'not_an_unconditional_external_disk_or_log_growth_guarantee':True}
(OUT/'budget_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
