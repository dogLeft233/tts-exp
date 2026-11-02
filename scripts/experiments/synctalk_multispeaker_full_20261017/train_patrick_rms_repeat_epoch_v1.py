"""Explicit new-run entry; delegate numerical work to byte-pinned original coarse source."""
import argparse,json,sys
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017');sys.path.insert(0,str(ROOT))
from patrick_rms_repeat_runtime_v1 import install,RUN
from flow_stream import sha
def run(a):
    contract=install()
    q=json.loads((RUN/'qualification/result.json').read_text())
    assert q['passed'] and q['contract_sha256']==sha(RUN/'contract.json')
    assert a.identity=='PatrickLeahy' and a.seed==42 and a.phase=='coarse' and a.epoch in range(1,11) and a.attempt==1
    import train_other_identity_epoch_v1 as original
    original.RUN=RUN
    original.run(a.identity,a.seed,a.gain,a.phase,a.epoch,a.attempt,a.checkpoint,a.resume,a.reference_trace)
    path=RUN/'training/PatrickLeahy/seed42'/f'gain_{a.gain:+d}'/'coarse'/f'ep{a.epoch:04d}/attempt01/epoch_receipt.json'
    # Add exact version-specific input provenance before job completion/persistence.
    r=json.loads(path.read_text());r['normalization_contract_sha256']=sha(RUN/'contract.json')
    r['normalized_binding_sha256']=sha(RUN/'datasets/PatrickLeahy/train'/{-6:'gain_m6',0:'gain_0',6:'gain_p6'}[a.gain]/'binding.json')
    r['new_independent_initialization']=a.epoch==1;r['old_fine_resumed']=False;r['independent_repeat']=True;r['input_normalization_parent_contract_sha256']=contract['parent_normalization_contract_sha256']
    path.write_text(json.dumps(r,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--gain',type=int,required=True)
    p.add_argument('--phase',required=True);p.add_argument('--epoch',type=int,required=True);p.add_argument('--attempt',type=int,default=1)
    p.add_argument('--checkpoint');p.add_argument('--resume');p.add_argument('--reference-trace');run(p.parse_args())
