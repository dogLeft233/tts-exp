"""Versioned dataset selector: scalar-RMS AVE, unchanged original trainer/math."""
import json
from pathlib import Path
from flow_stream import sha
import runtime_common as rc
RUN=rc.ROOT/'trials/patrick_ave_rms_coarse_v1'
def install():
    contract=json.loads((RUN/'contract.json').read_text())
    for name,digest in contract['sources'].items():assert sha(RUN/'sources'/name)==digest,name
    feature=RUN/'features/PatrickLeahy/ave_rms_qualification.json'
    proof=json.loads(feature.read_text());assert proof['passed'] and proof['contract_sha256']==sha(RUN/'contract.json')
    original=rc.options
    def options(identity,split,gain,seed,phase='coarse',preload=0,data_range=None):
        assert identity=='PatrickLeahy' and gain in [-6,0,6] and seed==42 and phase=='coarse'
        opt=original(identity,split,gain,seed,phase,preload,data_range)
        new=RUN/'datasets/PatrickLeahy'/split/{-6:'gain_m6',0:'gain_0',6:'gain_p6'}[gain]
        b=json.loads((new/'binding.json').read_text());old=Path(opt.path)
        assert b['parent_binding_sha256']==sha(old/'binding.json')
        assert b['normalization_contract_sha256']==sha(RUN/'contract.json')
        assert sha(new/'whole_ave.npy')==b['whole_ave_sha256']
        assert b['gain_db']==gain and b['split']==split
        opt.path=str(new);opt.verified_ave_cache=str(new/'whole_ave.npy')
        return opt
    rc.options=options
    return contract
