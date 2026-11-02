"""Independent from-scratch repeat; byte-identical RMS inputs and original numerical policy."""
import json
from pathlib import Path
from flow_stream import sha
from patrick_rms_runtime_v1 import install as install_parent, RUN as PARENT
RUN=Path('/dev/shm/synctalk-multispeaker-data-20261017/trials/patrick_ave_rms_coarse_repeat_v1')
def install():
    contract=json.loads((RUN/'contract.json').read_text())
    for name,digest in contract['sources'].items():assert sha(RUN/'sources'/name)==digest
    parent=install_parent()
    assert sha(PARENT/'contract.json')==contract['parent_normalization_contract_sha256']
    assert contract['normalization']==parent['normalization']
    assert contract['initial_checkpoint_sha256']==parent['initial_checkpoint_sha256']
    assert contract['initial_model_tensor_sha256']==parent['initial_model_tensor_sha256']
    assert not contract['density_cache_fallback'] and not contract['failed_state_warmstart']
    return contract
