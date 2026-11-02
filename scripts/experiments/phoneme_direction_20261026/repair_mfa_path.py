"""Repair missing conda executable PATH; preserve the failed first MFA attempt."""
import os
from pathlib import Path
import subprocess
from calibrate import MFA, MODELS, RUN, read, sha, write


def main():
    previous=read(RUN/'mfa_exit.json')
    if previous['returncode']==0:
        raise ValueError('Repair only applies to failed launch')
    log_text=(RUN/'mfa.log').read_text()
    if "Could not find 'fstcompile'" not in log_text:
        raise ValueError('Unexpected first failure; inspect before retry')
    if not (MFA.parent/'fstcompile').is_file():
        raise FileNotFoundError('OpenFST missing; PATH repair insufficient')
    dictionary=MODELS/'dictionary/mandarin_china_mfa.dict'
    acoustic=MODELS/'acoustic/mandarin_mfa.zip'
    for path,expected in read(RUN/'mfa_launch.json')['model_pins'].items():
        if sha(path)!=expected:raise ValueError('Model changed')
    env=dict(os.environ,PATH=str(MFA.parent)+os.pathsep+os.environ.get('PATH',''),
             CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             MFA_ROOT_DIR=str(RUN/'mfa_runtime_attempt02'),TMPDIR=str(RUN/'tmp'))
    command=[str(MFA),'align',str(RUN/'mfa_corpus'),str(dictionary),str(acoustic),str(RUN/'mfa_grids'),
             '--clean','--num_jobs','2']
    write(RUN/'mfa_launch_attempt02.json',{'command':command,'repair':'Add existing conda bin to process PATH; no model, data, or quality gate change',
        'previous_log_sha256':sha(RUN/'mfa.log'),'source_sha256':sha(__file__),
        'path_prefix':str(MFA.parent),'fstcompile_sha256':sha(MFA.parent/'fstcompile')})
    with (RUN/'mfa_attempt02.log').open('x') as f:
        result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=7200)
    write(RUN/'mfa_exit_attempt02.json',{'returncode':result.returncode})
    if result.returncode:raise RuntimeError('Repaired MFA launch failed; preserve evidence')


if __name__=='__main__':main()
