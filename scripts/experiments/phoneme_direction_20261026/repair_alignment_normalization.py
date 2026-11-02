"""Fix MFA orthographic alias coverage, retaining all original alignment evidence."""
import os
import subprocess
from pathlib import Path
from calibrate import ROOT,MODELS,MFA,read,write,sha
from repair_dictionary import parse_dictionary

RUN=ROOT/'runs/phoneme_direction_20261026_repair01'


def main():
    prior=RUN/'mandarin_china_mfa_fit_overlay.dict';text=prior.read_text();entries=parse_dictionary(text)
    if '綠' in entries:raise ValueError('Unexpected existing alias')
    # MFA's normalization produced 綠 although source LAB uses 绿. Carry every
    # existing pronunciation and probability from the exact same character.
    additions=['\t'.join(['綠',*numeric,' '.join(phones)]) for numeric,phones,_ in entries['绿']]
    dictionary=RUN/'mandarin_china_mfa_fit_overlay_revision02.dict'
    with dictionary.open('x',encoding='utf8') as f:f.write(text.rstrip()+'\n'+'\n'.join(additions)+'\n')
    acoustic=MODELS/'acoustic/mandarin_mfa.zip'
    write(RUN/'mfa_normalization_repair.json',{'prior_dictionary_sha256':sha(prior),'dictionary':str(dictionary),
          'dictionary_sha256':sha(dictionary),'alias':'綠 copies all original 绿 pronunciations/probabilities',
          'source_transcripts_changed':False,'acoustic_model_changed':False,'script_sha256':sha(__file__),
          'finding':'MFA database normalized_text contains 綠/為, though input LAB contains 绿/为',
          'separate_parser_finding':'Two 40.88s TTS outputs have a duplicate final word interval whose sole database phone is sil; no media is removed or retimed'})
    env=dict(os.environ,PATH=str(MFA.parent)+os.pathsep+os.environ.get('PATH',''),CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MFA_ROOT_DIR=str(RUN/'mfa_runtime_fit_revision02'),TMPDIR=str(RUN/'tmp'))
    command=[str(MFA),'align',str(RUN/'mfa_corpus_fit'),str(dictionary),str(acoustic),str(RUN/'mfa_grids_fit_revision02'),
             '--clean','--num_jobs','2']
    write(RUN/'mfa_launch_fit_revision02.json',{'command':command,'dictionary_sha256':sha(dictionary),'acoustic_sha256':sha(acoustic)})
    with (RUN/'mfa_fit_revision02.log').open('x') as f:
        p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=14400)
    write(RUN/'mfa_exit_fit_revision02.json',{'returncode':p.returncode})
    if p.returncode:raise RuntimeError('MFA failed')


if __name__=='__main__':main()
