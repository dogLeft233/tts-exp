"""CPU input/text/seed/reference bindings over all frozen bank histories."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import read,sha,save

def main(run):
    p=read(run/'B_protocol.json');samples={s['sample_id']:s for s in p['samples']};checked=[]
    for row in p['rows']:
        s=samples[row['sample_id']];text=''.join(s['transcript'].split())
        assert sha(row['codes']['path'])==row['codes']['sha256']
        if row['source']=='N':continue
        r=read(row['record']);assert r['target_text']==text and r['reference_text']==text
        draw=r.get('draw_index',r.get('seed_index'));assert draw==row['draw'] and r['seed']==row['seed']
        expected=2026102200+row['draw']+100*int(s['speaker_id'][1:]);assert r['seed']==expected
        n=next(v for v in p['rows'] if v['sample_id']==s['sample_id'] and v['source']=='N')
        assert r['reference_codes']['sha256']==n['codes']['sha256']
        kwargs=r['talker_effective_kwargs'];assert kwargs['temperature']==.9 and kwargs['subtalker_temperature']==.9
        assert kwargs['top_k']==kwargs['subtalker_top_k']==50 and kwargs['top_p']==kwargs['subtalker_top_p']==1.
        assert kwargs['repetition_penalty']==1.05 and kwargs['eos_token_id']==2150
        checked.append(row['id'])
    assert len(checked)==60
    save(run/'B_input_binding_audit.json',{'status':'PASS','verified_generated_histories':len(checked),'natural_histories':10,'same_text_reference_codes_and_existing_seeds':True,'native_policy_exact':True,'ids':checked,'B_protocol_sha':sha(run/'B_protocol.json')})

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
