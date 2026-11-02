"""Source/origin/codebook summaries of only the frozen descriptive endpoints."""
from pathlib import Path
import argparse,json,statistics

def avg(values):
    return '+inf' if any(isinstance(v,str) for v in values) else statistics.fmean(values)

def main(run):
    load=lambda f:json.loads(f.read_text())
    p=load(run/'B_protocol.json');diag=load(run/'B_history_diagnostics.json')['statistics'];metrics={(d['reader'],d['id']):d for d in diag if 'reader' in d};compare={d['id']:d['comparison'] for d in diag if 'comparison' in d}
    numerical=['raw_entropy','effective_content_entropy','raw_NLL','effective_NLL','zero_policy_probability_fraction','raw_target_rank'];source=[];balanced=[];comparisons=[]
    for sid in sorted({r['sample_id'] for r in p['rows']}):
        byreader={}
        for model in ['L06','L17']:
            origins=[]
            for origin in ['N','L06','L17']:
                ids=[r['id'] for r in p['rows'] if r['sample_id']==sid and r['source']==origin and r['qualified']]
                records=[metrics[model,key] for key in ids]
                books=[{'cb':cb,**{metric:avg([d['codebooks'][cb][metric] for d in records]) for metric in numerical}} for cb in range(16)]
                row={'sample_id':sid,'reader':model,'origin':origin,'qualified_histories':ids,'codebooks':books,'mean_EOS_probability_content_frames':avg([d['EOS_probability_content_frames_mean'] for d in records]),'mean_total_effective_coarse_entropy':avg([d['total_effective_coarse_entropy_content_frames'] for d in records]),'terminal_EOS_probabilities':[{'history':key,'probability':metrics[model,key]['EOS_probability_terminal'],'observed_target_EOS':metrics[model,key]['terminal_observed']} for key in ids]}
                source.append(row);origins.append(row)
            books=[{'cb':cb,**{metric:avg([origin['codebooks'][cb][metric] for origin in origins]) for metric in numerical}} for cb in range(16)]
            row={'sample_id':sid,'reader':model,'codebooks':books,'coarse':{metric:books[0][metric] for metric in numerical},'fine_equal_codebook_mean':{metric:avg([b[metric] for b in books[1:]]) for metric in numerical}}
            balanced.append(row);byreader[model]=row
        origin_comparisons=[]
        for origin in ['N','L06','L17']:
            ids=[r['id'] for r in p['rows'] if r['sample_id']==sid and r['source']==origin and r['qualified']]
            keys=['raw_JSD','effective_JSD','raw_top1_disagreement','effective_top1_disagreement']
            books=[{'cb':cb,**{key:avg([compare[i][cb][key] for i in ids]) for key in keys}} for cb in range(16)]
            origin_comparisons.append({'origin':origin,'codebooks':books})
        books=[{'cb':cb,**{key:avg([o['codebooks'][cb][key] for o in origin_comparisons]) for key in keys}} for cb in range(16)]
        comparisons.append({'sample_id':sid,'origins':origin_comparisons,'origin_balanced_codebooks':books,'coarse':{key:books[0][key] for key in keys},'fine_equal_codebook_mean':{key:avg([b[key] for b in books[1:]]) for key in keys}})
    out={'status':'DESCRIPTIVE_ONLY','weighting':'source equal; N/T06/T17 equal; T draws first averaged within origin; frames equal within history; fine codebooks equal','source_origin_codebooks':source,'source_balanced_reader_diagnostics':balanced,'source_pair_distribution_diagnostics':comparisons,'N_warning':'N is same-utterance reference in full ICL, not heldout-language correctness','policy_NLL_warning':'+inf means at least one observed target is outside native truncated support; zero-support fractions are also reported','no_additional_inferential_gates':True}
    with (run/'B_source_diagnostics.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
