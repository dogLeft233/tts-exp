"""Correct inherited labels only; preserve original runtime metadata and all science."""
import json
import shutil
from prepare import RUN,B,sha,write

def main():
    assert not (RUN/'stages/cogenav_complete.json').exists()
    old_protocol=json.loads((RUN/'protocol.json').read_text())
    old_inputs=json.loads((RUN/'inputs.json').read_text())
    backup=RUN/'inherited_metadata_original';backup.mkdir()
    for name in ['protocol.json','inputs.json']:
        shutil.copyfile(RUN/name,backup/name)
    science={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and str(p.relative_to(RUN)).startswith(('audio/','images/','videos/','generation/'))}
    current=dict(old_protocol)
    for name in ['parent_manifest_sha256','parent_run','parent_protocol_sha256']:current.pop(name,None)
    current.update(selected_dataset='chinese_lips_test21_20261031',sources=21,rows=42,arms=dict(N=21,TTS=21),unique_audio_files=42,
        template_checkpoint=str(B/'jobs/pctalk_chinese20_rms24_20261031_attempt01'),
        peavs='All42 lexically sorted input filenames; one batch of42 with official metric_batch_size64',
        reference_design='One distinct-ID reference utterance per speaker for voice cloning; not same-utterance reference. Actual recording interval nonoverlap unverified.')
    current['resources']=dict(current['resources'],generation_torch_memory_fraction=.45,scorers_torch_memory_fraction=.60)
    current['resources'].pop('torch_memory_fraction')
    selection={r['source']:r for r in json.loads((RUN/'selection.json').read_text())['selected']}
    for row in old_inputs['rows']:
        row.pop('strict_reference',None)
        row.update(reference_design='separate_id_utterance',reference_id=selection[row['source']]['reference_id'])
    write(RUN/'protocol.json',current);write(RUN/'inputs.json',old_inputs)
    assert all(sha(RUN/p)==h for p,h in science.items())
    write(RUN/'cohort_metadata_correction.json',dict(status='PASS',reason='Inherited old Chinese20 template carried stale counts, dataset/parent/PEAVS descriptions and old strict_reference label. Actual selected21 pairs and all inference/scoring parameters unchanged.',
        original_protocol_sha256=sha(backup/'protocol.json'),current_protocol_sha256=sha(RUN/'protocol.json'),
        original_inputs_sha256=sha(backup/'inputs.json'),current_inputs_sha256=sha(RUN/'inputs.json'),
        scope='Metadata only. Selection, IDs, source quality flags, primary17/all21 membership, audio/image/video SHA, normalization, model settings, scoring endpoints/support and seeds unchanged. No forward repeated.',
        preserved_science_files=len(science),source_sha256=sha(__file__)))
    print('COHORT_METADATA_CORRECTION_PASS',len(science))

if __name__=='__main__':main()
