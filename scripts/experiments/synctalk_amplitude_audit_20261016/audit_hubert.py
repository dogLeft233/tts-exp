"""Run the exact official HuBERT helper using the pinned public large model."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import HubertConfig, HubertModel, Wav2Vec2Processor

from audit import difference, extract, sha, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--frontend', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    processor = Wav2Vec2Processor.from_pretrained(a.model, local_files_only=True)
    # torch 2.5 + transformers 4.39 silently misses the old weight_norm pair in
    # from_pretrained. Map only its parameter names, then demand strict loading.
    config = HubertConfig.from_pretrained(a.model, local_files_only=True)
    model = HubertModel(config)
    raw = torch.load(a.model / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    state = {}
    renamed = {}
    target_keys = set(model.state_dict())
    for name, value in raw.items():
        if not name.startswith('hubert.'):
            continue  # CTC lm_head is not part of official HubertModel.
        key = name[len('hubert.'):]
        if key.endswith('.weight_g') and key not in target_keys:
            key = key[:-len('weight_g')] + 'parametrizations.weight.original0'
        elif key.endswith('.weight_v') and key not in target_keys:
            key = key[:-len('weight_v')] + 'parametrizations.weight.original1'
        state[key] = value
        if key != name[len('hubert.'):]:
            renamed[name] = key
    loaded = model.load_state_dict(state, strict=True)
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
    conv = model.encoder.pos_conv_embed.conv
    effective = torch._weight_norm(raw['hubert.encoder.pos_conv_embed.conv.weight_v'],
                                   raw['hubert.encoder.pos_conv_embed.conv.weight_g'], 2)
    effective_error = float(torch.max(torch.abs(conv.weight - effective)))
    assert effective_error == 0
    del raw, state, effective
    model = model.cuda().eval()  # Match official from_pretrained eval behavior.
    scope = dict(torch=torch, np=np, wav2vec2_processor=processor, hubert_model=model)
    provenance = []
    extract(a.root / 'official/data_utils/hubert.py',
            ['get_hubert_from_16k_speech', 'make_even_first_dim'], scope, provenance)
    captured = {}
    def hook(name):
        def save(module, inputs, outputs):
            value = outputs[0] if isinstance(outputs, tuple) else outputs
            captured[name] = value.detach().cpu().numpy()
        return save
    handles = [model.feature_extractor.register_forward_hook(hook('cnn')),
               model.feature_projection.layer_norm.register_forward_hook(hook('feature_layer_norm')),
               model.feature_projection.register_forward_hook(hook('projected'))]
    old = json.loads(a.frontend.read_text())
    result = {'model_id': 'facebook/hubert-large-ls960-ft',
              'revision': 'ece5fabbf034c1073acae96d5401b25be96709d8',
              'weights_sha256': sha(a.model / 'pytorch_model.bin'),
              'transformers': transformers.__version__, 'model_training': model.training,
              'strict_load_missing': list(loaded.missing_keys),
              'strict_load_unexpected': list(loaded.unexpected_keys),
              'parameter_name_compatibility_map': renamed,
              'effective_posconv_weight_max_error': effective_error,
              'provenance': provenance, 'stages': [], 'comparisons': [], 'repeats': []}
    for item in old['inputs']:
        variants = {}
        for c in item['conditions']:
            if c['status'] != 'RUN':
                continue
            db = c['gain_db']
            wav = Path(c['path'])
            speech, sr = sf.read(wav)
            assert sr == 16000 and speech.ndim == 1
            inputs = processor(speech, return_tensors='pt', sampling_rate=16000).input_values.numpy()
            hidden = scope['get_hubert_from_16k_speech'](speech).numpy()
            snapshots = {k: v.copy() for k, v in captured.items()}
            repeat = scope['get_hubert_from_16k_speech'](speech).numpy()
            result['repeats'].append({'input_id': item['id'], 'gain_db': db, **difference(repeat, hidden)})
            # Explicit epsilon law for the feature extractor's entire-utterance normalization.
            centered = speech.astype(np.float32) - np.mean(speech.astype(np.float32))
            formula = centered / np.sqrt(np.var(speech.astype(np.float32)) + 1e-7)
            values = dict(processor=inputs, hidden=hidden, paired_cache=hidden[:len(hidden)//2*2].reshape(-1,2,1024), **snapshots)
            out = a.output / f"input{item['id']}_gain{db:+d}"
            out.mkdir()
            for name, v in values.items():
                path = out / (name + '.npy')
                np.save(path, v)
                result['stages'].append({'input_id': item['id'], 'gain_db': db, 'stage': name,
                                         'path': str(path), **summary(v)})
            result['comparisons'].append({'input_id': item['id'], 'gain_db': db,
                                          'stage': 'processor_epsilon_formula', **difference(inputs[0], formula)})
            variants[db] = values
        for db, values in variants.items():
            for name, v in values.items():
                result['comparisons'].append({'input_id': item['id'], 'gain_db': db, 'stage': name,
                                              **difference(v, variants[0][name])})
        (a.output / 'statistics.json').write_text(json.dumps(result, indent=2))
        print('Completed HuBERT input', item['id'], flush=True)
    for handle in handles:
        handle.remove()
    print('statistics_sha256', sha(a.output / 'statistics.json'))


if __name__ == '__main__':
    main()
