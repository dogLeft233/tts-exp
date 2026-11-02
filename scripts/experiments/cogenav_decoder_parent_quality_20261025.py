"""Independent, prespecified audio quality evaluation; no SyncNet or tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import unicodedata


def normalized(text):
    return ''.join(c.lower() for c in unicodedata.normalize('NFKC', text)
                   if not c.isspace() and not unicodedata.category(c).startswith(('P', 'S')))


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis)+1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]


def load_audio(path):
    import numpy as np
    from scipy.io import wavfile
    rate, data = wavfile.read(path)
    if rate != 16000 or data.ndim != 1:
        raise ValueError('Expected mono16k audio')
    if data.dtype == np.int16:
        data = data.astype(np.float64)/32768
    elif data.dtype.kind == 'f':
        data = data.astype(np.float64)
    else:
        raise ValueError('Unsupported sample encoding')
    if not np.isfinite(data).all() or not len(data):
        raise ValueError('Invalid audio')
    return data


def pair_metrics(reference, reconstructed):
    import numpy as np
    from scipy.signal import stft
    from pesq import pesq
    from pystoi import stoi
    import warnings
    if reference.shape != reconstructed.shape:
        raise ValueError('Pair sample counts differ')
    rows = dict(sample_count=len(reference), rms_delta_db=float(20*np.log10(
        max(np.sqrt(np.mean(reconstructed**2)), 1e-12)/max(np.sqrt(np.mean(reference**2)), 1e-12))),
        peak=float(np.max(np.abs(reconstructed))), overrange_samples=int(np.sum(np.abs(reconstructed)>1)))
    metric_errors = {}
    for name, fn in [('pesq_wb', lambda: pesq(16000, reference, reconstructed, 'wb')),
                     ('stoi', lambda: stoi(reference, reconstructed, 16000, extended=False)),
                     ('estoi', lambda: stoi(reference, reconstructed, 16000, extended=True))]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                value = float(fn())
            rows[name] = value if np.isfinite(value) else None
            if caught:
                metric_errors[name+'_warnings'] = [str(w.message) for w in caught]
        except Exception as exc:
            rows[name] = None
            metric_errors[name] = str(exc)
    freq, _, ref_spec = stft(reference, 16000, window='hann', nperseg=1024, noverlap=768)
    _, _, rec_spec = stft(reconstructed, 16000, window='hann', nperseg=1024, noverlap=768)
    keep = (freq >= 80) & (freq <= 7600)
    a, b = np.abs(ref_spec[keep]), np.abs(rec_spec[keep])
    rows['log_spectral_rmse_db'] = float(np.sqrt(np.mean((20*np.log10(np.maximum(a, 1e-5))-
                                                         20*np.log10(np.maximum(b, 1e-5)))**2)))
    rows['spectral_convergence'] = float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-12))
    rows['metric_errors'] = metric_errors
    return rows


def conditions(args, rows):
    result = {'original': {Path(r['audio_path']).stem: Path(args.dataset)/r['audio_path'] for r in rows}}
    for condition in args.condition:
        name, directory = condition.split('=', 1)
        if name in result:
            raise ValueError('Duplicate condition')
        index = {}
        for path in Path(directory).rglob('*.wav'):
            if path.stem in index:
                raise ValueError('Ambiguous audio ID: '+path.stem)
            index[path.stem] = path
        result[name] = {Path(r['audio_path']).stem: index[Path(r['audio_path']).stem] for r in rows}
    return result


def run(args):
    selection = Path(args.selection)
    rows = json.loads(selection.read_text(encoding='utf-8'))['rows']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    paths = conditions(args, rows)
    results = []
    if args.mode == 'asr':
        import numpy as np
        import torch
        from funasr import AutoModel
        torch.set_num_threads(2)
        torch.manual_seed(20261025)
        np.random.seed(20261025)
        random.seed(20261025)
        model = AutoModel(model=args.asr_model, device='cpu', disable_update=True)
    for row in rows:
        identity = Path(row['audio_path']).stem
        reference = normalized(row['text'])
        for condition, index in paths.items():
            path = index[identity]
            item = dict(id=identity, speaker=row['speaker'], condition=condition,
                        audio_path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if condition == 'original' and item['sha256'] != row['output_sha256']:
                raise ValueError('Original differs from fixed manifest')
            if args.mode == 'asr':
                # Kaldi-style fbank dithering may consume RNG. Bind it to the
                # utterance, so interleaving conditions cannot change its draw.
                seed = int(hashlib.sha256(('asr:'+identity).encode()).hexdigest()[:8],16)
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                result = model.generate(input=str(path), disable_pbar=True)
                hypothesis = normalized(result[0]['text'])
                item.update(reference=reference, hypothesis=hypothesis,
                            edits=edit_distance(reference, hypothesis), characters=len(reference),
                            asr_rng_seed=seed)
            else:
                item.update(pair_metrics(load_audio(paths['original'][identity]), load_audio(path)))
            results.append(item)
        print(json.dumps(dict(done=len(results), id=identity)), flush=True)
    summary = {}
    for condition in paths:
        subset = [r for r in results if r['condition'] == condition]
        if args.mode == 'asr':
            edits = sum(r['edits'] for r in subset)
            chars = sum(r['characters'] for r in subset)
            summary[condition] = dict(n=len(subset), edits=edits, characters=chars, cer=edits/chars)
        else:
            summary[condition] = dict(n=len(subset))
            for key in ['pesq_wb','stoi','estoi','log_spectral_rmse_db','spectral_convergence','rms_delta_db']:
                values = [r[key] for r in subset if r[key] is not None]
                summary[condition][key] = dict(measured=len(values), mean=sum(values)/len(values) if values else None)
    result = dict(mode=args.mode, selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
                  selection_policy='Fixed before results; excludes training monitor clips',
                  quality_scope='Objective proxies; not a human listening test or downstream sync result',
                  summary=summary, rows=results)
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['quality','asr'], required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--condition', action='append', default=[])
    parser.add_argument('--asr-model', default='/home/wjj/tts-audio/model_cache/funasr_20261024/paraformer-zh')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
