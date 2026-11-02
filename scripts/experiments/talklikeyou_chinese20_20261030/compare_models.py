"""Compare TTS-minus-natural differences on identical eligible source intersections."""
import json
from pathlib import Path

from analyze import METRICS, holm, statistics
from prepare import B, RUN, sha, write


def main():
    current = json.loads((RUN/'paired_scores.json').read_text())
    inputs = json.loads((RUN/'scoring_inputs.json').read_text())['rows']
    protocol = json.loads((RUN/'protocol.json').read_text())
    result = dict(status='PASS', current_run=str(RUN), contrasts={},
        interpretation='Exploratory paired differences of TTS effects. Marginal source-bootstrap CI;18 endpoint-by-baseline Holm family within each stratum. Same input audio/face, different native video supports. No equivalence claim.',
        historical_wav2lip='Earlier Wav2Lip Chinese20 used a lower waveform level and different runtime. Its frozen scores are contextual only, not an isolated generator interaction at RMS24.')
    for name, run in [('FlashHead', B/'jobs/flashhead_chinese20_rms24_20261030_attempt01'),
                      ('EchoMimicV3-Flash', B/'jobs/echomimic_v3_flash_chinese20_rms24_20261030_attempt01')]:
        old = json.loads((run/'paired_scores.json').read_text())
        old_inputs = json.loads((run/'scoring_inputs.json').read_text())['rows']
        old_protocol = json.loads((run/'protocol.json').read_text())
        assert {r['id']:r['audio_sha256'] for r in inputs} == {r['id']:r['audio_sha256'] for r in old_inputs}
        assert protocol['generation']['face_sha256'] == old_protocol['generation']['face_sha256']
        comparison = dict(baseline_run=str(run), baseline_pairs_sha256=sha(run/'paired_scores.json'),
                          same_input_pcm_hashes=True, same_face=True, subsets={})
        for subset in ['same_reference12', 'extended_reference8', 'all20']:
            metrics = {}
            for metric in METRICS:
                a = {r['source']:r for r in current[metric]}
                b = {r['source']:r for r in old[metric]}
                common = sorted(a.keys() & b.keys())
                common = [s for s in common if subset == 'all20' or a[s]['strict_reference'] == (subset == 'same_reference12')]
                if not common:
                    metrics[metric] = dict(n_sources=0, status='UNMEASURABLE')
                    continue
                n = [b[s]['delta'] for s in common]
                t = [a[s]['delta'] for s in common]
                stats = statistics(n, t)
                stats.update(sources=common, baseline_T_minus_N=stats.pop('N_mean'),
                             TalkLikeYou_T_minus_N=stats.pop('T_mean'),
                             effect_difference=stats.pop('delta'),
                             same_strict_direction=sum((a[s]['delta']*b[s]['delta']) > 0 for s in common))
                metrics[metric] = stats
            comparison['subsets'][subset] = metrics
        result['contrasts'][name] = comparison
    for subset in ['same_reference12', 'extended_reference8', 'all20']:
        cells = [comparison['subsets'][subset][m] for comparison in result['contrasts'].values() for m in METRICS]
        for cell, adjusted in zip(cells, holm([cell.get('signflip_p', 1.) for cell in cells])):
            cell['holm18_p'] = adjusted
    write(RUN/'cross_model_comparison.json', result)
    lines = ['# Same-input TTS-effect comparisons', '', result['interpretation'], '',
             'Difference=(TalkLikeYou T−N)−(baseline T−N). Compare on shared eligible sources only.', '',
             '|Baseline|Subset|Scorer|n|Baseline T−N|TalkLikeYou T−N|Difference|95% CI|Holm18 p|',
             '|---|---|---|---:|---:|---:|---:|---|---:|']
    for name, comparison in result['contrasts'].items():
        for subset, metrics in comparison['subsets'].items():
            for metric, cell in metrics.items():
                if not cell['n_sources']:
                    continue
                digits = '.3f' if metric == 'SyncNet' else '.6f'
                fmt = lambda x: format(x, '+'+digits)
                lines.append(f"|{name}|{subset}|{metric}|{cell['n_sources']}|{fmt(cell['baseline_T_minus_N'])}|{fmt(cell['TalkLikeYou_T_minus_N'])}|{fmt(cell['effect_difference'])}|[{fmt(cell['ci95'][0])}, {fmt(cell['ci95'][1])}]|{cell['holm18_p']:.5f}|")
    lines += ['', result['historical_wav2lip'], '']
    (RUN/'cross_model_comparison.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
