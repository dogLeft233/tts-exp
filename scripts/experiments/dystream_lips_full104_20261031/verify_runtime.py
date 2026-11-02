"""Check shared scorer entrypoints and installed packages against both frozen baselines."""
import json
import subprocess
from prepare import B,D,RUN,sha,write


def main():
    baselines = [B/'jobs/flashhead_chinese20_rms24_20261030_attempt01',
                 B/'jobs/echomimic_v3_flash_chinese20_rms24_20261030_attempt01']
    record = dict(status='PASS', baselines=[])
    for parent in baselines:
        manifest = json.loads((parent/'code_snapshot/manifest.json').read_text())
        entries = [x for x in manifest if x['original_path'].startswith(str(B/'legacy_snapshot')+'/') or
                   x['original_path'] in [str(B/name) for name in ['run_scorer.sh', 'gpu_guard.py', 'score_avs_u.py',
                   'score_classifier.py', 'score_retrieval.py', 'score_peavs.py', 'score_syncreward.py']]]
        for entry in entries:
            assert sha(entry['original_path']) == entry['sha256'], entry['original_path']
        environments = []
        for name in ['eval-base', 'cogenav', 'synchformer', 'syncreward', 'peavs']:
            actual = subprocess.check_output([str(B/'envs'/name/'bin/python'), '-m', 'pip', 'freeze'], text=True)
            previous = (parent/'code_snapshot'/('environment-'+name+'.freeze.txt')).read_text()
            assert set(actual.splitlines()) == set(previous.splitlines()), (parent.name, name)
            environments.append(name)
        record['baselines'].append(dict(run=str(parent), checked_source_files=len(entries), matched_environments=environments))
    for path,h in json.loads((RUN/'source_pins.json').read_text())['files'].items():assert sha(path)==h,path
    for path,h in json.loads((RUN/'experiment_source_manifest.json').read_text())['files'].items():assert sha(path)==h,path
    freeze=subprocess.check_output([str(D/'env/bin/python'),'-m','pip','freeze'],text=True)
    assert set(freeze.splitlines())==set((RUN/'dystream_environment_baseline.txt').read_text().splitlines())
    record['dystream_source_assets_and_environment']='PASS'
    write(RUN/'scorer_runtime_comparison.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
