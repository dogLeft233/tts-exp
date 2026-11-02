"""Freeze full-TTS extension and bind immutable N/R parent assets."""
from pathlib import Path
import datetime
import hashlib
import json


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root=Path('/home/wjj/tts-audio/tts-exp')
    parent=root/'runs/qwen_codec_roundtrip_20261022_attempt01'
    out=root/'runs/qwen_full_tts_extension_20261022_attempt01'
    out.mkdir(exist_ok=False)
    pm=json.loads((parent/'input_manifest.json').read_text())
    old=json.loads((parent/'artifact_manifest.json').read_text())
    assets={}
    # Bind data and numeric results, but do not rewrite or regenerate them.
    for row in old['files']:
        rel=row['path']
        if (rel in ['input_manifest.json','audio_manifest.json','video_manifest.json','matrix_scores.json']
            or rel.startswith(('features/','scores/','distances/','video/','audio/','input24/','codec_attempt02/'))):
            p=parent/rel
            if sha(p)!=row['sha256']:raise ValueError(f'parent changed: {rel}')
            assets[rel]=dict(path=str(p),sha256=row['sha256'],bytes=row['bytes'])
    model=root/'models/Qwen3-TTS-12Hz-0.6B-Base'
    if not model.exists():
        model=Path('/home/wjj/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc')
    if sha(model/'model.safetensors')!='180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6':raise ValueError('LM weight mismatch')
    p=dict(run_id=out.name,status='FROZEN_BEFORE_ANY_NEW_TTS_WAVEFORM',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           parent_run=str(parent),parent_manifest_sha256=sha(parent/'input_manifest.json'),
           parent_artifact_manifest_sha256=sha(parent/'artifact_manifest.json'),parent_assets=assets,
           model_dir=str(model.resolve()),model_weights_sha256=sha(model/'model.safetensors'),
           codec_weights_sha256=pm['model_weights_sha256'],samples=pm['samples'],seed_bases=[2026102201,2026102202,2026102203],
           generation_kwargs=dict(do_sample=True,temperature=.9,top_k=50,top_p=1.,max_new_tokens=512),
           non_streaming_mode=False,reference_policy='self_icl_same_target',
           reference=dict(audio='parent input24_path, same natural utterance',text='remove all whitespace from fixed corpus transcript for both ref_text and target text',
                          x_vector_only=False,prompt='one fixed prompt per speaker reused across three predetermined seeds'),
           execution=dict(training=False,device='cuda:0',dtype='float32',attention='eager',offline=True,native_padding=True,
                          decoder='explicit same parent eager FP32 tokenizer; all 16 codebooks; preserve official ICL reference prefix and cut',
                          code_capture='observe model.generate outputs and actual decoder inputs without changing returned codes/waves',
                          parity='actual concat codes redecoded and cut with official formula must exactly reproduce official returned new waveform',
                          reference_code_bridge='each reference code array must equal parent natural-codec code array exactly',
                          max_tts=30,max_new_videos=30,tts_timeout_s=1800,render_timeout_s=1800,score_timeout_s=900,storage_cap_bytes=2000000000),
           audio_diagnostics=dict(main='official full ICL T output only',
                                  new_codes_only='save additional audio from new codes without decoder reference prefix; never substitute for main T',
                                  R_with_reference_context='first seed per source only: decode refcode+parent R code then official prefix cut, compare to parent raw R audio',
                                  interpretation='if decoder reference context differs from R, T-R is full generative-path contrast, not isolated LM-training-objective effect'),
           downstream=pm['generation'],target_rms=pm['primary_audio']['target_rms'],
           scoring=dict(conditions=['N','R','T1','T2','T3'],reuse='exact bound parent N/R video, source PCM and raw feature arrays; no rerender',
                        new_cells='only self-paired T1,T2,T3; N/R self curves recomputed from parent embeddings',
                        lags=list(range(-15,16)),primary='same q interval for all five self-paired conditions per speaker: [15,min(native feature counts)-15)',
                        sensitivity=['full native all-lag-valid support per condition',
                                     'equal query count at evenly spaced unique positions through each condition native valid interval'],
                        timing='no waveform warp, no re-alignment or peak selection; own source PCM for every video; different T speaking times are part of contrast',
                        endpoints=['C','B','Dmin','D0','offset'],report_low_query_count_without_dropping=True),
           statistics=dict(unit='10 speakers; first average T1/T2/T3 within speaker; seeds are repeated generation, not independent people',
                           bootstrap_draws=10000,seed=20261022,primary=['T-N C','T-R C'],
                           intervals='95% descriptive plus 97.5% per contrast for Bonferroni two-main-comparison family; development evidence only',
                           per_seed='report all 3 seed indices; never select best seed',
                           other_endpoints='diagnostic, no confirmatory or perceptual claim'),
           stop_rules=['no model/input hash changes','no training','no source/seed replacement based on waveform or score',
                       'nonfinite waveform or normalization clipping stops stage; retain failures',
                       'absent common real support blocks primary rather than silently dropping a source',
                       'max token exhaustion retained and labeled, not resampled'],
           scope='same historically exposed development sources, single Wav2Lip/static face; upstream training overlap and content/perception not independently certified')
    (out/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(path=str(out/'protocol.json'),sha256=sha(out/'protocol.json'),parent_assets=len(assets),samples=len(p['samples']),tts_count=30),ensure_ascii=False))


if __name__=='__main__':main()
