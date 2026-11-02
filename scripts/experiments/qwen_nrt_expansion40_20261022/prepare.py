"""Freeze all 40 metadata-qualified speakers before any new N/R/T inference."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess
import wave

ROOT=Path('/home/wjj/tts-audio/tts-exp')
RUN=ROOT/'runs/qwen_nrt_expansion40_20261022_attempt01'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    RUN.mkdir(exist_ok=False)
    parent=ROOT/'runs/qwen_codec_roundtrip_20261022_attempt01'
    pilot=ROOT/'runs/qwen_full_tts_extension_20261022_attempt01'
    assert sha(parent/'artifact_manifest.json')=='be88daf12c0318ca5b73750383691a5dbe97cd60df22a08a47d860ddaf03bb6b'
    assert sha(pilot/'artifact_manifest.json')=='680399a642b6be7067e672a63ce949040275d1a6c8740ca9893b93007a27c3ba'
    p=read(pilot/'protocol.json');old={s['speaker_id']:s for s in p['samples']}
    base=ROOT/'data/native_mechanism_holdout_20260926';chosen={}
    for audio in sorted((base/'audio').glob('*.wav')):
        sid=re.search(r'S\d{4}',audio.stem).group();text=base/'transcripts'/(audio.stem+'.txt')
        if sid in chosen or not text.exists():continue
        with wave.open(str(audio)) as w:
            frames=w.getnframes();rate=w.getframerate();channels=w.getnchannels();duration=frames/rate
        if not 3<=duration<=8:continue
        if sid in old:
            row=dict(old[sid]);assert row['sample_id']==audio.stem and row['source_sha256']==sha(audio)
        else:
            dest=RUN/'input24'/(audio.stem+'.wav');dest.parent.mkdir(exist_ok=True)
            cmd=['/usr/bin/ffmpeg','-nostdin','-v','error','-n','-i',str(audio),'-map','0:a:0','-vn','-ac','1','-ar','24000','-c:a','pcm_f32le',str(dest)]
            subprocess.run(cmd,check=True,timeout=30)
            row=dict(sample_id=audio.stem,speaker_id=sid,source_path=str(audio),source_sha256=sha(audio),transcript_path=str(text),
                     transcript_sha256=sha(text),transcript=text.read_text().strip(),duration_s=duration,
                     source_meta=dict(samples=frames,sample_rate_hz=rate,channels=channels),input24_path=str(dest),input24_sha256=sha(dest),conversion_command=cmd)
        row['cohort']='pilot10' if sid in old else 'new30';chosen[sid]=row
    assert len(chosen)==40 and set(chosen)=={f'S{i:04d}' for i in range(2,42)}
    model=Path(p['model_dir'])
    assert sha(model/'model.safetensors')==p['model_weights_sha256']
    assert sha(model/'speech_tokenizer/model.safetensors')==p['codec_weights_sha256']
    reused={}
    for run in [parent,pilot]:
        for f in ['artifact_manifest.json','input_manifest.json','audio_manifest.json','video_manifest.json','feature_manifest.json','matrix_scores.json','analysis.json']:
            file=run/f
            if file.exists():reused[str(file)]=dict(sha256=sha(file),bytes=file.stat().st_size)
    result=dict(run_id=RUN.name,status='FROZEN_BEFORE_NEW_MODEL_FORWARD',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         parent_run=str(parent),pilot_tts_run=str(pilot),model_dir=str(model),model_weights_sha256=p['model_weights_sha256'],codec_weights_sha256=p['codec_weights_sha256'],
         samples=list(chosen.values()),reused_bindings=reused,seed_bases=p['seed_bases'],generation_kwargs=p['generation_kwargs'],non_streaming_mode=False,
         target_rms=p['target_rms'],downstream=p['downstream'],
         execution=dict(training=False,device='cuda:0',dtype='float32',requested_attention='eager',nested_attention='record actual SDPA configs; match prior loader',
                        offline=True,native_padding=True,strict_deterministic_algorithms=False,no_TF32=True,single_gpu_owner=True,
                        max_new_R=30,max_new_TTS=90,max_new_videos=150,tts_timeout_s=7200,render_timeout_s=3600,score_timeout_s=1200,
                        generation_storage_cap_bytes=1000000000,total_storage_cap_bytes=8000000000),
         reference=dict(policy='same utterance full self-ICL',audio='same parent-style input24 FLOAT',text='fixed transcript whitespace removed, both target/ref',
                        codec='all16 codebooks; R standalone refcodes decode; T official ref+new decode followed by official ratio cut',
                        version_bridge='old S0002 codes and raw R must reproduce exactly before new generation',
                        parity='every T actual ref+new concat repeated decode and official cut must reproduce exact raw output'),
         cohort=dict(selection='all40 existing speakers in same source folder; first sorted transcript-existing 3..8s original per speaker; exclude no outputs',
                     pilot='10 previously inspected speakers S0002..S0011; existing sealed N/R/T and embeddings reused',
                     added='30 speakers S0012..S0041; new N/R/T protocol outputs; historically exposed source corpus, not guaranteed unseen or training-disjoint',
                     primary_analysis_population='new30 only',secondary_analysis_population='all40, and pilot10 as bridge; no optional stopping'),
         scoring=dict(conditions=['N','R','T1','T2','T3'],source_audio='own canonical PCM16 for every video, never mux AAC',
                      lags=list(range(-15,16)),primary='all five same absolute q [15,min(feature_counts)-15), real all-lag support only',
                      sensitivities=['native_full per condition all real valid q','equal_count_spread unique evenly spaced q through own native valid interval'],
                      endpoints=['C','B','Dmin','D0','offset'],no_timewarp=True,no_SyncNet_realignment=True,
                      low_support_warning_queries=25,empty_real_support='report unmeasurable coverage; no replacement or silent exclusion',
                      previous_bridge='all10 pilot all5 conditions on same supports reproduce prior scalar analysis within1e-12'),
         statistics=dict(unit='speaker: first average three T draws; new30 main, all40 secondary; never audio draws or frames as independent people',
                         bootstrap_draws=10000,seed=20261022,primary=['new30 common T-N C','new30 common T-R C'],
                         CI='95% descriptive plus 97.5% per main comparison, Bonferroni two-comparison family',
                         sensitivity='all3 supports and all3 seeds reported; diagnostic, no additional confirmatory hypotheses',
                         leave_one_out='all speakers symmetrically; no score-selected omissions'),
         reliability=dict(main='all three predetermined outputs retained regardless EOS/duration/ASR/score',
                          supplemental_valid_draw='EOS is True and 0.5 <= generated_duration/natural_duration <= 2.0',
                          supplemental='average remaining qualified draws within each person on original five-condition support; if none, report missing coverage',
                          interpretation='conditional-generation diagnostic, not replacement main or human naturalness gate; cutoff chosen after pilot, frozen before new30',
                          ASR='new30 all150 N/R/T canonical PCM with existing Whisper-small CPU int8, beam5,temp0,no VAD/no target hint; CER proxies with numeric-spelling caution'),
         stop_rules=['no training or model/hash change','no source/seed replacement or resampling','no trimming long tails from main',
                     'nonfinite/normalization clipping/hash/version bridge/parity failure stops affected stage and retains artifacts',
                     'no dropping to gain significance; fixed30 expansion regardless interim score','source/media current original seals never rewritten'],
         scope='single fixed face and Wav2Lip; self-score endpoint, no human mouth accuracy, no isolated LM objective claim')
    file=RUN/'protocol.json';file.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(path=str(file),sha256=sha(file),all40=len(chosen),new30=sum(s['cohort']=='new30' for s in chosen.values()),new_TTS=90,new_videos=150)))


if __name__=='__main__':main()
