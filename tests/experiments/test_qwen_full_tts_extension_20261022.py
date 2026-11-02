import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('qwen_full_tts_stage',Path(__file__).parents[2]/'scripts/experiments/qwen_full_tts_extension_20261022/tts_stage.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def test_text_removes_only_whitespace_preserving_content():
    assert M.strip_whitespace(' 而\t对\n楼市，\u3000成交。 ')== '而对楼市，成交。'


def test_primary_cut_is_official_ratio_not_resizing():
    assert M.official_cut(3,8,15360)==int(3/8*15360)
    with pytest.raises(ValueError):M.official_cut(8,8,15360)
    x=np.arange(15360,dtype=np.float32)
    assert np.array_equal(x[M.official_cut(3,8,len(x)):],x[5760:])


def test_all_16_codebooks_and_vocab_are_required():
    with pytest.raises(ValueError):M.codes(np.zeros((20,1),np.int64))
    a=np.zeros((2,16),np.int64);a[:,0]=4095;a[:,1:]=2047
    assert np.array_equal(M.codes(a),a)
    a[0,2]=2048
    with pytest.raises(ValueError):M.codes(a)


def test_float_wave_preserves_length_and_amplitude(tmp_path):
    x=np.linspace(-1.2,1.2,19201,dtype=np.float32)
    saved=M.save_wave(tmp_path/'official.wav',x)
    assert saved['samples']==19201 and saved['peak']>1
    with pytest.raises(ValueError):M.waveform(np.array([np.nan]),24000)


def make_fake_wrapper(fail=False):
    ref=np.zeros((3,16),np.int64);new=np.ones((5,16),np.int64)
    sequence_result=SimpleNamespace(sequences=np.array([[1,1,1,1,1,2150]],np.int64))
    talker=SimpleNamespace(generate=lambda **kw:sequence_result)
    def generate(**kw):
        talker.generate(max_new_tokens=512,eos_token_id=2150)
        if fail:raise RuntimeError('test failure')
        return ([new],['unmodified extra'])
    def decode(encoded):return ([np.arange(len(encoded[0]['audio_codes'])*1920,dtype=np.float32)],24000)
    tokenizer=SimpleNamespace(decode=decode)
    wrapper=SimpleNamespace(model=SimpleNamespace(generate=generate,talker=talker,speech_tokenizer=tokenizer))
    return wrapper,ref,new,sequence_result


def test_observation_hooks_do_not_modify_returns_and_restore():
    w,ref,new,seq=make_fake_wrapper();old=(w.model.generate,w.model.talker.generate,w.model.speech_tokenizer.decode)
    with M.capture_official(w) as observed:
        returned=w.model.generate(voice_clone_prompt={'ref_code':[ref]})
        assert returned[0][0] is new
        decoded=w.model.speech_tokenizer.decode([{'audio_codes':np.concatenate([ref,new])}])
        assert len(decoded[0][0])==8*1920
    assert (w.model.generate,w.model.talker.generate,w.model.speech_tokenizer.decode)==old
    assert observed['core_generate_calls']==observed['talker_calls']==observed['decoder_calls']==1
    assert np.array_equal(observed['new_codes'][0],new)
    assert np.array_equal(observed['ref_codes'][0],ref)
    assert np.array_equal(observed['decode_codes'][0],np.concatenate([ref,new]))
    assert np.array_equal(observed['talker_sequences'],seq.sequences)


def test_observation_hooks_restore_after_exception():
    w,ref,new,seq=make_fake_wrapper(fail=True);old=(w.model.generate,w.model.talker.generate,w.model.speech_tokenizer.decode)
    with pytest.raises(RuntimeError,match='test failure'):
        with M.capture_official(w):w.model.generate(voice_clone_prompt={'ref_code':[ref]})
    assert (w.model.generate,w.model.talker.generate,w.model.speech_tokenizer.decode)==old


def test_seed_plan_all_30_no_selection():
    samples=[{'sample_id':f'u{i}','speaker_id':f'S{i:04}','transcript':'你 好'} for i in range(1,11)]
    m={'samples':samples,'seed_bases':[2026102201,2026102202,2026102203],
       'generation_kwargs':{'do_sample':True,'temperature':.9,'top_k':50,'top_p':1.,'max_new_tokens':512}}
    rows=M.validate_manifest(m)
    assert len(rows)==30 and len({(r['sample_id'],r['seed_index']) for r in rows})==30
    assert rows[0]['seed']==2026102301 and rows[0]['target_text']==rows[0]['reference_text']=='你好'
    m['seed_bases'].append(123)
    with pytest.raises(ValueError):M.validate_manifest(m)
