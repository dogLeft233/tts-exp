"""Scientific unit and waveform contracts; no GPU/model or cloud API calls."""
import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest

S=Path(__file__).parents[1]/'scripts/experiments/talklikeyou_hdtf20_20261031'
sys.path.insert(0,str(S))
def module(name):
    spec=importlib.util.spec_from_file_location('hdtf_'+name,S/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
analysis=module('analyze');bundle=module('build_bundle');prepare=module('prepare')

def test_two_clips_are_one_speaker_and_missing_clip_excludes_whole_speaker():
    rows=[dict(source=s,speaker=person) for s,person in [('a','one'),('b','one'),('c','two'),('d','two')]]
    clips=[dict(source=s,speaker=person,N=n,T=t,qc_flag=False) for s,person,n,t in [('a','one',1,3),('b','one',9,9),('c','two',3,4),('d','two',4,6)]]
    pairs,missing=analysis.speaker_pairs(rows,clips)
    assert not missing and len(pairs)==2
    assert pairs[0]['N']==5 and pairs[0]['T']==6 and pairs[0]['delta']==1
    pairs,missing=analysis.speaker_pairs(rows,clips[:-1])
    assert len(pairs)==1 and missing==['two']

def test_rms_roundtrip_preserves_length_and_refuses_peak_overflow():
    x=np.sin(np.arange(160000)*.14)*.3
    pcm,stats=bundle.normalize(x)
    assert len(pcm)==len(x) and pcm.dtype==np.int16
    assert abs(20*np.log10(np.sqrt(np.mean((pcm.astype(float)/32768)**2)))+27)<.002
    peak=np.zeros(160000);peak[2]=.9
    with pytest.raises(AssertionError,match='peak ceiling'):bundle.normalize(peak)
    assert prepare.native_frames(160000)==245

def test_pairing_rejects_draw_pseudoreplication_and_uses_expected_arm():
    rows=[dict(source='c',speaker='s',id='c_N',arm='N',tts_qc_flag=False),dict(source='c',speaker='s',id='c_T',arm='T',tts_qc_flag=False)]
    pairs,missing=analysis.pair_clips(rows,{'c_N':4.,'c_T':5.})
    assert not missing and pairs[0]['delta']==1
    with pytest.raises(AssertionError):analysis.pair_clips(rows+[dict(rows[1],id='c_T2')],{'c_N':4.,'c_T':5.,'c_T2':8.})


def test_validation_accepts_native816_and_rejects_unexpected_resize(monkeypatch,tmp_path):
    import json,types
    import soundfile as sf
    generation=module('generate')
    pcm=np.arange(1920,dtype=np.int16)
    audio=tmp_path/'audio.wav';sf.write(audio,pcm,16000,subtype='PCM_16')
    row=dict(audio_path=str(audio),face_path=str(tmp_path/'face.png'),expected_frames=3)
    actual_size=[816]
    class Capture:
        def __init__(self):self.index=0
        def get(self,key):return 25
        def read(self):
            if self.index==3:return False,None
            image=np.full((actual_size[0],actual_size[0],3),self.index,dtype=np.uint8)
            self.index+=1;return True,image
        def release(self):pass
    fake=types.SimpleNamespace(VideoCapture=lambda _:Capture(),CAP_PROP_FPS=1,COLOR_BGR2GRAY=0,
        imread=lambda _:np.zeros((816,816,3),np.uint8),cvtColor=lambda x,_:x[:,:,0],
        resize=lambda x,size:np.zeros((size[1],size[0],3),np.uint8),imwrite=lambda *args:True)
    monkeypatch.setitem(sys.modules,'cv2',fake)
    def output(command):
        if '-select_streams' in command:
            return json.dumps(dict(frames=[dict(best_effort_timestamp_time=str(i/25)) for i in range(3)])).encode()
        return pcm.astype('<i2').tobytes()
    monkeypatch.setattr(generation.subprocess,'check_output',output)
    assert generation.validate(tmp_path/'native.mp4',row)['resolution']==[816,816]
    actual_size[0]=512
    with pytest.raises(AssertionError):generation.validate(tmp_path/'resized.mp4',row)

def test_native_face_gate_keeps95percent_and_rejects94_without_relaxation():
    adapter=module('batch_adapters')
    frames=[None]*100
    assert len(adapter.native_landmarks(lambda _: [1]*95+[None]*5,frames))==100
    with pytest.raises(adapter.NativeFaceSupportError) as error:
        adapter.native_landmarks(lambda _: [1]*94+[None]*6,frames)
    assert error.value.detected==94 and error.value.total==100
    def no_faces(_):raise AssertionError('Cannot detect any frames in the video')
    with pytest.raises(adapter.NativeFaceSupportError):adapter.native_landmarks(no_faces,frames)
    def other_error(_):raise AssertionError('unrelated model defect')
    with pytest.raises(AssertionError,match='unrelated'):adapter.native_landmarks(other_error,frames)
