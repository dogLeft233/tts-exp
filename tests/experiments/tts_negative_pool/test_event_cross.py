import numpy as np

from scripts.experiments.tts_event_cross import CONTRASTS, distance


def test_matching_modalities_and_mismatches_have_expected_distance():
    visual=np.array([1.,0.])
    audio=np.array([[1.,0.],[0.,1.]])
    result=distance(visual,audio)
    np.testing.assert_allclose(result,[np.sqrt(2)*1e-6,np.sqrt(2+2e-12)])


def test_modality_main_effects_have_zero_interaction():
    # Additive row/column effects must not become a self-pair interaction.
    cells={"NN":2.,"TN":5.,"NT":9.,"TT":12.}
    effects={name:sum(cells[c]*w for c,w in weights.items()) for name,weights in CONTRASTS.items()}
    assert effects["interaction"]==0.
    assert effects["visual_at_N_audio"]==effects["visual_at_T_audio"]==3.
    assert effects["audio_at_N_visual"]==effects["audio_at_T_visual"]==7.
