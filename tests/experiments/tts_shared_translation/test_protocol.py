import numpy as np

from scripts.experiments.tts_shared_translation import metric


def test_common_translation_preserves_native_geometry():
    x=np.array([1.,2.,3.])
    y=np.array([[1.,2.,3.],[4.,2.,1.],[2.,4.,8.]])
    delta=np.array([.2,-2.,.75])
    np.testing.assert_allclose(metric(x,y),metric(x-delta,y-delta),rtol=0,atol=1e-14)


def test_zero_sentinel_must_transform_for_coordinate_identity():
    x=np.array([1.,2.]);padded=np.array([[0.,0.],[2.,3.]])
    delta=np.array([.2,.5])
    np.testing.assert_allclose(np.linalg.norm(x-padded,axis=1),np.linalg.norm((x-delta)-(padded-delta),axis=1))
    assert not np.isclose(np.linalg.norm(x),np.linalg.norm(x-delta))


def test_audio_weight_is_not_multiplied_by_number_of_images():
    da=np.array([1.,0.]);dv=np.array([0.,2.])
    # Three identical visual estimates retain 50:50 modality weighting.
    delta=(da+np.mean([dv,dv,dv],axis=0))/2
    np.testing.assert_array_equal(delta,[.5,1.])
