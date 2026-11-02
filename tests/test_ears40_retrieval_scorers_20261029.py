import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts/experiments'))
import ears40_retrieval_scorers_20261029 as m


def test_physical_pcm_shift_keeps_length_without_circular_wrap():
    import numpy as np
    x = np.arange(12, dtype=np.int16)+1
    assert np.array_equal(m.shift_pcm(x, 3), np.r_[np.zeros(3, dtype=x.dtype), x[:-3]])
    assert np.array_equal(m.shift_pcm(x, -3), np.r_[x[3:], np.zeros(3, dtype=x.dtype)])
    assert np.array_equal(m.shift_pcm(x, 0), x)


def test_k_tolerance_is_fixed_while_context_changes():
    import numpy as np
    z = np.zeros((80, 31), dtype=np.float32)
    z[:, 18] = 10
    assert all(v == 0 for v in m.independent_reduction(z).values())
    z[:, 18] = 0
    z[:, 15] = 10
    assert all(v == 1 for v in m.independent_reduction(z).values())


def test_relative_tracking_does_not_make_failed_k_control_pass():
    native = dict(interior_offset_frames=-3, metrics={'K5': 0.})
    shifted = dict(interior_offset_frames=-13, metrics={'K5': 0.})
    c = m.control_record(native, shifted, 6400)
    assert c['relative_offset_recovered']
    assert c['relative_offset_error_frames'] == 0
    assert not c['k5_decreased']
