from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts/experiments'))
import ears40_regressor_scorers_20261029 as m


def test_control_batches_never_split_a_source_family_at_batch64_boundary():
    natural = [dict(id='p%03d_N_0'%i, video_path='p%03d_N_0.mp4'%i) for i in range(40)]
    shifted = [dict(id=r['id'], shift_samples=s, video_path=r['id']+'_'+str(s)+'.mp4')
               for r in natural for s in [-6400, 6400]]
    batches = m.source_control_batches(natural, shifted)
    assert [len(b) for b in batches] == [63, 57]
    for bi, batch in enumerate(batches):
        assert len(batch) <= 64
        assert len(set(batch)) == len(batch)
        for r in natural[bi*21:bi*21+21]:
            assert r['video_path'] in batch
            assert r['id']+'_-6400.mp4' in batch
            assert r['id']+'_6400.mp4' in batch


def test_shift_is_non_circular_and_exact_int16():
    import numpy as np
    pcm = np.array([1, 2, -3, 4, -5], dtype=np.int16)
    assert np.array_equal(m.shift_pcm(pcm, 2), [0, 0, 1, 2, -3])
    assert np.array_equal(m.shift_pcm(pcm, -2), [-3, 4, -5, 0, 0])
    assert m.shift_pcm(pcm, 2).dtype == np.int16
