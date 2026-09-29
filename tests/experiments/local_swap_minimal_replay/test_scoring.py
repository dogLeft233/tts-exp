import numpy as np

from scripts.experiments.local_swap_minimal_replay.scoring import pairwise_distance_reference, reconstruct_global


def test_matrix_reconstruction_is_finite_and_has_thirty_one_offsets() -> None:
    matrix = np.linspace(0.1, 2.0, 31, dtype=np.float64)[None, :].repeat(12, axis=0)
    matrix[:, 15] = 0.01
    result = reconstruct_global(matrix)
    assert len(result["offsets"]) == 31
    assert result["offset"] == 0
    assert result["sync_d"] == float(matrix[:, 15].mean())


def test_pairwise_reference_uses_syncnet_zero_based_offset_axis() -> None:
    visual = np.eye(4, dtype=np.float64)
    audio = np.eye(4, dtype=np.float64)
    matrix = pairwise_distance_reference(visual, audio, vshift=1)
    assert matrix.shape == (4, 3)
    assert int(np.argmin(matrix[1])) == 1
    assert matrix[1, 1] < matrix[1, 0]


def test_pairwise_reference_matches_torch_pairwise_distance() -> None:
    import torch

    visual = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    audio = np.array([[0.0, 0.0], [1.0, 3.0], [3.0, 5.0]], dtype=np.float64)
    reference = pairwise_distance_reference(visual, audio, vshift=1)
    visual_tensor = torch.as_tensor(visual, dtype=torch.float32)
    padded_tensor = torch.nn.functional.pad(torch.as_tensor(audio, dtype=torch.float32), (0, 0, 1, 1))
    expected = np.asarray([
        torch.nn.functional.pairwise_distance(
            visual_tensor[[index]].repeat(3, 1), padded_tensor[index : index + 3]
        ).detach().numpy()
        for index in range(len(visual))
    ])
    assert np.allclose(reference, expected, atol=1e-5, rtol=0.0)
