"""One independent research-array check of the existing primitive projection."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "whisper_primitive_projection_attempt01.json"
OUTPUT = ROOT / "root_primitive_projection_qa.json"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert not OUTPUT.exists()
data = json.loads(RESULT.read_text(encoding="utf-8"))
assert sha(RESULT) == "0ee1129a7201932ee1b99ab8ea0dcdb6cc4fb44660c442c56e8ccdce1121e6d4"
checks = []
for row in data["rows"]:
    refs = row["candidate_arrays"]
    arrays = {}
    for arm, ref in refs.items():
        p = Path(ref["path"])
        assert sha(p) == ref["file_sha256"]
        arrays[arm] = np.load(p, allow_pickle=False)
    base = arrays["zero"]
    n, width, dim = base.shape
    assert width == 50 and dim == 384
    # From the pinned source: 2j+c-4 underlying time, five tuple layers.
    f = np.arange(width)
    keys = ((2*np.arange(n)[:, None] + f[None, :]//5 - 4)*5 + f[None, :]%5).ravel()
    _, first, inverse, count = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
    flat = base.reshape(-1, dim)
    assert np.array_equal(flat, flat[first][inverse])
    owned = np.zeros(n, dtype=bool)
    owned[row["effective_owned_output_rows_from_rotated_status_only"]] = True
    for arm in ("plus", "minus"):
        delta = arrays[arm].astype(np.float64) - base.astype(np.float64)
        sums = np.zeros((len(count), dim), dtype=np.float64)
        np.add.at(sums, inverse, delta.reshape(-1, dim))
        projected = (sums/count[:, None])[inverse].reshape(base.shape)
        residual = delta - projected
        energy = lambda x: float(np.sum(x*x, dtype=np.float64))
        original_e, projected_e, residual_e = energy(delta), energy(projected), energy(residual)
        affected = np.flatnonzero(np.any(projected != 0, axis=(1, 2)) & ~owned).tolist()
        values = {
            "candidate_delta_energy": original_e,
            "projected_delta_energy": projected_e,
            "residual_energy": residual_e,
            "residual_energy_ratio": residual_e/original_e,
            "projected_energy_ratio": projected_e/original_e,
            "owned_target_retained_energy_ratio": energy(projected[owned])/energy(delta[owned]),
            "owned_target_retained_norm_ratio": np.sqrt(energy(projected[owned])/energy(delta[owned])),
            "projected_nonowned_energy_fraction_of_projected": energy(projected[~owned])/projected_e,
        }
        errors = {k: abs(v-row["arms"][arm][k]) for k, v in values.items()}
        assert max(errors.values()) < 1e-9, errors
        assert affected == row["arms"][arm]["projected_nonowned_affected_frame_indices"]
        assert abs(original_e-projected_e-residual_e)/original_e < 1e-12
        assert abs(float(np.sum(projected*residual)))/original_e < 1e-12
        checks.append({"sample_id": row["sample_id"], "arm": arm, "zero_alias_bitwise": True,
                       "metrics_max_abs_error": max(errors.values()), "nonowned_frames": len(affected),
                       "projection_pythagorean_relative_error": abs(original_e-projected_e-residual_e)/original_e})
out = {"schema": "root_whisper_primitive_projection_actual_array_qa_v1", "status": "PASS",
       "result_sha256": sha(RESULT), "qa_source_sha256": sha(__file__), "checked_rows": checks,
       "scope": "Existing actual native arrays; independent vectorized gather and projection. No encoder, renderer, scorer, or implementation tests."}
with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(out, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"status": "PASS", "path": str(OUTPUT), "sha256": sha(OUTPUT), "checks": checks}))
