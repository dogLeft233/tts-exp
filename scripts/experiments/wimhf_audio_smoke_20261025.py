"""Bounded CPU engineering pilot; no claim of scientific preference fidelity."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(p, obj):
    with Path(p).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sparse_signed(h, k=4, threshold=None):
    if threshold is not None:
        return h * (h.abs() > threshold)
    flat = h.abs().flatten()
    indices = flat.topk(k * len(h), sorted=False).indices
    mask = torch.zeros_like(flat)
    mask[indices] = 1
    return h * mask.reshape_as(h)


class SignedSAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(5632, 64, bias=False)
        self.decoder = nn.Linear(64, 5632, bias=False)
        self.normalize()

    @torch.no_grad()
    def normalize(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))

    def forward(self, x, threshold=None):
        z = sparse_signed(self.encoder(x), threshold=threshold)
        return self.decoder(z), z


def swap_label(a, b):
    return float(a) - float(b)


def run(dataset, output):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(20261025)
    rng = np.random.default_rng(20261025)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dataset / "dataset280_consumer_manifest.json"
    patch_file = dataset / "dataset280_patch_index.json"
    assert sha(manifest) == "046b4cbb1a53fc11b2455f90eb943a7cfbac57c80e6ec34039a4f00b0b865acf"
    rows = load(manifest)["rows"]
    patches = load(patch_file)
    sources = sorted({r["sample_id"] for r in rows})[:4]
    fit, hold = sources[:3], sources[3:]
    protocol = {
        "status": "ENGINEERING_PILOT_ONLY", "source_rule": "first4 sorted existing40 source IDs; no score selection",
        "sources": sources, "fit": fit, "engineering_holdout": hold,
        "all_are_formal_fit_speakers": True, "official_dev_or_confirmation_access": False,
        "input": "audio512 only; native ordered 11x512 flattened reversibly; no PCA or initial pooling",
        "manifest_sha256": sha(manifest), "patch_index_sha256": sha(patch_file),
        "script_sha256": sha(__file__), "device": "CPU", "seed": 20261025,
        "SAE": {"features": 64, "batch_topk_average": 4, "bias": False,
                "steps": 128, "batch_size": 32, "lr": .001,
                "scale": "single fit-only RMS scalar; no per-clip normalization",
                "sampling": "uniform fit source / primary family / pair / shared character, random sign",
                "inference_threshold": "fit-only absolute preactivation quantile 1-4/64"},
        "head": {"bias": False, "steps": 128, "lr": .03, "l2": .001,
                 "target": "one clip-pair label, NOT per-window labels", "minimum_abs_delta_C": .1},
        "limits": {"wall_seconds": 120, "no_GPU": True, "no_new_media": True,
                   "no_hyperparameter_search": True, "no_feature_naming_or_causal_claim": True},
        "three_seed_repeat_and_dense_baseline": "NOT_RUN_ENGINEERING_SCOPE",
    }
    save(output / "protocol.json", protocol)
    selected = {r["id"]: r for r in rows if r["sample_id"] in sources}
    assert len(selected) == 28
    arrays = {}
    for key, r in selected.items():
        assert sha(r["sequence_path"]) == r["sequence_sha256"]
        with np.load(r["sequence_path"], allow_pickle=False) as z:
            arrays[key] = torch.from_numpy(z["audio512"].copy())
        assert arrays[key].shape == (r["rows"], 512)
        assert torch.isfinite(arrays[key]).all()
    lookup = {(p["id"], p["character_index"]): p for p in patches["patches"] if p["id"] in selected}
    pairs = [p for p in patches["pair_coverage"] if p["source"] in sources]
    assert len(pairs) == 60
    material, index, eligible = [], [], []
    for p in pairs:
        start = len(material)
        for ci in p["common_character_indices"]:
            pa, pb = (lookup[key, ci] for key in (p["a_id"], p["b_id"]))
            va, vb = pa["variants"][1], pb["variants"][1]
            assert va["shift_s"] == vb["shift_s"] == 0 and va["support_ok"] and vb["support_ok"]
            xa, xb = [] , []
            for key, v, dest in ((p["a_id"], va, xa), (p["b_id"], vb, xb)):
                ids = v["patch_rows"]
                assert len(ids) == 11 and ids == list(range(ids[0], ids[0]+11))
                x = arrays[key][ids]
                assert torch.equal(x.flatten().reshape(11, 512), x)
                dest.append(x)
            delta = (xa[0]-xb[0]).flatten()
            assert torch.equal(-delta, (xb[0]-xa[0]).flatten())
            material.append(delta)
            index.append({"pair": len(eligible), "character_index": ci,
                          "a_rows": va["patch_rows"], "b_rows": vb["patch_rows"]})
        y = swap_label(selected[p["a_id"]]["score_float32"], selected[p["b_id"]]["score_float32"])
        assert y == -swap_label(selected[p["b_id"]]["score_float32"], selected[p["a_id"]]["score_float32"])
        eligible.append({**p, "begin": start, "end": len(material), "delta_C": y})
    x = torch.stack(material)
    train_pairs = [p for p in eligible if p["source"] in fit and "within" not in p["family"] and p["temporary_measurability_gate"]]
    test_pairs = [p for p in eligible if p["source"] in hold and "within" not in p["family"] and p["temporary_measurability_gate"]]
    assert len(train_pairs) == 27 and len(test_pairs) == 9
    train_indices = [i for p in train_pairs for i in range(p["begin"], p["end"])]
    test_indices = [i for p in test_pairs for i in range(p["begin"], p["end"])]
    scale = x[train_indices].square().mean().sqrt()
    assert scale > 0
    x = x / scale
    sae = SignedSAE()

    @torch.no_grad()
    def threshold():
        return float(torch.quantile(sae.encoder(x[train_indices]).abs().flatten(), 1-4/64))

    @torch.no_grad()
    def reconstruct(indices, value):
        yh, z = sae(x[indices], value)
        assert torch.isfinite(yh).all() and torch.isfinite(z).all()
        err = (yh-x[indices]).square()
        return {"mse": float(err.mean()), "zero_reconstruction_mse": float(x[indices].square().mean()),
                "active_features": float((z != 0).float().sum(1).mean()),
                "mse_by_temporal_row": err.reshape(-1, 11, 512).mean((0, 2)).tolist()}

    initial_threshold = threshold()
    initial = {"fit": reconstruct(train_indices, initial_threshold), "holdout": reconstruct(test_indices, initial_threshold)}
    opt = torch.optim.Adam(sae.parameters(), lr=.001)
    history = []
    families = ("N-L06", "N-L17", "L17-L06")
    for step in range(128):
        assert time.monotonic()-started < 120, "hard timeout"
        batch = []
        for _ in range(32):
            source, family = rng.choice(fit), rng.choice(families)
            group = [p for p in train_pairs if p["source"] == source and p["family"] == family]
            p = group[int(rng.integers(len(group)))]
            batch.append(x[int(rng.integers(p["begin"], p["end"]))] * rng.choice((-1, 1)))
        bx = torch.stack(batch)
        opt.zero_grad(set_to_none=True)
        out, _ = sae(bx)
        loss = (out-bx).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in sae.parameters())
        opt.step()
        sae.normalize()
        if step % 32 == 0 or step == 127:
            history.append({"step": step+1, "loss": float(loss.detach())})
    frozen_threshold = threshold()
    final = {"fit": reconstruct(train_indices, frozen_threshold), "holdout": reconstruct(test_indices, frozen_threshold)}
    with torch.no_grad():
        rec, codes = sae(x, frozen_threshold)
        reverse, negative = sae(-x, frozen_threshold)
        assert torch.equal(codes, -negative) and torch.equal(rec, -reverse)
        summaries = torch.stack([codes[p["begin"]:p["end"]].mean(0) for p in eligible])
    for p in sae.parameters():
        p.requires_grad_(False)
    head = nn.Linear(64, 1, bias=False)
    nn.init.zeros_(head.weight)
    opt_head = torch.optim.Adam(head.parameters(), lr=.03)
    fit_head = [i for i, p in enumerate(eligible) if p in train_pairs and abs(p["delta_C"]) >= .1]
    test_head = [i for i, p in enumerate(eligible) if p in test_pairs and abs(p["delta_C"]) >= .1]
    assert fit_head and test_head
    labels = torch.tensor([float(p["delta_C"] > 0) for p in eligible])

    @torch.no_grad()
    def head_metrics(ids):
        logits = head(summaries[ids]).flatten()
        return {"pairs": len(ids), "bce": float(nn.functional.binary_cross_entropy_with_logits(logits, labels[ids])),
                "accuracy": float(((logits > 0) == labels[ids].bool()).float().mean())}

    before_head = {"fit": head_metrics(fit_head), "holdout": head_metrics(test_head)}
    for _ in range(128):
        assert time.monotonic()-started < 120
        opt_head.zero_grad(set_to_none=True)
        logits = head(summaries[fit_head]).flatten()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels[fit_head]) + .001*head.weight.square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(head.weight.grad).all()
        opt_head.step()
    after_head = {"fit": head_metrics(fit_head), "holdout": head_metrics(test_head)}
    with torch.no_grad():
        logits = head(summaries).flatten()
        assert torch.equal(head(-summaries), -head(summaries))
    checkpoint = output / "pilot.pt"
    torch.save({"sae": sae.state_dict(), "head": head.state_dict(), "scale": float(scale),
                "threshold": frozen_threshold, "protocol_sha256": sha(output / "protocol.json")}, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded_sae, reloaded_head = SignedSAE(), nn.Linear(64, 1, bias=False)
    reloaded_sae.load_state_dict(loaded["sae"])
    reloaded_head.load_state_dict(loaded["head"])
    with torch.no_grad():
        _, reload_codes = reloaded_sae(x, loaded["threshold"])
        assert torch.equal(codes, reload_codes)
        reload_summary = torch.stack([reload_codes[p["begin"]:p["end"]].mean(0) for p in eligible])
        assert torch.equal(logits, reloaded_head(reload_summary).flatten())
    save(output / "pair_inputs.json", {"sources": sources, "pairs": eligible, "patch_index": index})
    np.savez_compressed(output / "learned_codes.npz", codes=codes.numpy(), pair_summary=summaries.numpy(),
                        pair_logits=logits.numpy())
    result = {"status": "PASS_BOUNDED_TRAIN_SAVE_RELOAD", "scientific_status": "ENGINEERING_ONLY",
              "training_authorized": True, "device": "cpu", "sources": sources,
              "fit_sources": fit, "engineering_holdout_sources": hold, "media": 28,
              "pairs": len(eligible), "local_difference_patches": len(x), "shape": list(x.shape),
              "scale_fit_only": float(scale), "threshold_fit_only": frozen_threshold,
              "sae_initial": initial, "sae_final": final, "training_history": history,
              "head_initial": before_head, "head_final": after_head,
              "direction_antisymmetry": "EXACT", "save_reload": "EXACT",
              "fit_head_pairs": len(fit_head), "holdout_head_pairs": len(test_head),
              "original_manifest_unchanged": sha(manifest) == protocol["manifest_sha256"],
              "original_patch_index_unchanged": sha(patch_file) == protocol["patch_index_sha256"],
              "elapsed_seconds": time.monotonic()-started, "FA_truth": "UNVERIFIED",
              "no_human_quality_or_causal_explanation_claim": True}
    assert result["original_manifest_unchanged"] and result["original_patch_index_unchanged"]
    save(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.output)
