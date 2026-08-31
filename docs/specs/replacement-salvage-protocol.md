# Replacement Salvage Protocol: Phone-Aligned Cross-Attention

**Status**: Draft (Option B last-resort validation)
**Date**: 2026-08-31
**Target**: Downstream agent (constrained prototype, rapid validation)
**Context**: Exp 30 shows replacement fails to generalize (3/12) and TTS context is not utilized (NAT-kv ≥ TTS-kv). This protocol validates whether phone-aligned cross-attention can salvage the replacement approach before abandoning it entirely.

---

## Background

**Exp 30 failure modes**:
1. **Generalization**: 20-record training → 3/12 favorable on unseen test (< 7/12 threshold)
2. **TTS context unutilized**: NAT-kv (0.4) ≥ TTS-kv (0.2), SHUF-kv (0.4) = NAT-kv
3. **Hypothesis**: Frame-level cross-attention operates on misaligned features; TTS and Natural WavLM features may be too similar in frame space

**This protocol tests**:
- **Week 1**: Are TTS features distinguishable from Natural in WavLM space? (diagnostic, go/no-go gate)
- **Week 2**: Can phone-aligned cross-attention utilize TTS context? (architecture fix)

**Success criteria**:
- Week 1: CKA < 0.7 AND linear probe acc > 0.7 → proceed to Week 2
- Week 2: Test favorable ≥ 7/12 AND TTS-kv rate − NAT-kv rate ≥ 0.2 → replacement salvaged

**Failure → pivot**: If either week fails, abandon replacement and pivot to duration control (MFA-linear on LRS3).

---

## Experiment Design

### Scope Constraints (Prototype Validation)

**DO**:
- Reuse Exp 30's infrastructure: splits, adapters, proxy/official eval, paired gate
- Add diagnostics (CKA, linear probe, attention visualization)
- Test ONE architectural change: phone-aligned cross-attention

**DO NOT**:
- Multi-task losses (reconstruction, contrastive) → adds hyperparameters, slows iteration
- New data splits → confounds comparison with Exp 30
- Ensemble/multi-head variants → premature optimization
- Production engineering (distributed training, checkpointing strategy) → prototype only

---

## Week 1: TTS Feature Diagnostic

### Goal
Quantify whether TTS features contain exploitable information in WavLM space that frame-level cross-attention failed to capture.

### Data
Use Exp 30's **Exp A split** (train 20, val 8, test 12) for consistency.

### Procedure

#### Step 1.1: Feature Extraction

For each record in train + val + test (40 total):

```python
# Load audio
natural_audio = load_audio(record["natural_path"])  # 16kHz mono
tts_audio = load_audio(record["tts_path"])

# Extract WavLM layer 6 features (same as Exp 30)
wavlm = load_wavlm_large()
natural_feats = wavlm.extract_features(natural_audio)[5]  # (T_nat, 1024)
tts_feats = wavlm.extract_features(tts_audio)[5]  # (T_tts, 1024)

# Slice to aligned segment (reuse Exp 30's alignment)
alignment = record["policy"]["alignment"]  # natural_start, natural_end, tts_start, tts_end (frames)
natural_aligned = natural_feats[alignment["natural_start"]:alignment["natural_end"]]  # (T, 1024)
tts_aligned = tts_feats[alignment["tts_start"]:alignment["tts_end"]]  # (T, 1024)

# Save
np.save(f"{record_id}_natural_wavlm_l6.npy", natural_aligned)
np.save(f"{record_id}_tts_wavlm_l6.npy", tts_aligned)
```

**Output**: `runs/diagnostic_week1/features/{record_id}_{natural,tts}_wavlm_l6.npy` (40 × 2 = 80 files)

---

#### Step 1.2: Frame-Level Similarity (CKA)

Centered Kernel Alignment between Natural and TTS features **per record**:

```python
from cka import cka_score  # use existing implementation or minCKA

results = []
for record_id in all_records:
    nat = np.load(f"{record_id}_natural_wavlm_l6.npy")  # (T, 1024)
    tts = np.load(f"{record_id}_tts_wavlm_l6.npy")      # (T, 1024)

    # CKA requires same sequence length; truncate to min(T_nat, T_tts)
    T = min(len(nat), len(tts))
    nat_trunc = nat[:T]
    tts_trunc = tts[:T]

    score = cka_score(nat_trunc, tts_trunc)  # scalar in [0, 1]
    results.append({"record_id": record_id, "cka": score})

# Aggregate
mean_cka = np.mean([r["cka"] for r in results])
std_cka = np.std([r["cka"] for r in results])
```

**Interpretation**:
- `mean_cka > 0.9` → Natural and TTS are nearly identical in WavLM space → **STOP, pivot to duration control**
- `mean_cka < 0.7` → Features are distinguishable → proceed to Step 1.3

**Output**: `runs/diagnostic_week1/cka_summary.json`:
```json
{
  "mean_cka": 0.82,
  "std_cka": 0.05,
  "per_record": [
    {"record_id": "lrs3_...", "cka": 0.84},
    ...
  ],
  "decision": "PROCEED" | "STOP_HIGH_CKA"
}
```

---

#### Step 1.3: Linear Separability (Probe Classifier)

Train a simple logistic regression to classify Natural vs TTS from WavLM features:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Prepare dataset (frame-level binary classification)
X_train, y_train = [], []
for record_id in train_records:
    nat = np.load(f"{record_id}_natural_wavlm_l6.npy")
    tts = np.load(f"{record_id}_tts_wavlm_l6.npy")
    X_train.append(nat)  # frames × 1024
    y_train.extend([0] * len(nat))  # label 0 = natural
    X_train.append(tts)
    y_train.extend([1] * len(tts))  # label 1 = TTS

X_train = np.vstack(X_train)  # (N_frames, 1024)
y_train = np.array(y_train)

# Train probe
probe = LogisticRegression(max_iter=1000, C=1.0)
probe.fit(X_train, y_train)

# Evaluate on val
X_val, y_val = []  # same procedure for val records
val_acc = probe.score(X_val, y_val)
val_auc = roc_auc_score(y_val, probe.predict_proba(X_val)[:, 1])
```

**Interpretation**:
- `val_acc < 0.6` → Features are not separable → **STOP, no exploitable difference**
- `val_acc > 0.7` → Features carry signal → proceed to Week 2

**Output**: `runs/diagnostic_week1/probe_summary.json`:
```json
{
  "train_acc": 0.73,
  "val_acc": 0.68,
  "val_auc": 0.72,
  "decision": "PROCEED" | "STOP_LOW_SEPARABILITY"
}
```

---

#### Step 1.4: Attention Weight Visualization (Optional, Retrospective)

Load Exp 30's trained TTS-kv adapter checkpoint (step 20) and visualize cross-attention:

```python
# Forward one val record through adapter
adapter.eval()
with torch.no_grad():
    output, attn_weights = adapter(natural_query, tts_kv, return_attention=True)
    # attn_weights: (batch=1, heads=8, T_q, T_kv)

# Average over heads
attn_mean = attn_weights[0].mean(dim=0).cpu().numpy()  # (T_q, T_kv)

# Visualize heatmap
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(attn_mean, aspect='auto', cmap='viridis')
plt.xlabel('TTS key position')
plt.ylabel('Natural query position')
plt.colorbar()
plt.title('Cross-Attention Weights (Exp 30 TTS-kv adapter)')
plt.savefig(f"runs/diagnostic_week1/attn_viz_{record_id}.png")
```

**Look for**:
- Diagonal pattern → adapter learned frame-to-frame correspondence
- Uniform/flat → adapter ignores TTS (attends to padding or uniform weights)
- Blocky structure → may indicate phone-level structure that frame-level attention can't capture

**Output**: `runs/diagnostic_week1/attention_patterns/` (PNG files, qualitative inspection)

---

### Week 1 Decision Gate

**Proceed to Week 2 if ALL**:
1. `mean_cka < 0.7` (features distinguishable)
2. `val_acc > 0.7` (linear separability exists)

**Stop and pivot if ANY**:
1. `mean_cka ≥ 0.9` → TTS and Natural are redundant in WavLM space
2. `val_acc < 0.6` → No exploitable difference

**Output**: `runs/diagnostic_week1/final_decision.json`:
```json
{
  "mean_cka": 0.65,
  "val_acc": 0.74,
  "decision": "PROCEED_TO_WEEK2",
  "rationale": "Features are distinguishable (CKA=0.65) and separable (acc=0.74), suggesting frame-level cross-attention failed due to alignment issues rather than feature redundancy."
}
```

---

## Week 2: Phone-Aligned Cross-Attention

### Goal
Test if phone-aligned cross-attention can utilize TTS context where frame-level failed.

### Key Architectural Change

**Exp 30 (frame-level)**:
```
Natural audio → WavLM → features (T_nat × 1024) → CrossAttention(Q=nat, KV=tts) → output (T_nat × 1024)
TTS audio     → WavLM → features (T_tts × 1024) ↗
```
- Operates at ~50 frames/sec
- Natural and TTS frames may be misaligned (phone boundaries don't match)

**Week 2 (phone-aligned)**:
```
Natural audio → WavLM → features (T_nat × 1024) → phone pooling → (N_phones × 1024)
                                                                    ↓
                                                   PhoneCrossAttention(Q=nat_phones, KV=tts_phones)
                                                                    ↓
TTS audio     → WavLM → features (T_tts × 1024) → phone pooling → (N_phones × 1024)
                                                                    ↓
                                                   upsample to frames → output (T_nat × 1024)
```
- Operates at ~10 phones/sec (typical speech)
- Phone boundaries from MFA ensure alignment
- Pooling: mean or max within each phone segment

---

### Data Preparation

#### Step 2.1: MFA Phone Alignment

Run Montreal Forced Aligner on all records in Exp A split:

```bash
# Install MFA (if not already)
conda install -c conda-forge montreal-forced-aligner

# Prepare corpus
mkdir -p runs/week2_mfa/corpus
for record_id in $(cat data/splits/exp_a_split.json | jq -r '.train[], .val[], .test[]'); do
    cp tmp/lrs3_policy_a1_200_20260828/policy_cohort/${record_id}/natural.wav \
       runs/week2_mfa/corpus/${record_id}_natural.wav
    cp tmp/lrs3_policy_a1_200_20260828/policy_cohort/${record_id}/tts.wav \
       runs/week2_mfa/corpus/${record_id}_tts.wav

    # Transcription (use existing ASR transcripts from policy_cohort)
    echo "${transcript}" > runs/week2_mfa/corpus/${record_id}_natural.txt
    echo "${transcript}" > runs/week2_mfa/corpus/${record_id}_tts.txt
done

# Run MFA
mfa align runs/week2_mfa/corpus \
    english_us_arpa english_us_arpa \
    runs/week2_mfa/alignments

# Output: runs/week2_mfa/alignments/{record_id}_{natural,tts}.TextGrid
```

Each TextGrid contains phone boundaries:
```
intervals [1]:
    xmin = 0.0
    xmax = 0.15
    text = "DH"
intervals [2]:
    xmin = 0.15
    xmax = 0.28
    text = "AH"
...
```

**Parse to JSON**:
```python
import textgrid

def parse_textgrid(path):
    tg = textgrid.TextGrid.fromFile(path)
    phones = []
    for interval in tg[0]:  # assume phone tier is first
        phones.append({
            "phone": interval.mark,
            "start": interval.minTime,
            "end": interval.maxTime
        })
    return phones

# Save per record
for record_id in all_records:
    nat_phones = parse_textgrid(f"runs/week2_mfa/alignments/{record_id}_natural.TextGrid")
    tts_phones = parse_textgrid(f"runs/week2_mfa/alignments/{record_id}_tts.TextGrid")

    json.dump({
        "record_id": record_id,
        "natural_phones": nat_phones,
        "tts_phones": tts_phones
    }, open(f"runs/week2_mfa/phone_alignments/{record_id}.json", "w"))
```

**Quality check**:
- Assert: `len(natural_phones) == len(tts_phones)` (same transcript → same phone sequence)
- If mismatch > 2 phones → flag record, use frame-level fallback

**Output**: `runs/week2_mfa/phone_alignments/{record_id}.json` (40 files)

---

### Model Architecture

#### Step 2.2: Phone-Pooling Module

```python
class PhonePooling(nn.Module):
    def __init__(self, pooling_mode="mean"):
        super().__init__()
        self.mode = pooling_mode  # "mean" or "max"

    def forward(self, features, phone_boundaries):
        """
        Args:
            features: (batch=1, T_frames, D=1024) WavLM features
            phone_boundaries: list of (start_frame, end_frame) for each phone
        Returns:
            pooled: (batch=1, N_phones, D=1024)
        """
        pooled = []
        for start, end in phone_boundaries:
            segment = features[:, start:end, :]  # (1, len, 1024)
            if self.mode == "mean":
                pooled.append(segment.mean(dim=1))  # (1, 1024)
            elif self.mode == "max":
                pooled.append(segment.max(dim=1)[0])
        return torch.stack(pooled, dim=1)  # (1, N_phones, 1024)
```

**Phone boundary conversion** (time → frame index):
```python
def time_to_frame(time_sec, sample_rate=16000, hop_length=320):
    """WavLM uses 320-sample hop → 50 fps"""
    return int(time_sec * sample_rate / hop_length)

natural_phone_frames = [
    (time_to_frame(p["start"]), time_to_frame(p["end"]))
    for p in natural_phones
]
```

---

#### Step 2.3: Phone Cross-Attention Adapter

Modify Exp 30's `CrossAttentionResidual` to operate on phone-pooled features:

```python
class PhoneCrossAttentionAdapter(nn.Module):
    def __init__(self, d_model=1024, n_heads=8):
        super().__init__()
        self.phone_pooling = PhonePooling(mode="mean")
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.upsample_proj = nn.Linear(d_model, d_model)

        # Initialize to zero-output (same as Exp 30)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.upsample_proj.weight)
        nn.init.zeros_(self.upsample_proj.bias)

    def forward(self, natural_feats, tts_feats, natural_phone_bounds, tts_phone_bounds):
        """
        Args:
            natural_feats: (1, T_nat, 1024)
            tts_feats: (1, T_tts, 1024)
            natural_phone_bounds: [(start, end), ...] N_phones entries
            tts_phone_bounds: [(start, end), ...] N_phones entries
        Returns:
            output: (1, T_nat, 1024) frame-level residual
        """
        # Pool to phone level
        nat_phones = self.phone_pooling(natural_feats, natural_phone_bounds)  # (1, N, 1024)
        tts_phones = self.phone_pooling(tts_feats, tts_phone_bounds)          # (1, N, 1024)

        # Phone-level cross-attention
        residual_phones, _ = self.cross_attn(
            query=nat_phones,
            key=tts_phones,
            value=tts_phones
        )  # (1, N, 1024)

        # Upsample to frame level (repeat each phone's output for its frame span)
        residual_frames = []
        for i, (start, end) in enumerate(natural_phone_bounds):
            phone_output = residual_phones[:, i:i+1, :]  # (1, 1, 1024)
            repeated = phone_output.repeat(1, end - start, 1)  # (1, len, 1024)
            residual_frames.append(repeated)

        residual_frames = torch.cat(residual_frames, dim=1)  # (1, T_nat, 1024)

        # Project and add to natural
        residual_frames = self.upsample_proj(residual_frames)
        output = natural_feats + residual_frames
        return output
```

**Key differences from Exp 30**:
- Operates on N_phones (~10) instead of T_frames (~200–500) → 20–50× fewer attention computations
- Phone boundaries ensure structural alignment (phone i in natural ↔ phone i in TTS)
- Upsample step is deterministic (repeat), no learnable upsampling

---

### Training

#### Step 2.4: Train Phone-Aligned Adapters

Reuse Exp 30's training loop with modified forward pass:

```python
# Reuse: loss = replacement_target_margin_v1, optimizer = AdamW(lr=1e-5), steps = 200

def train_phone_aligned_adapter(records, adapter, wavlm, hifigan, proxy_wav2lip, syncnet):
    for step in range(200):
        for record in records:
            # Load audio and phone boundaries
            natural_audio = load_audio(record["natural_path"])
            tts_audio = load_audio(record["tts_path"])
            natural_phone_bounds = load_phone_bounds(record["phone_alignment_path"], "natural")
            tts_phone_bounds = load_phone_bounds(record["phone_alignment_path"], "tts")

            # Extract WavLM features
            natural_feats = wavlm(natural_audio)  # (1, T_nat, 1024)
            tts_feats = wavlm(tts_audio)          # (1, T_tts, 1024)

            # Phone-aligned cross-attention
            replacement_feats = adapter(natural_feats, tts_feats, natural_phone_bounds, tts_phone_bounds)

            # Decode to waveform
            replacement_audio = hifigan(replacement_feats)

            # Proxy evaluation (differentiable)
            D_replacement, C_replacement = proxy_eval(replacement_audio, proxy_wav2lip, syncnet)

            # Pristine baseline (no adapter)
            pristine_feats = natural_feats  # no modification
            pristine_audio = hifigan(pristine_feats)
            D_pristine, C_pristine = proxy_eval(pristine_audio, proxy_wav2lip, syncnet)

            # Natural-driven baseline
            natural_driven_audio = natural_audio  # ground truth
            D_natural, C_natural = proxy_eval(natural_driven_audio, proxy_wav2lip, syncnet)

            # Loss (same as Exp 30)
            loss = replacement_target_margin_v1(
                D_replacement, C_replacement,
                D_pristine, C_pristine,
                D_natural, C_natural,
                replacement_audio, natural_audio  # log-mel trust region
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

        # Save checkpoints at 0, 40, 80, 120, 160, 200
        if step in [0, 40, 80, 120, 160, 200]:
            save_checkpoint(adapter, f"runs/week2_train_tts_phone/checkpoint_step_{step}.pt")
```

**Train 3 conditions** (parallel to Exp B):
1. **TTS-kv-phone**: KV = TTS phone-pooled features
2. **NAT-kv-phone**: KV = Natural phone-pooled features (self-attention control)
3. **SHUF-kv-phone**: KV = Shuffled TTS (reuse Exp B's shuffle_map)

**Output**:
- `runs/week2_train_tts_phone/checkpoint_step_{0,40,80,120,160,200}.pt`
- `runs/week2_train_nat_phone/...`
- `runs/week2_train_shuf_phone/...`

---

### Validation and Test

#### Step 2.5: Select Checkpoints (Val Set)

Identical to Exp 30's validation protocol, but using phone-aligned adapters:

```python
# For each checkpoint and each val record:
#   - Generate U (pristine), F (checkpoint)
#   - Proxy eval → D_U, C_U, D_F, C_F
#   - Paired gate: D_U - D_F > 2q AND C_F - C_U > 2q
# Select checkpoint with max favorable_rate

# Output: runs/week2_val_{tts,nat,shuf}_phone/val_summary.json
```

---

#### Step 2.6: Official Test Evaluation

Identical to Exp 30's test protocol:

```python
# For each test record:
#   - Render A (natural), U×2, F_tts×2, F_nat×2, F_shuf×2 with official Wav2Lip
#   - Score with SyncNet V2
#   - Apply paired gate (strict)
# Aggregate favorable counts per condition

# Output: runs/week2_test_phone/test_summary.json
```

**Success criteria**:
- `favorable_tts >= 7/12` (generalization threshold)
- `rate_tts - rate_nat >= 0.2` (TTS context contribution)

---

### Week 2 Decision

**Replacement salvaged if BOTH**:
1. TTS-kv-phone favorable ≥ 7/12 (better than Exp 30's 3/12)
2. rate_tts − rate_nat ≥ 0.2 (TTS context utilized, unlike Exp 30's −0.2)

**Replacement abandoned if EITHER**:
1. TTS-kv-phone favorable < 7/12 (architecture change didn't help)
2. NAT-kv-phone ≥ TTS-kv-phone (TTS still not utilized)

**Output**: `runs/week2_test_phone/final_verdict.json`:
```json
{
  "test_favorable": {
    "tts_phone": 8,
    "nat_phone": 5,
    "shuf_phone": 4
  },
  "rates": {
    "tts": 0.67,
    "nat": 0.42,
    "shuf": 0.33
  },
  "tts_nat_gap": 0.25,
  "verdict": "REPLACEMENT_SALVAGED",
  "rationale": "Phone-aligned cross-attention enables TTS context utilization (gap=0.25 > 0.2) and achieves generalization (8/12 > 7/12)."
}
```

OR:

```json
{
  "test_favorable": {
    "tts_phone": 4,
    "nat_phone": 5,
    "shuf_phone": 4
  },
  "rates": {
    "tts": 0.33,
    "nat": 0.42,
    "shuf": 0.33
  },
  "tts_nat_gap": -0.08,
  "verdict": "REPLACEMENT_ABANDONED",
  "rationale": "Phone alignment did not improve TTS context utilization (gap=-0.08 < 0.2) or generalization (4/12 < 7/12). Pivot to duration control (MFA-linear)."
}
```

---

## Implementation Constraints for Downstream Agent

### MUST DO
1. **Reuse Exp 30 infrastructure**: Same splits, same loss, same optimizer config, same eval protocol
2. **Week 1 decision gate**: Stop immediately if CKA ≥ 0.9 OR probe acc < 0.6
3. **Phone alignment quality check**: Assert `len(natural_phones) == len(tts_phones)` ± 2 tolerance
4. **Zero-output initialization**: Phone adapter must start at zero (same as Exp 30)
5. **Paired gate**: Use same strict criteria (D gain, C gain, offset, best_second_gap, V_audio=0)

### MUST NOT DO
1. **No multi-task losses**: Do not add reconstruction, contrastive, or KL terms → increases hyperparameters
2. **No new splits**: Use Exp A split (train 20, val 8, test 12) for direct comparison
3. **No ensemble**: Train single adapters, not multi-head ensembles
4. **No learnable upsampling**: Phone → frame upsampling is deterministic repeat, not learned
5. **No early stopping on train loss**: Follow checkpoint schedule (0, 40, 80, 120, 160, 200), select on val

### Error Handling
- **MFA fails on record**: Skip record, report in summary, proceed with remaining
- **Phone count mismatch > 2**: Use frame-level fallback for that record (log warning)
- **CKA/probe crashes**: Report partial results, make conservative decision (STOP if <50% records processed)

### Logging
Each step must produce a JSON summary:
- `diagnostic_week1/cka_summary.json` (Step 1.2)
- `diagnostic_week1/probe_summary.json` (Step 1.3)
- `diagnostic_week1/final_decision.json` (Week 1 gate)
- `week2_val_{tts,nat,shuf}_phone/val_summary.json` (Step 2.5)
- `week2_test_phone/test_summary.json` (Step 2.6)
- `week2_test_phone/final_verdict.json` (Week 2 decision)

---

## Deliverables

### If Replacement Salvaged (Week 2 success)
1. **Exp 31 report**: "Phone-Aligned Cross-Attention Enables TTS Context Utilization"
   - Compare Exp 30 (frame-level, 3/12) vs Exp 31 (phone-level, ≥7/12)
   - Evidence: TTS-kv gap positive (vs Exp 30's negative gap)
2. **Code**: `scripts/experiments/lrs3_syncnet_finetune/phone_aligned_replacement_train.py`
3. **Artifacts**: Checkpoints, phone alignments, attention visualizations

### If Replacement Abandoned (Week 1 or Week 2 failure)
1. **Exp 31 report**: "Replacement Audio-Head Feasibility Boundary"
   - Document why replacement failed: CKA evidence (features redundant) OR phone alignment didn't help
   - Recommendation: Pivot to duration control (MFA-linear on LRS3)
2. **Pivot plan**: Week 3 spec for MFA-linear validation (separate document)

---

## Timeline

| Week | Task | Time | Decision Point |
|------|------|------|----------------|
| 1 | Feature extraction (Step 1.1) | 0.5 day | - |
| 1 | CKA + probe (Step 1.2–1.3) | 1 day | STOP if CKA ≥ 0.9 OR acc < 0.6 |
| 1 | Attention viz (Step 1.4) | 0.5 day | - |
| 2 | MFA alignment (Step 2.1) | 1 day | - |
| 2 | Train phone adapters (Step 2.4) | 2 days | - |
| 2 | Val + test (Step 2.5–2.6) | 1 day | - |
| 2 | Decision + report | 0.5 day | SALVAGED or ABANDONED |

**Total**: 6.5 days (1.5 weeks with some overlap)

---

## Success Definition

**Minimum success** (replacement salvaged):
- Week 2 test: TTS-kv-phone ≥ 7/12 favorable
- Week 2 test: rate_tts − rate_nat ≥ 0.2

**Strong success**:
- Week 2 test: TTS-kv-phone ≥ 9/12 favorable
- Week 2 test: rate_tts − rate_nat ≥ 0.3
- Week 2 test: f_beats_a ≥ 4/12 (beats natural baseline)

**Acceptable negative result**:
- Week 1: CKA = 0.92, acc = 0.54 → "TTS and Natural are redundant in WavLM space"
- Week 2: TTS-kv-phone 4/12, NAT-kv-phone 5/12 → "Phone alignment insufficient, pivot to duration"

---

## Notes

- **Prototype mindset**: This is a rapid go/no-go validation, not production code. Prioritize clarity and reproducibility over optimization.
- **Comparison anchor**: All results must be compared against Exp 30 (frame-level) to isolate the effect of phone alignment.
- **Negative results are valid**: If Week 1 diagnostic shows CKA > 0.9, that IS a valuable result (it bounds the feasibility of replacement). Document and pivot.
- **No scope creep**: Do not add speculative improvements (e.g., learnable pooling, attention regularization) unless Week 2 shows clear TTS utilization but borderline generalization (6/12). Stick to the spec.

---

**End of spec. Downstream agent: start with Week 1 feature extraction.**
