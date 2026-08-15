#!/usr/bin/env python3
"""Generate the mfa_linear third arm for pilot utterances.

Frozen kNN-VC chain: WavLM-Large L6 features of natural and canonical TTS,
mapped onto the natural clock with mfa_linear_target (MFA tokens both
sides), vocoded by the frozen prematched HiFi-GAN, right-cropped/padded to
the exact natural sample count. 16 kHz, no loudness normalization.
"""

# ============================================================================
# 背景概念
# ----------------------------------------------------------------------------
# 这个脚本生成实验的"第三臂"（third arm）。项目主问题是：TTS 语音是否比
# 自然语音产生更好的唇形同步（TFG）。为了做受控对比，我们需要把"音频内容"
# 和"声学特征来源"解耦 —— 三个臂分别对应：
#   1. natural_raw   : 原始自然语音（基线）
#   2. tts_raw       : 原始 TTS 语音（自克隆）
#   3. mfa_linear    : 把 TTS 的声学特征"搬"到自然语音的时间轴上（本脚本）
#
# 为什么需要第三臂？因为直接比较 natural_raw vs tts_raw 时，两者在时长、
# 韵律、发音节奏上都有差异，唇形同步的差异可能来自这些混淆因素。mfa_linear
# 臂用 MFA（Montreal Forced Aligner，强制对齐器）产生音素/词的时间边界，
# 把 TTS 的特征沿着这些边界线性映射到自然语音的时长上，从而隔离出
# "声学特征本身"对唇形同步的影响，而控制住"时序/节奏"这个变量。
#
# 技术链路（全部冻结，即不训练、不更新任何权重）：
#   WavLM-Large (第6层) 提取特征  →  kNN-VC 映射（mfa_linear_target）  →  HiFi-GAN 声码还原
# 这就是"kNN-VC 语音转换"的经典流程：用 k 近邻在源/目标特征间做匹配，
# 这里 "mfa_linear" 是特殊的时序对齐策略 —— 用 MFA token 边界做线性伸缩。
# ============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from knn_vc_retrieval import mfa_linear_target  # noqa: E402
from wavlm_knn_vc_adapter import KNN_VC_REVISION, SAMPLE_RATE, WavLMKNNVCAdapter  # noqa: E402

REPO = SCRIPT_DIR.parent
KNN_VC_LOCAL = REPO / "third_party/knn-vc"


# ============================================================================
# 工具函数
# ============================================================================

def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。用于记录每个产物的指纹，保证可复现性。

    分块读取（1MB/块）避免一次载入整个大文件到内存。
    """
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_natural_length(values: np.ndarray, count: int) -> tuple[np.ndarray, dict[str, Any]]:
    """把声码器输出裁剪/补零到与自然语音完全相同的采样数。

    为什么必须精确等长？因为我们最终要用 SyncNet 对"生成的视频"和"原始音频"
    做对齐评测。如果第三臂音频的长度和自然音频不一致，SyncNet 的评分窗口就会错位，
    导致唇形同步分数被长度差异污染。所以这里强制右裁剪（多了就截掉尾部）或
    右补零（少了就补静音），确保 sample count 完全一致。

    返回 (处理后的音频, 调整元信息)，元信息记录原始长度、目标长度、做了什么操作，
    便于事后审计这个臂是否"真的等长"。
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size > count:
        output = values[:count].copy()
        action = "right_crop"
    elif values.size < count:
        output = np.pad(values, (0, count - values.size)).astype(np.float32)
        action = "right_zero_pad"
    else:
        output = values.copy()
        action = "none"
    return output, {"raw_samples": int(values.size), "target_samples": int(count), "action": action}


def extend_tokens_for_feature_tail(tokens: list[Mapping[str, Any]], frame_count: int) -> tuple[list[dict[str, Any]], float]:
    """把最后一个 MFA token 的时间边界向后延伸，以覆盖特征序列的尾部。

    有个实现细节问题：WavLM 提取的特征帧数（frame_count）和 MFA token 的
    时间边界常常不完全对齐 —— 特征可能比 token 覆盖的时长略长一点。
    如果特征"超出"了最后一个 token 的结束时间，kNN 匹配时这些尾部帧就找不到
    对应的目标 token 区间。所以这里把最后一个 token 的 end_s 向后延，让
    特征帧数刚好被 token 序列完整覆盖。

    注意返回的 extension 时长是"实际延长了多少秒"，记录进元数据。
    """
    if not tokens or frame_count <= 0:
        raise ValueError("MFA tokens and feature frames must be non-empty")
    required_end = (frame_count - 0.5) * 320 / SAMPLE_RATE
    current_end = float(tokens[-1]["end_s"])
    if required_end <= current_end + 1e-7:
        return [dict(token) for token in tokens], 0.0
    extended = [dict(token) for token in tokens]
    extended[-1]["end_s"] = required_end
    extended[-1]["duration_s"] = required_end - float(extended[-1]["start_s"])
    return extended, required_end - current_end


# ============================================================================
# 核心生成函数
# ============================================================================

def generate(
    manifest: Mapping[str, Any],
    tts_meta: Mapping[str, Any],
    tokens_path: Path,
    outdir: Path,
    device: str,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """主流程：遍历 cohort 中每个样本，生成 mfa_linear 第三臂音频。

    输入三个数据源：
      1. manifest  : cohort 清单，含自然音频路径、speaker、split 等身份信息
      2. tts_meta  : TTS 元数据（canonical_16k_audio 路径、源音频哈希等）
      3. tokens_path : MFA 强制对齐产生的 token 清单（自然侧和 TTS 侧的
                       音素/词边界，带 paired_key、speaker_id 等身份字段）

    每个样本的处理流程（概念上）：
      1. 身份校验：确保 manifest、TTS 元数据、token 清单三者指向**同一个**样本
         （通过 paired_key / speaker_id / 音频哈希交叉验证）—— 这是防止
         "错位拼接"事故的关键防线，因为一旦音画不同步，整个实验结果就无意义。
      2. 特征提取：自然语音和 TTS 语音都过 WavLM-Large 第 6 层，得到特征序列。
      3. 时序对齐：用 mfa_linear_target 把 TTS 特征沿着 MFA token 边界
         线性映射到自然语音的时间轴上 —— 这一步是"搬时钟"，让 TTS 的
         声学内容按自然语音的节奏重新排布。
      4. 声码还原：冻结的 HiFi-GAN 把映射后的特征重新合成波形。
      5. 精确等长：裁剪/补零到自然语音的精确采样数。
      6. 质量自检：把输出音频再提取一次特征，和 conditioning 特征算余弦相似度，
         作为"还原保真度"的代理指标（越接近 1 说明声码器还原越忠实地保留了
         映射后的特征）。
    全部结果（含产物哈希、形状、元信息、保真度）写进 summary.json。
    """
    outdir.mkdir(parents=True, exist_ok=True)
    tokens_payload = json.loads(tokens_path.read_text(encoding="utf-8"))
    if tokens_payload.get("failures"):
        raise ValueError(f"token manifest has {len(tokens_payload['failures'])} failures")
    if manifest.get("manifest_type") == "aishell1_mfa_linear_predefined_cohort":
        if len(manifest.get("records", [])) != 25 or tokens_payload.get("samples_ok") != 25:
            raise ValueError("n=25 cohort/token manifest is incomplete")
    # 把 TTS 元数据按 sample_id 建索引，方便按样本快速查找
    tts_by_id = {}
    for row in tts_meta.get("results", {}).values():
        sample_id = str(row["sample_id"])
        if sample_id in tts_by_id:
            raise ValueError(f"duplicate TTS sample_id {sample_id}")
        tts_by_id[sample_id] = row
    # 加载冻结的 kNN-VC 适配器（WavLM 特征提取器 + 预匹配的 HiFi-GAN 声码器）
    adapter = WavLMKNNVCAdapter.load_pretrained(device=device, source=KNN_VC_LOCAL, revision=KNN_VC_REVISION)
    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for row in manifest["records"]:
        sample_id = str(row["sample_id"])
        side = tokens_payload["records"].get(sample_id)
        if side is None:
            failures.append({"sample_id": sample_id, "error": "missing tokens"})
            continue
        try:
            # ---- 身份一致性校验（关键防线）----
            if str(side.get("paired_key")) != str(row.get("paired_key")) or str(side.get("speaker_id")) != str(row.get("speaker_id")):
                raise ValueError("cohort/token identity mismatch")
            if sample_id not in tts_by_id:
                raise ValueError("missing TTS metadata")
            if str(tts_by_id[sample_id].get("paired_key")) != str(row.get("paired_key")):
                raise ValueError("cohort/TTS identity mismatch")
            tts_source_hash = tts_by_id[sample_id].get("source_audio_sha256")
            if tts_source_hash is not None and str(side["tts"].get("audio_sha256")) != str(tts_source_hash):
                raise ValueError("MFA token TTS source hash does not match TTS metadata")

            # ---- 读取音频并提取特征 ----
            natural_audio, natural_sr = sf.read(row["audio_path"], dtype="float32", always_2d=False)
            tts_audio, tts_sr = sf.read(tts_by_id[sample_id]["canonical_16k_audio"], dtype="float32", always_2d=False)
            if natural_sr != SAMPLE_RATE or tts_sr != SAMPLE_RATE:
                raise ValueError(f"sample rates {natural_sr}/{tts_sr} != {SAMPLE_RATE}")
            natural_tensor = torch.from_numpy(natural_audio).unsqueeze(0)
            tts_tensor = torch.from_numpy(tts_audio).unsqueeze(0)
            # WavLM-Large L6 提取特征序列（帧数 = 音频时长 / 320 hop，约每 20ms 一帧）
            natural_features = adapter.extract(natural_tensor)
            tts_features = adapter.extract(tts_tensor)

            # ---- 处理特征尾部超出 token 边界的情况 ----
            natural_tokens, natural_tail_extension_s = extend_tokens_for_feature_tail(
                list(side["natural"]["tokens"]), natural_features.shape[0],
            )
            tts_tokens, tts_tail_extension_s = extend_tokens_for_feature_tail(
                list(side["tts"]["tokens"]), tts_features.shape[0],
            )

            # ---- 核心：MFA 线性时序映射 ----
            # mfa_linear_target 把 TTS 特征沿着 MFA token 边界线性伸缩到
            # 自然语音的时长轴上，返回映射后的 conditioning 特征（供声码器用）
            conditioning, meta = mfa_linear_target(
                natural_features.shape[0], tts_features,
                natural_tokens, tts_tokens,
            )

            # ---- 声码还原 + 精确等长 ----
            raw = adapter.vocode(conditioning).numpy()
            output, length_adjustment = exact_natural_length(raw, natural_audio.size)
            wav_path = outdir / f"{sample_id}.wav"
            sf.write(wav_path, output, SAMPLE_RATE, subtype="FLOAT")  # FLOAT 子类型避免编码损失

            # ---- 质量自检：输出与 conditioning 的余弦相似度 ----
            encoded_output = adapter.extract(torch.from_numpy(output).unsqueeze(0))
            cosine = 1.0 - torch.nn.functional.cosine_similarity(
                encoded_output, conditioning.to(device) if encoded_output.is_cuda else conditioning, dim=-1,
            ).mean().item() if encoded_output.shape[0] > 0 else float("nan")

            results[sample_id] = {
                "sample_id": sample_id,
                "paired_key": row.get("paired_key"),
                "speaker_id": row.get("speaker_id"),
                "split": row.get("split"),
                "audio_path": str(wav_path),
                "audio_sha256": sha256_file(wav_path),
                "natural_samples": int(natural_audio.size),
                "output_samples": int(output.size),
                "exact_natural_length": bool(output.size == natural_audio.size),
                "length_adjustment": length_adjustment,
                "natural_feature_shape": list(natural_features.shape),
                "tts_feature_shape": list(tts_features.shape),
                "conditioning_shape": list(conditioning.shape),
                "mfa_linear_meta": {
                    **meta,
                    "natural_feature_tail_extension_s": natural_tail_extension_s,
                    "tts_feature_tail_extension_s": tts_tail_extension_s,
                },
                "output_to_conditioning_cosine": float(cosine),
                "peak": float(np.abs(output).max()) if output.size else 0.0,
            }
            print(f"OK {sample_id} samples={output.size} peak={results[sample_id]['peak']:.4f}", flush=True)
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})
            print(f"FAIL {sample_id}: {exc}", flush=True)

    # ---- 汇总：把整个 cohort 的结果 + 所有输入指纹写进 summary.json ----
    summary = {
        "schema_version": 1,
        "arm": "mfa_linear",
        "cohort_manifest": str(manifest_path.resolve()) if manifest_path is not None else str(manifest.get("source_manifest", "")),
        "cohort_manifest_sha256": sha256_file(manifest_path.resolve()) if manifest_path is not None else None,
        "tokens_path": str(tokens_path.resolve()),
        "tokens_sha256": sha256_file(tokens_path.resolve()),
        "model": adapter.metadata(),
        "samples_total": len(manifest["records"]),
        "samples_ok": len(results),
        "failures": failures,
        "results": results,
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps({"samples_ok": len(results), "failures": len(failures)}))
    return summary


def main(argv: list[str] | None = None) -> int:
    """命令行入口：解析参数后调用 generate，任一样本失败即返回非零退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tts-meta", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tts_meta = json.loads(args.tts_meta.read_text(encoding="utf-8"))
    summary = generate(manifest, tts_meta, args.tokens.resolve(), args.outdir.resolve(), args.device, args.manifest.resolve())
    return 0 if not summary["failures"] and summary["samples_ok"] == summary["samples_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
