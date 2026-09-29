from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_ID = "tts_evidence_temporal_patch_v1"
REVISION = "20260922"
DEFAULT_PARENT_RUN = REPO_ROOT / "runs/phoneme_tfg_association_external_visual_cpu_audit_v1_20260922"
DEFAULT_BINDINGS = REPO_ROOT / "openspec/changes/audit-tts-evidence-and-temporal-representation/input-bindings.json"
MODELS = ("qwen_cloud", "qwen_local", "index_tts2", "cosyvoice2")
NATURAL = "natural"
SCIENCE_CONDITIONS = ("BASE", "COHERENT", "SCRAMBLED", "ERASE")
DIRECTIONS = ("natural_from_tts", "tts_from_natural")
VSHIFT = 15
FRAME_RATE = 25.0
SAMPLE_RATE = 16_000
MEL_RATE = 80.0
MEL_WIDTH = 16
MEL_HOP_SAMPLES = 200
DEFAULT_SEED = 20260922
DEFAULT_BOOTSTRAP_DRAWS = 20_000
MAX_NEW_RENDER_CELLS = 128
MAX_GPU_MINUTES = 120
GPU_LOCK_PATH = Path("/tmp/tts-exp-wav2lip-gpu0.lock")


class ProtocolError(RuntimeError):
    """A frozen protocol or artifact contract was violated."""


class InputMismatch(ProtocolError):
    """A parent input or cached artifact does not match its binding."""


class InsufficientSupport(ProtocolError):
    """The pre-registered cohort/support gate cannot be met."""


class ResourceBusy(ProtocolError):
    """A requested CUDA stage would collide with another process or lock."""


class ResourceUnknown(ProtocolError):
    """GPU ownership could not be established safely."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path | str) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(value)).hexdigest()


def read_json(path: Path | str) -> Any:
    target = Path(path)
    return json.loads(
        target.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def write_json(path: Path | str, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, target)
    return target


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    target = Path(path)
    rows: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
            if not isinstance(value, dict):
                raise ProtocolError(f"JSONL row {line_number} is not an object: {target}")
            rows.append(value)
    return rows


def write_csv(path: Path | str, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)
    return target


def run_id_valid(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ProtocolError(f"invalid run id: {value!r}")
    return value


def resolve_path(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def protocol(self) -> Path:
        return self.root / "00_protocol"

    @property
    def evidence(self) -> Path:
        return self.root / "01_existing_evidence"

    @property
    def visual(self) -> Path:
        return self.root / "02_visual"

    @property
    def blind(self) -> Path:
        return self.root / "03_blind"

    @property
    def patch_plan(self) -> Path:
        return self.root / "04_patch_plan"

    @property
    def latents(self) -> Path:
        return self.root / "05_latents"

    @property
    def patch_video(self) -> Path:
        return self.root / "06_patch_video"

    @property
    def patch_eval(self) -> Path:
        return self.root / "07_patch_eval"

    @property
    def analysis(self) -> Path:
        return self.root / "08_analysis"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"

    def ensure_dirs(self) -> None:
        for directory in (
            self.protocol,
            self.evidence,
            self.visual,
            self.blind,
            self.patch_plan,
            self.latents,
            self.patch_video,
            self.patch_eval,
            self.analysis,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def run_paths(run_id: str, *, repo_root: Path = REPO_ROOT) -> RunPaths:
    return RunPaths(resolve_path(Path("runs") / run_id, base=repo_root))


def load_yaml(path: Path | str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - only minimal environments
        raise ProtocolError("PyYAML is required to load the experiment config") from exc
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"config must be a mapping: {path}")
    return dict(value)


def environment_fingerprint() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _binding_path(parent_root: Path, relative: str) -> Path:
    path = (parent_root / relative).resolve()
    try:
        path.relative_to(parent_root.resolve())
    except ValueError as exc:
        raise InputMismatch(f"input binding escapes parent run: {relative}") from exc
    return path


def verify_input_bindings(parent_root: Path, bindings_path: Path = DEFAULT_BINDINGS) -> dict[str, Any]:
    bindings = read_json(bindings_path)
    if bindings.get("protocol_id") != PROTOCOL_ID:
        raise InputMismatch("input-bindings protocol_id mismatch")
    verified: list[dict[str, Any]] = []
    for item in bindings.get("files", []):
        relative = str(item.get("relative_to_parent", ""))
        expected = str(item.get("sha256", ""))
        path = _binding_path(parent_root, relative)
        if not path.is_file():
            raise InputMismatch(f"bound input is missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise InputMismatch(f"bound input hash changed: {relative}: {actual} != {expected}")
        verified.append({"relative_to_parent": relative, "path": str(path), "sha256": actual})
    if len(verified) != 5:
        raise InputMismatch(f"expected five top-level input bindings, got {len(verified)}")
    return {"bindings_path": str(Path(bindings_path).resolve()), "bindings_sha256": file_sha256(bindings_path), "files": verified}


def _source_audio_pcm_sha256(path: Path, ffmpeg: Path | None = None) -> str:
    """Hash decoded mono 16 kHz PCM without trusting a WAV container hash."""
    import wave

    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != SAMPLE_RATE:
                raise ValueError("WAV is not mono PCM16/16k")
            return bytes_sha256(handle.readframes(handle.getnframes()))
    except (wave.Error, EOFError, ValueError):
        if ffmpeg is None:
            raise InputMismatch(f"cannot decode PCM without ffmpeg: {path}")
        result = subprocess.run(
            [str(ffmpeg), "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"],
            capture_output=True,
            check=False,
            timeout=120,
        )
        if result.returncode != 0:
            raise InputMismatch(f"PCM decode failed: {path}: {result.stderr[-500:]!r}")
        return bytes_sha256(result.stdout)


def decoded_pcm_sha256(path: Path, ffmpeg: Path) -> str:
    """Decode a media file's first audio stream under the fixed PCM contract."""

    result = subprocess.run(
        [str(ffmpeg), "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"],
        capture_output=True,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        raise InputMismatch(f"media PCM decode failed: {path}: {result.stderr[-500:]!r}")
    return bytes_sha256(result.stdout)


def _ffprobe_for(ffmpeg: Path) -> Path:
    sibling = ffmpeg.with_name("ffprobe")
    if sibling.is_file():
        return sibling
    located = shutil.which("ffprobe")
    if located:
        return Path(located)
    raise InputMismatch("strict media audit requires ffprobe")


def _probe_video_contract(video: Path, receipt: Mapping[str, Any], ffmpeg: Path) -> dict[str, Any]:
    """Check the decoded stream contract without trusting container hashes."""

    ffprobe = _ffprobe_for(ffmpeg)
    stream_result = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_streams", "-of", "json", str(video)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if stream_result.returncode != 0:
        raise InputMismatch(f"ffprobe stream inspection failed: {video}")
    try:
        streams = json.loads(stream_result.stdout).get("streams", [])
    except (TypeError, ValueError) as exc:
        raise InputMismatch(f"ffprobe returned invalid JSON: {video}") from exc
    video_streams = [item for item in streams if item.get("codec_type") == "video"]
    audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
    if len(video_streams) != 1 or len(audio_streams) != 1:
        raise InputMismatch(f"parent media must have exactly one video and one audio stream: {video}")
    video_stream = video_streams[0]
    audio_stream = audio_streams[0]
    rate_text = str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/0")
    try:
        fps = float(Fraction(rate_text))
    except (ValueError, ZeroDivisionError):
        raise InputMismatch(f"video fps is not parseable: {video}: {rate_text}")
    if not math.isfinite(fps) or abs(fps - FRAME_RATE) > 0.01:
        raise InputMismatch(f"video is not {FRAME_RATE:g} fps: {video}: {fps}")
    expected_frames = receipt.get("generated_frames", receipt.get("frame_count", receipt.get("mel_chunk_count")))
    # Packet PTS are cheap to inspect and remain the mux-level timing source;
    # decoded frame count/pixels are checked independently below with the same
    # OpenCV BGR path used by the visual metrics.
    pts_result = subprocess.run(
        [str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if pts_result.returncode != 0:
        raise InputMismatch(f"ffprobe PTS inspection failed: {video}")
    try:
        pts = [float(line.strip()) for line in pts_result.stdout.splitlines() if line.strip() and line.strip() != "N/A"]
    except ValueError as exc:
        raise InputMismatch(f"video PTS is not numeric: {video}") from exc
    if not pts or any(not math.isfinite(value) for value in pts) or any(right <= left for left, right in zip(pts, pts[1:])):
        raise InputMismatch(f"decoded video PTS is missing or non-monotone: {video}")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - the real pipeline has cv2
        raise InputMismatch("strict media audit requires OpenCV") from exc
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise InputMismatch(f"decoded video cannot be opened: {video}")
    frame_count = 0
    decoded_hash = hashlib.sha256()
    decoded_shape: tuple[int, ...] | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != "uint8":
                raise InputMismatch(f"decoded video frame is malformed: {video}/{frame_count}")
            if decoded_shape is None:
                decoded_shape = tuple(int(item) for item in frame.shape)
            if tuple(int(item) for item in frame.shape) != decoded_shape:
                raise InputMismatch(f"decoded video frame shape changed: {video}/{frame_count}")
            decoded_hash.update(frame.tobytes())
            frame_count += 1
    finally:
        capture.release()
    if len(pts) != frame_count:
        raise InputMismatch(f"PTS/frame count mismatch: {video}: {len(pts)} != {frame_count}")
    if expected_frames is not None and frame_count != int(expected_frames):
        raise InputMismatch(f"decoded frame count changed: {video}: {frame_count} != {expected_frames}")
    return {
        "stream_count": {"video": len(video_streams), "audio": len(audio_streams)},
        "fps": fps,
        "frame_count": frame_count,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "decoded_frame_shape": list(decoded_shape or ()),
        "decoded_frame_sha256": decoded_hash.hexdigest(),
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name"),
        "audio_sample_rate": int(audio_stream.get("sample_rate", 0)),
        "audio_channels": int(audio_stream.get("channels", 0)),
        "pts_first": pts[0],
        "pts_last": pts[-1],
        "pts_sha256": canonical_hash(pts),
    }


def _reference_pixel_hash(path: Path) -> str:
    """Hash the decoded BGR reference pixels used by the parent generator."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - the real pipeline has cv2
        raise InputMismatch("reference frame hash requires OpenCV") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise InputMismatch(f"reference frame cannot be decoded: {path}")
    return bytes_sha256(image.tobytes())


def _cell_key(cell: Mapping[str, Any]) -> str:
    return str(cell.get("cell_key") or f"{cell.get('source_group')}::{cell.get('sample_id')}::{cell.get('arm')}")


def _require_cell_identity(cell: Mapping[str, Any]) -> None:
    for key in ("cell_key", "sample_id", "source_group", "arm"):
        if not cell.get(key):
            raise InputMismatch(f"generation cell lacks {key}: {cell}")
    if str(cell.get("tfg")) != "wav2lip":
        raise InputMismatch(f"unexpected TFG in parent cell: {cell.get('tfg')}")


def audit_parent(
    parent_root: Path,
    *,
    bindings_path: Path = DEFAULT_BINDINGS,
    ffmpeg: Path | None = None,
    strict_media: bool = True,
) -> dict[str, Any]:
    """Freeze the parent cohort and verify every reusable artifact.

    ``strict_media=False`` is intentionally available for unit fixtures only;
    the real ``prepare`` stage always uses the strict default.
    """

    parent_root = parent_root.resolve()
    binding = verify_input_bindings(parent_root, bindings_path)
    plan_path = parent_root / "00_protocol/generation_plan.json"
    blocks_path = parent_root / "00_protocol/blocks.json"
    official_path = parent_root / "07_official_syncnet/full/summary.json"
    plan = read_json(plan_path)
    blocks = read_json(blocks_path)
    official = read_json(official_path)
    cells = plan.get("cells")
    if not isinstance(cells, list):
        raise InputMismatch("generation plan has no cells list")
    if len(cells) != int(binding.get("parent_expected", {}).get("video_cells", 108)):
        raise InputMismatch(f"parent cell count is {len(cells)}, expected 108")
    keys: set[str] = set()
    group_models: dict[str, set[str]] = {}
    normalized_cells: list[dict[str, Any]] = []
    official_records = official.get("records", [])
    official_by_key = {_cell_key(row): row for row in official_records}
    if len(official_by_key) != len(official_records):
        raise InputMismatch("official summary has duplicate cell keys")
    if int(official.get("completed", -1)) != len(cells) or int(official.get("failed", -1)) != 0:
        raise InputMismatch("official full-track summary is not the required 108/108 complete cohort")

    for raw_cell in cells:
        _require_cell_identity(raw_cell)
        cell = dict(raw_cell)
        key = _cell_key(cell)
        if key in keys:
            raise InputMismatch(f"duplicate parent cell: {key}")
        keys.add(key)
        group = str(cell["source_group"])
        arm = str(cell["arm"])
        if arm != NATURAL:
            group_models.setdefault(group, set()).add(arm)
        reference = cell.get("reference")
        if not isinstance(reference, Mapping):
            raise InputMismatch(f"cell has no reference: {key}")
        visual_source = resolve_path(str(reference.get("visual_source", "")))
        if "lrs3" in {part.lower() for part in visual_source.parts}:
            raise InputMismatch(f"LRS3 visual source is forbidden: {visual_source}")
        if not visual_source.is_file():
            raise InputMismatch(f"external visual source is missing: {visual_source}")
        actual_visual_source_sha = file_sha256(visual_source)
        if reference.get("visual_source_sha256") and actual_visual_source_sha != reference["visual_source_sha256"]:
            raise InputMismatch(f"external visual source hash changed: {visual_source}")
        feature = cell.get("feature")
        arms = feature.get("arms") if isinstance(feature, Mapping) else None
        arm_info = arms.get(arm) if isinstance(arms, Mapping) else None
        if not isinstance(arm_info, Mapping):
            raise InputMismatch(f"cell lacks feature audio arm: {key}/{arm}")
        audio = resolve_path(str(arm_info.get("audio", "")))
        textgrid = resolve_path(str(arm_info.get("textgrid", "")))
        official_row = official_by_key.get(key)
        if official_row is None:
            raise InputMismatch(f"official result missing cell: {key}")
        if str(official_row.get("source_group")) != group or str(official_row.get("sample_id")) != str(cell["sample_id"]) or str(official_row.get("arm")) != arm or str(official_row.get("tfg")) != "wav2lip":
            raise InputMismatch(f"official record identity does not match generation cell: {key}")
        video = resolve_path(str(official_row.get("video", "")))
        receipt = resolve_path(str(official_row.get("receipt", "")))
        if not video.is_file() or not receipt.is_file() or not audio.is_file() or not textgrid.is_file():
            raise InputMismatch(f"parent media/input missing for {key}")
        receipt_value = read_json(receipt)
        if str(receipt_value.get("status")) != "complete":
            raise InputMismatch(f"parent receipt is not complete: {receipt}")
        if str(receipt_value.get("cell_key")) != key or str(receipt_value.get("sample_id")) != str(cell["sample_id"]) or str(receipt_value.get("source_group")) != group or str(receipt_value.get("arm")) != arm:
            raise InputMismatch(f"parent receipt identity does not match generation cell: {receipt}")
        expected_video_sha = str(official_row.get("video_sha256", ""))
        actual_video_sha = file_sha256(video)
        if expected_video_sha and actual_video_sha != expected_video_sha:
            raise InputMismatch(f"parent video hash changed: {video}")
        if receipt_value.get("output_sha256") and receipt_value["output_sha256"] != actual_video_sha:
            raise InputMismatch(f"receipt/output hash mismatch: {receipt}")
        if receipt_value.get("output") and resolve_path(str(receipt_value["output"])) != video:
            raise InputMismatch(f"receipt output path does not match official output: {receipt}")
        if official_row.get("receipt_sha256") and file_sha256(receipt) != official_row["receipt_sha256"]:
            raise InputMismatch(f"parent receipt hash changed: {receipt}")
        expected_audio_sha = str(arm_info.get("audio_sha256", ""))
        actual_audio_sha = file_sha256(audio)
        if expected_audio_sha and actual_audio_sha != expected_audio_sha:
            raise InputMismatch(f"parent audio hash changed: {audio}")
        if receipt_value.get("audio") and resolve_path(str(receipt_value["audio"])) != audio:
            raise InputMismatch(f"receipt audio path does not match frozen audio: {receipt}")
        if receipt_value.get("audio_sha256") and receipt_value["audio_sha256"] != actual_audio_sha:
            raise InputMismatch(f"receipt audio hash does not match frozen audio: {receipt}")
        actual_pcm_sha = None
        actual_video_pcm_sha = None
        if strict_media:
            actual_pcm_sha = _source_audio_pcm_sha256(audio, ffmpeg)
            expected_pcm_sha = str(arm_info.get("pcm_sha256", ""))
            if expected_pcm_sha and actual_pcm_sha != expected_pcm_sha:
                raise InputMismatch(f"source PCM hash changed: {audio}")
            if ffmpeg is None:
                raise InputMismatch("strict media audit requires ffmpeg for decoded video PCM")
            actual_video_pcm_sha = decoded_pcm_sha256(video, ffmpeg)
            if expected_pcm_sha and actual_video_pcm_sha != expected_pcm_sha:
                raise InputMismatch(f"video PCM differs from frozen recipient PCM: {video}")
        reference_path = resolve_path(str(reference.get("reference", "")))
        crop_path = resolve_path(str(reference.get("crop", "")))
        if not reference_path.is_file() or not crop_path.is_file():
            raise InputMismatch(f"reference image/crop missing: {key}")
        reference_sha = file_sha256(reference_path)
        crop_sha = file_sha256(crop_path)
        if reference.get("reference_sha256") and reference["reference_sha256"] != reference_sha:
            raise InputMismatch(f"reference image hash changed: {reference_path}")
        if reference.get("crop_sha256") and reference["crop_sha256"] != crop_sha:
            raise InputMismatch(f"reference crop hash changed: {crop_path}")
        if arm_info.get("textgrid_sha256") and file_sha256(textgrid) != arm_info["textgrid_sha256"]:
            raise InputMismatch(f"TextGrid hash changed: {textgrid}")
        if strict_media and reference.get("frame_hash"):
            if _reference_pixel_hash(reference_path) != str(reference["frame_hash"]):
                raise InputMismatch(f"decoded reference frame hash changed: {reference_path}")
        media_contract = None
        if strict_media:
            media_contract = _probe_video_contract(video, receipt_value, ffmpeg)
        model_path = receipt_value.get("checkpoint") or receipt_value.get("runtime", {}).get("checkpoint")
        model_sha = None
        if model_path:
            model = resolve_path(str(model_path))
            if not model.is_file():
                raise InputMismatch(f"Wav2Lip checkpoint missing: {model}")
            model_sha = file_sha256(model)
            if receipt_value.get("checkpoint_sha256") and receipt_value["checkpoint_sha256"] != model_sha:
                raise InputMismatch(f"checkpoint hash changed: {model}")
        normalized_cells.append(
            {
                "cell_key": key,
                "block_id": str(cell.get("block_id")),
                "sample_id": str(cell["sample_id"]),
                "source_group": group,
                "arm": arm,
                "audio": str(audio),
                "audio_sha256": actual_audio_sha,
                "audio_pcm_sha256": actual_pcm_sha or arm_info.get("pcm_sha256"),
                "video_pcm_sha256": actual_video_pcm_sha,
                "textgrid": str(textgrid),
                "textgrid_sha256": file_sha256(textgrid),
                "video": str(video),
                "video_sha256": actual_video_sha,
                "receipt": str(receipt),
                "receipt_sha256": file_sha256(receipt),
                "official": {
                    "confidence": official_row.get("sync_c"),
                    "min_dist": official_row.get("sync_d"),
                    "av_offset": official_row.get("av_offset"),
                    "min_track": official_row.get("min_track"),
                    "scorer": official_row.get("scorer"),
                },
                "reference": {
                    "visual_source": str(visual_source),
                    "visual_source_sha256": actual_visual_source_sha,
                    "reference": str(reference_path),
                    "reference_sha256": reference_sha,
                    "crop": str(crop_path),
                    "crop_sha256": crop_sha,
                    "box_top_bottom_left_right": list(reference.get("box_top_bottom_left_right", [])),
                    "frame_hash": reference.get("frame_hash"),
                },
                "media_contract": media_contract,
                "checkpoint": str(resolve_path(str(model_path))) if model_path else None,
                "checkpoint_sha256": model_sha or receipt_value.get("checkpoint_sha256"),
                "cell": cell,
            }
        )
    groups = sorted({row["source_group"] for row in normalized_cells})
    if len(groups) != int(binding.get("parent_expected", {}).get("source_groups", 36)):
        raise InputMismatch(f"source-group count is {len(groups)}, expected 36")
    pair_count = sum(1 for row in normalized_cells if row["arm"] != NATURAL)
    if pair_count != int(binding.get("parent_expected", {}).get("natural_tts_pairs", 72)):
        raise InputMismatch(f"natural/TTS pair count is {pair_count}, expected 72")
    inventory = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "revision": REVISION,
        "status": "complete",
        "parent_root": str(parent_root),
        "parent_bindings": binding,
        "parent_protocol_sha256": file_sha256(parent_root / "00_protocol/protocol.json"),
        "generation_plan_sha256": file_sha256(plan_path),
        "blocks_sha256": file_sha256(blocks_path),
        "official_summary_sha256": file_sha256(official_path),
        "source_groups": groups,
        "source_group_count": len(groups),
        "video_cells": len(normalized_cells),
        "natural_tts_pairs": pair_count,
        "group_models": {key: sorted(value) for key, value in sorted(group_models.items())},
        "model_sha256": sorted({row["checkpoint_sha256"] for row in normalized_cells if row["checkpoint_sha256"]}),
        "cells": normalized_cells,
    }
    inventory["artifact_sha256"] = canonical_hash({key: value for key, value in inventory.items() if key != "artifact_sha256"})
    return inventory


def cache_key(*, inputs: Mapping[str, Any], parameters: Mapping[str, Any], code: Mapping[str, Any], environment: Mapping[str, Any]) -> str:
    return canonical_hash({"inputs": dict(inputs), "parameters": dict(parameters), "code": dict(code), "environment": dict(environment)})


def _group_rows(inventory: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in inventory.get("cells", []):
        groups.setdefault(str(row["source_group"]), {})[str(row["arm"])] = dict(row)
    return groups


def _stable_group_order(groups: Iterable[str], seed: int) -> list[str]:
    return sorted(groups, key=lambda group: (hashlib.sha256(f"{seed}|{group}".encode()).hexdigest(), group))


def _assignment_search(
    group_order: Sequence[str],
    eligible: Mapping[str, set[str]],
    quotas: Mapping[str, int],
    *,
    forbidden: set[str] | None = None,
) -> dict[str, str] | None:
    """Find a deterministic disjoint assignment of groups to model quotas."""

    forbidden = set(forbidden or set())
    models = tuple(sorted(quotas))
    target = sum(int(quotas[model]) for model in models)
    assignment: dict[str, str] = {}
    counts = {model: 0 for model in models}

    def available(index: int, model: str) -> int:
        return sum(1 for group in group_order[index:] if group not in forbidden and model in eligible.get(group, set()))

    def visit(index: int) -> bool:
        if len(assignment) == target:
            return all(counts[model] == int(quotas[model]) for model in models)
        if index >= len(group_order) or len(group_order) - index < target - len(assignment):
            return False
        for model in models:
            if counts[model] < int(quotas[model]) and available(index, model) < int(quotas[model]) - counts[model]:
                return False
        group = group_order[index]
        if group not in forbidden:
            for model in sorted(eligible.get(group, set())):
                if model not in counts or counts[model] >= int(quotas[model]):
                    continue
                assignment[group] = model
                counts[model] += 1
                if visit(index + 1):
                    return True
                counts[model] -= 1
                del assignment[group]
        return visit(index + 1)

    return dict(assignment) if visit(0) else None


def select_patch_cohort(
    inventory: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    technical_per_model: int = 1,
    science_per_model: int = 3,
    support_checker: Callable[[Mapping[str, Any], str], bool] | None = None,
) -> dict[str, Any]:
    """Select technical/science groups without consulting outcome fields.

    Only the frozen input rows and the optional strict support checker are
    inspected.  In particular, no score, gain, silhouette, or rating field is
    read by this function.
    """

    groups = _group_rows(inventory)
    group_ids = _stable_group_order(groups, seed)
    technical_eligible: dict[str, set[str]] = {}
    science_eligible: dict[str, set[str]] = {}
    for group, arms in groups.items():
        for model in MODELS:
            pair = arms.get(model)
            if "natural" not in arms or pair is None:
                continue
            technical_eligible.setdefault(group, set()).add(model)
            if support_checker is None or bool(support_checker({"source_group": group, "arms": arms}, model)):
                science_eligible.setdefault(group, set()).add(model)
    technical = _assignment_search(group_ids, technical_eligible, {model: technical_per_model for model in MODELS})
    if technical is None:
        raise InsufficientSupport("cannot allocate one technical group per TTS model")
    science = _assignment_search(
        group_ids,
        science_eligible,
        {model: science_per_model for model in MODELS},
        forbidden=set(technical),
    )
    if science is None:
        raise InsufficientSupport("cannot allocate three science groups per TTS model")
    rows: list[dict[str, Any]] = []
    for stage, assignment in (("technical", technical), ("science", science)):
        for group in group_ids:
            if group not in assignment:
                continue
            model = assignment[group]
            arms = groups[group]
            rows.append(
                {
                    "stage": stage,
                    "source_group": group,
                    "tts_arm": model,
                    "sample_id": arms[model]["sample_id"],
                    "natural_cell_key": arms[NATURAL]["cell_key"],
                    "tts_cell_key": arms[model]["cell_key"],
                    "direction_keys": {
                        "natural_from_tts": [arms[NATURAL]["cell_key"], arms[model]["cell_key"]],
                        "tts_from_natural": [arms[model]["cell_key"], arms[NATURAL]["cell_key"]],
                    },
                }
            )
    if len({row["source_group"] for row in rows}) != technical_per_model * len(MODELS) + science_per_model * len(MODELS):
        raise InsufficientSupport("technical/science source groups overlap")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "seed": int(seed),
        "status": "complete",
        "technical_source_groups": technical_per_model * len(MODELS),
        "scientific_source_groups": science_per_model * len(MODELS),
        "rows": rows,
        "technical_assignment": technical,
        "science_assignment": science,
        "selection_inputs": "frozen_input_availability_and_strict_support_only",
        "artifact_sha256": None,
    }


def finalize_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["artifact_sha256"] = canonical_hash({key: item for key, item in result.items() if key != "artifact_sha256"})
    return result


def _parse_csv_rows(text: str) -> list[dict[str, str]]:
    reader = csv.reader(line for line in text.splitlines() if line.strip())
    rows: list[dict[str, str]] = []
    for fields in reader:
        if len(fields) < 1:
            continue
        rows.append({"pid": fields[0].strip(), "process_name": fields[1].strip() if len(fields) > 1 else "", "used_memory_mib": fields[2].strip() if len(fields) > 2 else ""})
    return rows


def _run_nvidia(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), capture_output=True, text=True, check=False, timeout=5)


def query_gpu_state(*, target_uuid: str | None = None, runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run_nvidia) -> dict[str, Any]:
    """Return a conservative GPU/process snapshot; any unknown state raises."""

    gpu_command = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    process_command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    pmon_command = ["nvidia-smi", "pmon", "-c", "1", "-s", "um"]
    try:
        gpu_result = runner(gpu_command)
        process_result = runner(process_command)
        pmon_result = runner(pmon_command)
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        raise ResourceUnknown(f"nvidia-smi query failed: {exc}") from exc
    if any(result.returncode != 0 for result in (gpu_result, process_result, pmon_result)):
        raise ResourceUnknown("nvidia-smi did not provide a complete process/device snapshot")
    gpus: list[dict[str, str]] = []
    for line in gpu_result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 4 or not fields[1]:
            raise ResourceUnknown(f"unparseable GPU row: {line!r}")
        gpus.append({"index": fields[0], "uuid": fields[1], "memory_used_mib": fields[2], "utilization_gpu": fields[3]})
    if not gpus:
        raise ResourceUnknown("nvidia-smi reported no visible GPUs")
    if target_uuid is not None and target_uuid not in {row["uuid"] for row in gpus}:
        raise ResourceUnknown(f"target GPU UUID is not visible: {target_uuid}")
    process_rows = _parse_csv_rows(process_result.stdout)
    seen = {row["pid"] for row in process_rows if row.get("pid", "").isdigit()}
    for line in pmon_result.stdout.splitlines():
        fields = line.split()
        if not fields or fields[0] == "#":
            continue
        # ``nvidia-smi pmon`` starts with GPU index, then PID; tolerate
        # compact test/driver output that presents PID as the first field.
        pid = fields[1] if len(fields) > 1 and fields[1].isdigit() else fields[0]
        if not pid.isdigit() or pid == "-" or pid in seen:
            continue
        process_rows.append({"pid": pid, "process_name": fields[1] if len(fields) > 1 else "pmon", "used_memory_mib": "unknown"})
        seen.add(pid)
    return {"status": "known", "target_uuid": target_uuid, "gpus": gpus, "processes": process_rows, "checked_at_epoch": time.time()}


def assert_gpu_available(
    *,
    target_uuid: str | None = None,
    allow_pids: set[int] | None = None,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run_nvidia,
) -> dict[str, Any]:
    """Recheck ownership while the project lease is held, fail-closed."""

    snapshot = query_gpu_state(target_uuid=target_uuid, runner=runner)
    allowed = {str(os.getpid())} | {str(pid) for pid in (allow_pids or set())}
    blocking = [row for row in snapshot["processes"] if str(row.get("pid")) not in allowed]
    if blocking:
        raise ResourceBusy(f"CUDA resource became busy: {blocking}")
    return snapshot


@contextlib.contextmanager
def gpu_lease(
    path: Path = GPU_LOCK_PATH,
    *,
    target_uuid: str | None = None,
    allow_pids: set[int] | None = None,
    receipt_path: Path | None = None,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run_nvidia,
) -> Iterator[dict[str, Any]]:
    """Acquire a non-blocking project lock and fail closed on GPU ambiguity."""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Linux project runtime
        raise ResourceUnknown("fcntl is unavailable; cannot coordinate GPU ownership") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if target_uuid is not None:
            # UUID selection is inherited by the worker/model initialization;
            # never silently run on whichever ordinal happens to be visible.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(target_uuid)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResourceBusy(f"GPU lease is already held: {path}") from exc
        snapshot = query_gpu_state(target_uuid=target_uuid, runner=runner)
        allowed = {str(os.getpid())} | {str(pid) for pid in (allow_pids or set())}
        blocking = [row for row in snapshot["processes"] if str(row.get("pid")) not in allowed]
        if blocking:
            raise ResourceBusy(f"CUDA resource is busy: {blocking}")
        if receipt_path is not None:
            write_json(receipt_path, {"status": "acquired", "lock": str(path), "snapshot": snapshot, "pid": os.getpid()})
        yield snapshot
    finally:
        if previous_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def verify_artifact_hash(value: Mapping[str, Any]) -> None:
    expected = value.get("artifact_sha256")
    if expected:
        actual = canonical_hash({key: item for key, item in value.items() if key != "artifact_sha256"})
        if actual != expected:
            raise InputMismatch("artifact_sha256 mismatch")
