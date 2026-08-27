from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
    AlignmentError,
    build_mfa_command,
)

MFA_DICTIONARY = "english_mfa"
MFA_ACOUSTIC_MODEL = "english_mfa"
MFA_DEFAULT_ROOT = Path("/home/wjj/Documents/MFA/mfa3_runtime_20260815")
MFA_VERSION = "3.4.1"
NORMALIZATION_POLICY_ID = "english_mfa3_token_normalization_v1"
EVENT_TOKENS = frozenset({"{LG}", "{LAUGHTER}", "[LAUGHTER]"})
_UNKNOWN_EVENT_RE = re.compile(r"^[{\[].*[}\]]$")
_NUMBER_RE = re.compile(r"^(\d+)(ST|ND|RD|TH)?$")

_UNDER_20 = (
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_ORDINAL_DIRECT = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth", 30: "thirtieth", 40: "fortieth",
    50: "fiftieth", 60: "sixtieth", 70: "seventieth", 80: "eightieth",
    90: "ninetieth", 100: "one hundredth", 1000: "one thousandth",
}


def _cardinal(value: int) -> str:
    if value < 20:
        return _UNDER_20[value]
    if value < 100:
        tens, remainder = divmod(value, 10)
        return _TENS[tens] if remainder == 0 else f"{_TENS[tens]} {_UNDER_20[remainder]}"
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        prefix = f"{_UNDER_20[hundreds]} hundred"
        return prefix if remainder == 0 else f"{prefix} {_cardinal(remainder)}"
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        prefix = f"{_cardinal(thousands)} thousand"
        return prefix if remainder == 0 else f"{prefix} {_cardinal(remainder)}"
    raise AlignmentError(f"numeric token is outside normalization range: {value}")


def _ordinal(value: int) -> str:
    if value in _ORDINAL_DIRECT:
        return _ORDINAL_DIRECT[value]
    if value < 100:
        tens, remainder = divmod(value, 10)
        return f"{_TENS[tens]}th" if remainder == 0 else f"{_TENS[tens]} {_ordinal(remainder)}"
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        prefix = f"{_UNDER_20[hundreds]} hundred"
        return f"{prefix}th" if remainder == 0 else f"{prefix} {_ordinal(remainder)}"
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        prefix = f"{_cardinal(thousands)} thousand"
        return f"{prefix}th" if remainder == 0 else f"{prefix} {_ordinal(remainder)}"
    raise AlignmentError(f"ordinal token is outside normalization range: {value}")


def _numeric_replacement(token: str) -> tuple[str, str] | None:
    match = _NUMBER_RE.fullmatch(token)
    if not match:
        return None
    value = int(match.group(1))
    suffix = match.group(2)
    if suffix:
        if value <= 0:
            raise AlignmentError(f"ordinal token must be positive: {token}")
        expected_suffix = "TH" if 10 <= value % 100 <= 20 else {1: "ST", 2: "ND", 3: "RD"}.get(value % 10, "TH")
        if suffix != expected_suffix:
            raise AlignmentError(f"ordinal suffix does not match token: {token}")
        return _ordinal(value), "ordinal"
    return _cardinal(value), "number"


def normalization_events(text: str) -> list[dict[str, Any]]:
    raw = str(text).replace("’", "'").replace("‘", "'")
    events: list[dict[str, Any]] = []
    for index, token in enumerate(raw.upper().split()):
        replacement = _numeric_replacement(token)
        if replacement:
            output, kind = replacement
            events.append({"token_index": index, "input": token, "output": output, "kind": kind})
        elif token in EVENT_TOKENS:
            events.append({"token_index": index, "input": token, "output": "", "kind": "nonlexical_event_removed"})
        elif _UNKNOWN_EVENT_RE.fullmatch(token):
            raise AlignmentError(f"unsupported nonlexical annotation token: {token!r}")
    return events


def normalize_mfa3_transcript(text: str) -> str:
    raw = str(text).replace("’", "'").replace("‘", "'")
    output: list[str] = []
    for token in raw.upper().split():
        replacement = _numeric_replacement(token)
        if replacement:
            output.extend(replacement[0].upper().split())
        elif token in EVENT_TOKENS:
            continue
        elif _UNKNOWN_EVENT_RE.fullmatch(token):
            raise AlignmentError(f"unsupported nonlexical annotation token: {token!r}")
        else:
            output.append(token)
    normalized = " ".join(output)
    if not normalized:
        raise AlignmentError("transcript is empty after MFA3 normalization")
    return normalized


def build_mfa3_command(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    mfa_executable: str,
) -> list[str]:
    return build_mfa_command(
        input_dir,
        output_dir,
        dictionary=MFA_DICTIONARY,
        acoustic_model=MFA_ACOUSTIC_MODEL,
        mfa_executable=mfa_executable,
    )


def run_mfa3_alignment(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    mfa_executable: str,
    mfa_root_dir: str | Path = MFA_DEFAULT_ROOT,
    expected_sample_ids: Sequence[str],
    runner=subprocess.run,
) -> dict[str, Any]:
    expected = [str(value) for value in expected_sample_ids]
    if len(expected) != 24 or len(set(expected)) != 24:
        raise AlignmentError("MFA3 execution requires the ordered 24-record cohort")
    command = build_mfa3_command(input_dir, output_dir, mfa_executable=mfa_executable)
    executable_path = Path(mfa_executable).resolve()
    if not executable_path.is_file():
        raise FileNotFoundError(executable_path)
    environment = os.environ.copy()
    environment["MFA_ROOT_DIR"] = str(Path(mfa_root_dir).resolve())
    environment["PATH"] = os.pathsep.join([str(executable_path.parent), environment.get("PATH", "")])
    result = runner(command, check=False, capture_output=True, text=True, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"MFA3 alignment failed with exit {result.returncode}: {result.stderr[-2000:]}")
    missing = [
        f"{sample_id}.TextGrid"
        for sample_id in expected
        if not (Path(output_dir) / f"{sample_id}.TextGrid").is_file()
    ]
    if missing:
        raise AlignmentError(f"MFA3 completed without all expected TextGrids: {missing[:5]}")
    return {
        "command": command,
        "returncode": int(result.returncode),
        "mfa_root_dir": str(Path(mfa_root_dir).resolve()),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }
