#!/usr/bin/env python3
"""Read-only, streaming ONNX protobuf metadata inspection for Ditto HuBERT.

This intentionally skips TensorProto payload bytes. It verifies the pinned
1.46 GB model by streaming SHA-256, then records graph operators, names,
initializer shapes/dtypes, and scalar normalization attributes. It never
constructs an inference session and never evaluates audio.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import BinaryIO, Any


EXPECTED_BYTES = 1_460_740_880
EXPECTED_SHA256 = "04cd6c7c388aaf99cdea8d4599cefede9930e60d6051abd5a33c791aeb931697"
EXPECTED_PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
EXPECTED_MODEL_RELATIVE_PATH = (
    "ditto_pytorch/aux_models/hubert_streaming_fix_kv.onnx"
)


def varint(f: BinaryIO, end: int) -> int:
    value = 0
    shift = 0
    while True:
        if f.tell() >= end:
            raise ValueError("truncated protobuf varint")
        b = f.read(1)
        if not b:
            raise ValueError("unexpected EOF in protobuf varint")
        n = b[0]
        value |= (n & 0x7F) << shift
        if n < 0x80:
            return value
        shift += 7
        if shift > 70:
            raise ValueError("invalid protobuf varint")


def span(f: BinaryIO, end: int) -> tuple[int, int]:
    size = varint(f, end)
    start = f.tell()
    stop = start + size
    if stop > end:
        raise ValueError("protobuf length-delimited field exceeds parent")
    return start, stop


def skip_wire(f: BinaryIO, wire: int, end: int) -> None:
    if wire == 0:
        varint(f, end)
    elif wire == 1:
        f.seek(8, os.SEEK_CUR)
    elif wire == 2:
        _, stop = span(f, end)
        f.seek(stop)
    elif wire == 5:
        f.seek(4, os.SEEK_CUR)
    else:
        raise ValueError(f"unsupported protobuf wire type {wire}")
    if f.tell() > end:
        raise ValueError("protobuf field exceeds parent")


def read_text(f: BinaryIO, start: int, stop: int, cap: int = 1_000_000) -> str:
    size = stop - start
    if size > cap:
        raise ValueError(f"unexpectedly large metadata string ({size} bytes)")
    f.seek(start)
    return f.read(size).decode("utf-8", errors="replace")


def parse_model_header(f: BinaryIO, end: int) -> dict[str, Any]:
    out: dict[str, Any] = {"opsets": []}
    f.seek(0)
    while f.tell() < end:
        tag = varint(f, end)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value = varint(f, end)
            if number == 1:
                out["ir_version"] = value
            elif number == 5:
                out["model_version"] = value
        elif wire == 2:
            start, stop = span(f, end)
            if number in (2, 3, 4, 6):
                key = {2: "producer_name", 3: "producer_version", 4: "domain", 6: "doc_string"}[number]
                out[key] = read_text(f, start, stop, cap=4096)
            elif number == 7:
                out["graph_span"] = [start, stop]
            elif number == 8:
                out["opsets"].append(parse_opset(f, start, stop))
            else:
                f.seek(stop)
        elif wire in (1, 5):
            skip_wire(f, wire, end)
        else:
            raise ValueError(f"unsupported ModelProto wire type {wire}")
    return out


def parse_opset(f: BinaryIO, start: int, stop: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    f.seek(start)
    while f.tell() < stop:
        tag = varint(f, stop)
        number, wire = tag >> 3, tag & 7
        if wire == 2:
            a, b = span(f, stop)
            if number == 1:
                out["domain"] = read_text(f, a, b, cap=256)
            else:
                f.seek(b)
        elif wire == 0:
            value = varint(f, stop)
            if number == 2:
                out["version"] = value
        else:
            skip_wire(f, wire, stop)
    return out


def parse_packed_varints(f: BinaryIO, start: int, stop: int) -> list[int]:
    f.seek(start)
    values: list[int] = []
    while f.tell() < stop:
        values.append(varint(f, stop))
    return values


def parse_tensor_meta(f: BinaryIO, start: int, stop: int) -> dict[str, Any]:
    out: dict[str, Any] = {"dims": []}
    f.seek(start)
    while f.tell() < stop:
        tag = varint(f, stop)
        number, wire = tag >> 3, tag & 7
        if number == 1 and wire == 0:
            out["dims"].append(varint(f, stop))
        elif number == 1 and wire == 2:
            a, b = span(f, stop)
            out["dims"].extend(parse_packed_varints(f, a, b))
        elif number == 2 and wire == 0:
            out["data_type"] = varint(f, stop)
        elif number == 8 and wire == 2:
            a, b = span(f, stop)
            out["name"] = read_text(f, a, b, cap=4096)
        elif wire in (0, 1, 2, 5):
            skip_wire(f, wire, stop)
        else:
            raise ValueError(f"unsupported TensorProto wire type {wire}")
    return out


def parse_attribute(f: BinaryIO, start: int, stop: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    f.seek(start)
    while f.tell() < stop:
        tag = varint(f, stop)
        number, wire = tag >> 3, tag & 7
        if number == 1 and wire == 2:
            a, b = span(f, stop)
            out["name"] = read_text(f, a, b, cap=256)
        elif number == 2 and wire == 5:
            import struct
            out["f"] = struct.unpack("<f", f.read(4))[0]
        elif number == 3 and wire == 0:
            out["i"] = varint(f, stop)
        elif number == 20 and wire == 0:
            out["type"] = varint(f, stop)
        elif number == 5 and wire == 2:
            a, b = span(f, stop)
            out["tensor_meta"] = parse_tensor_meta(f, a, b)
        elif wire in (0, 1, 2, 5):
            skip_wire(f, wire, stop)
        else:
            raise ValueError(f"unsupported AttributeProto wire type {wire}")
    return out


def parse_node(f: BinaryIO, start: int, stop: int, index: int) -> dict[str, Any]:
    out: dict[str, Any] = {"index": index, "inputs": [], "outputs": [], "attributes": []}
    f.seek(start)
    while f.tell() < stop:
        tag = varint(f, stop)
        number, wire = tag >> 3, tag & 7
        if number in (1, 2, 3, 4, 7) and wire == 2:
            a, b = span(f, stop)
            text = read_text(f, a, b, cap=65_536)
            if number == 1:
                out["inputs"].append(text)
            elif number == 2:
                out["outputs"].append(text)
            elif number == 3:
                out["name"] = text
            elif number == 4:
                out["op_type"] = text
            elif number == 7:
                out["domain"] = text
        elif number == 5 and wire == 2:
            a, b = span(f, stop)
            out["attributes"].append(parse_attribute(f, a, b))
        elif wire in (0, 1, 2, 5):
            skip_wire(f, wire, stop)
        else:
            raise ValueError(f"unsupported NodeProto wire type {wire}")
    out.setdefault("name", "")
    out.setdefault("op_type", "")
    out.setdefault("domain", "")
    return out


def parse_value_info_name(f: BinaryIO, start: int, stop: int) -> str:
    f.seek(start)
    while f.tell() < stop:
        tag = varint(f, stop)
        number, wire = tag >> 3, tag & 7
        if number == 1 and wire == 2:
            a, b = span(f, stop)
            return read_text(f, a, b, cap=4096)
        skip_wire(f, wire, stop)
    return ""


def parse_graph(f: BinaryIO, start: int, stop: int) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    initializers: list[dict[str, Any]] = []
    inputs: list[str] = []
    outputs: list[str] = []
    graph_name = ""
    f.seek(start)
    while f.tell() < stop:
        tag = varint(f, stop)
        number, wire = tag >> 3, tag & 7
        if wire == 2 and number in (1, 2, 5, 11, 12):
            a, b = span(f, stop)
            if number == 1:
                nodes.append(parse_node(f, a, b, len(nodes)))
            elif number == 2:
                graph_name = read_text(f, a, b, cap=4096)
            elif number == 5:
                initializers.append(parse_tensor_meta(f, a, b))
            elif number == 11:
                inputs.append(parse_value_info_name(f, a, b))
            elif number == 12:
                outputs.append(parse_value_info_name(f, a, b))
        elif wire in (0, 1, 2, 5):
            skip_wire(f, wire, stop)
        else:
            raise ValueError(f"unsupported GraphProto wire type {wire}")
    return {
        "name": graph_name,
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes,
        "initializers": initializers,
    }


def streaming_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            block = f.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


def analyze(path: Path) -> dict[str, Any]:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or path.is_symlink():
        raise ValueError("model path must be a regular non-symlink file")
    if st.st_size != EXPECTED_BYTES:
        raise ValueError(f"model size pin mismatch: {st.st_size}")
    raw_sha, byte_count = streaming_sha256(path)
    if byte_count != EXPECTED_BYTES or raw_sha != EXPECTED_SHA256:
        raise ValueError("model byte identity does not match the frozen plan")
    with path.open("rb") as f:
        header = parse_model_header(f, st.st_size)
        if "graph_span" not in header:
            raise ValueError("ONNX ModelProto graph field was not found")
        graph = parse_graph(f, *header["graph_span"])

    nodes = graph["nodes"]
    init_by_name = {i.get("name", ""): i for i in graph["initializers"]}
    counts = collections.Counter(n.get("op_type", "") for n in nodes)
    norm_labels = ("norm", "meanvariancenormalization", "instancenormalization", "groupnormalization")
    norm_nodes = [
        n for n in nodes
        if n.get("op_type", "").lower() in {
            "layernormalization", "instancenormalization", "batchnormalization",
            "groupnormalization", "meanvariancenormalization", "normalizel2",
        }
        or any(label in (n.get("name", "") + " " + n.get("op_type", "")).lower() for label in norm_labels)
    ]

    reachable = set(graph["inputs"])
    reachable_prefix = []
    for node in nodes:
        if any(value in reachable for value in node.get("inputs", [])):
            reachable_prefix.append(node)
            reachable.update(node.get("outputs", []))
    reachable_norm = [n for n in norm_nodes if n in reachable_prefix]

    norm_detail = []
    for node in norm_nodes:
        norm_detail.append({
            "index": node["index"],
            "name": node["name"],
            "op_type": node["op_type"],
            "inputs": node["inputs"],
            "outputs": node["outputs"],
            "attributes": node["attributes"],
            "initializer_inputs": [init_by_name[x] for x in node["inputs"] if x in init_by_name],
            "input_reachable_from_graph_input": node in reachable_prefix,
        })

    first_norm_idx = min((n["index"] for n in reachable_norm), default=None)
    first_norm_prefix = [n for n in reachable_prefix if first_norm_idx is not None and n["index"] <= first_norm_idx]
    return {
        "schema": "cross_tfg_ditto_hubert_onnx_graph_metadata_v1",
        "status": "METADATA_ONLY_NO_INFERENCE",
        "model": {
            "path": str(path),
            "relative_path_from_frozen_model_root": EXPECTED_MODEL_RELATIVE_PATH,
            "bytes": byte_count,
            "sha256": raw_sha,
            "frozen_plan_raw_sha256": EXPECTED_PLAN_SHA256,
            "producer_name": header.get("producer_name"),
            "producer_version": header.get("producer_version"),
            "domain": header.get("domain"),
            "ir_version": header.get("ir_version"),
            "opsets": header.get("opsets"),
        },
        "graph": {
            "name": graph["name"],
            "input_names": graph["inputs"],
            "output_names": graph["outputs"],
            "node_count": len(nodes),
            "operator_histogram": dict(sorted(counts.items())),
            "first_128_input_reachable_nodes": reachable_prefix[:128],
            "reachable_normalization_candidates": reachable_norm,
            "normalization_candidate_details": norm_detail,
            "first_reachable_normalizer_path": first_norm_prefix,
            "initializer_count": len(graph["initializers"]),
            "normalization_related_initializers": [
                item for item in graph["initializers"]
                if any(k in item.get("name", "").lower() for k in (
                    "norm", "mean", "variance", "running_var", "scale", "gamma", "beta"
                ))
            ][:512],
        },
        "scope": {
            "audio_bytes_opened": False,
            "audio_features_computed": False,
            "onnx_session_created": False,
            "model_inference": False,
            "tensor_payload_bytes_loaded_into_memory": False,
            "initializer_names_shapes_and_types_only": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    args.out.write_bytes(raw)
    print(json.dumps({"out": str(args.out), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "model_sha256": report["model"]["sha256"], "node_count": report["graph"]["node_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
