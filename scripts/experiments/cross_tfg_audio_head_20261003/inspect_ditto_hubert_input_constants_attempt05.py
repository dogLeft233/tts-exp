#!/usr/bin/env python3
"""Read only the two ONNX constants on Ditto HuBERT's raw waveform input path.

This bounded protobuf reader checks the model path and its prior qualified
deployment report, then reads GraphProto node/initializer metadata and payload
bytes only for TensorProto producers consumed by Sub(input_values, ...) and the
next Div, whether stored as graph initializers or Constant-node `value` attrs.
It never constructs ONNX Runtime, reads audio, or evaluates the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
from pathlib import Path
from typing import Any, BinaryIO

MODEL_BYTES = 1_460_740_880
MODEL_SHA256 = "04cd6c7c388aaf99cdea8d4599cefede9930e60d6051abd5a33c791aeb931697"
PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
REPORT_SHA256 = "866fc656d69caef7203bc2a3d3c61e12046353c4c3515722fd8ad73c9d16170c"
MODEL_PATH = "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt03/models/ditto_pytorch/ditto_pytorch/aux_models/hubert_streaming_fix_kv.onnx"
REPORT_PATH = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/job_control/model_deploy_attempt18/ditto_portrait_render_attempt03.json"
TARGETS = {"onnx::Sub_423", "onnx::Div_425"}
MAX_PAYLOAD_BYTES = 4096


def varint(f: BinaryIO, end: int) -> int:
    value = shift = 0
    while f.tell() < end:
        b = f.read(1)
        if not b:
            break
        n = b[0]
        value |= (n & 0x7F) << shift
        if n < 0x80:
            return value
        shift += 7
        if shift > 70:
            raise ValueError("invalid protobuf varint")
    raise ValueError("truncated protobuf varint")


def span(f: BinaryIO, end: int) -> tuple[int, int]:
    size = varint(f, end)
    start, stop = f.tell(), f.tell() + size
    if stop > end:
        raise ValueError("protobuf field exceeds parent")
    return start, stop


def skip(f: BinaryIO, wire: int, end: int) -> None:
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


def text(f: BinaryIO, start: int, stop: int) -> str:
    if stop - start > 65536:
        raise ValueError("unexpectedly large protobuf text")
    f.seek(start)
    return f.read(stop - start).decode("utf-8", errors="replace")


def model_graph_span(f: BinaryIO, size: int) -> tuple[int, int]:
    f.seek(0)
    found = None
    while f.tell() < size:
        tag = varint(f, size)
        number, wire = tag >> 3, tag & 7
        if number == 7 and wire == 2:
            current = span(f, size)
            if found is not None:
                raise ValueError("multiple ModelProto graph fields")
            found = current
            # GraphProto is length-delimited; continue at the next top-level field.
            f.seek(current[1])
        else:
            skip(f, wire, size)
    if found is None:
        raise ValueError("ModelProto graph field missing")
    return found


def tensor_meta(f: BinaryIO, start: int, stop: int) -> dict[str, Any]:
    out: dict[str, Any] = {"dims": []}
    f.seek(start)
    while f.tell() < stop:
        tag = varint(f, stop)
        number, wire = tag >> 3, tag & 7
        if number == 1 and wire == 0:
            out["dims"].append(varint(f, stop))
        elif number == 1 and wire == 2:
            a, b = span(f, stop)
            while f.tell() < b:
                out["dims"].append(varint(f, b))
        elif number == 2 and wire == 0:
            out["data_type"] = varint(f, stop)
        elif number == 8 and wire == 2:
            a, b = span(f, stop)
            out["name"] = text(f, a, b)
        elif number == 14 and wire == 0:
            out["data_location"] = varint(f, stop)
        else:
            skip(f, wire, stop)
    return out


def node_meta(f: BinaryIO, start: int, stop: int, index: int) -> dict[str, Any]:
    out: dict[str, Any] = {"index": index, "inputs": [], "outputs": [], "name": "", "op_type": "", "attributes": []}
    f.seek(start)
    while f.tell() < stop:
        tag = varint(f, stop)
        number, wire = tag >> 3, tag & 7
        if number in (1, 2, 3, 4) and wire == 2:
            a, b = span(f, stop)
            value = text(f, a, b)
            if number == 1:
                out["inputs"].append(value)
            elif number == 2:
                out["outputs"].append(value)
            elif number == 3:
                out["name"] = value
            else:
                out["op_type"] = value
        elif number == 5 and wire == 2:
            a, b = span(f, stop)
            out["attributes"].append(attribute_meta(f, a, b, len(out["attributes"])))
        else:
            skip(f, wire, stop)
    return out


def attribute_meta(f: BinaryIO, start: int, stop: int, index: int) -> dict[str, Any]:
    """Read AttributeProto metadata and any TensorProto in its `t` field."""
    out: dict[str, Any] = {"index": index}
    f.seek(start)
    while f.tell() < stop:
        tag = varint(f, stop)
        number, wire = tag >> 3, tag & 7
        if number == 1 and wire == 2:
            a, b = span(f, stop)
            out["name"] = text(f, a, b)
        elif number == 2 and wire == 5:
            raw = f.read(4)
            if len(raw) != 4:
                raise ValueError("truncated AttributeProto float value")
            out["scalar_float_value"] = struct.unpack("<f", raw)[0]
        elif number == 3 and wire == 0:
            value = varint(f, stop)
            out["scalar_int_value"] = value - (1 << 64) if value >= (1 << 63) else value
        elif number == 4 and wire == 2:
            a, b = span(f, stop)
            out["scalar_string_bytes"] = b - a
            if b - a <= 1024:
                out["scalar_string_value"] = text(f, a, b)
        elif number == 5 and wire == 2:
            a, b = span(f, stop)
            out["tensor_value_meta"] = tensor_meta(f, a, b)
            out["_tensor_value_span"] = [a, b]
        elif number in (2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21) and wire in (0, 1, 2, 5):
            # Retain enough information to distinguish non-TensorProto Constant
            # forms without opening large tensor/string payloads.
            if number in (20, 21) and wire == 0:
                out["attribute_type" if number == 20 else "ref_attr_present"] = varint(f, stop)
            else:
                skip(f, wire, stop)
        else:
            skip(f, wire, stop)
    return out


def graph_index(f: BinaryIO, start: int, stop: int) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int, dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    nodes: list[dict[str, Any]] = []
    init_spans: dict[str, tuple[int, int, dict[str, Any]]] = {}
    constants_by_output: dict[str, list[dict[str, Any]]] = {}
    f.seek(start)
    while f.tell() < stop:
        tag = varint(f, stop)
        number, wire = tag >> 3, tag & 7
        if wire == 2 and number in (1, 5):
            a, b = span(f, stop)
            if number == 1:
                node = node_meta(f, a, b, len(nodes))
                nodes.append(node)
                if node["op_type"] == "Constant":
                    tensor_attrs = [attr for attr in node["attributes"] if attr.get("name") == "value" and "_tensor_value_span" in attr]
                    for output in node["outputs"]:
                        constants_by_output.setdefault(output, []).append({"node": node, "tensor_attributes": tensor_attrs})
            else:
                meta = tensor_meta(f, a, b)
                if meta.get("name") in TARGETS:
                    if meta["name"] in init_spans:
                        raise ValueError("duplicate target initializer name")
                    init_spans[meta["name"]] = (a, b, meta)
        else:
            skip(f, wire, stop)
    return nodes, init_spans, constants_by_output


def decode_target(f: BinaryIO, start: int, stop: int, meta: dict[str, Any]) -> dict[str, Any]:
    out = {**meta, "payload_fields": [], "external_data_entries": [], "resolution_reasons": []}
    payloads: list[tuple[str, bytes]] = []
    f.seek(start)
    while f.tell() < stop:
        tag = varint(f, stop)
        number, wire = tag >> 3, tag & 7
        if number == 9 and wire == 2:  # raw_data
            a, b = span(f, stop)
            size = b - a
            out["payload_fields"].append({"field": "raw_data", "bytes": size})
            if size > MAX_PAYLOAD_BYTES:
                raise ValueError(f"target initializer payload exceeds bounded read: {size}")
            f.seek(a)
            payloads.append(("raw_data", f.read(size)))
        elif number in (4, 5, 7, 10, 11) and wire in (0, 1, 2, 5):
            field_names = {4: "float_data", 5: "int32_data", 7: "int64_data", 10: "double_data", 11: "uint64_data"}
            width = {4: 4, 5: 4, 7: None, 10: 8, 11: None}[number]
            chunks: list[bytes] = []
            if wire == 2:
                a, b = span(f, stop)
                size = b - a
                if size > MAX_PAYLOAD_BYTES:
                    raise ValueError(f"target initializer typed payload exceeds bounded read: {size}")
                f.seek(a)
                chunks.append(f.read(size))
            elif wire == 5:
                chunks.append(f.read(4))
            elif wire == 1:
                chunks.append(f.read(8))
            else:
                # Varints are retained as unsigned integers; target inputs are expected to be float tensors.
                value = varint(f, stop)
                out.setdefault("typed_varints", {}).setdefault(field_names[number], []).append(value)
            if chunks:
                payloads.extend((field_names[number], chunk) for chunk in chunks)
                out["payload_fields"].append({"field": field_names[number], "bytes": sum(map(len, chunks))})
        elif number == 13 and wire == 2:  # external_data entry
            a, b = span(f, stop)
            entry = {}
            f.seek(a)
            while f.tell() < b:
                t = varint(f, b)
                n, w = t >> 3, t & 7
                if n in (1, 2) and w == 2:
                    x, y = span(f, b)
                    entry["key" if n == 1 else "value"] = text(f, x, y)
                else:
                    skip(f, w, b)
            out["external_data_entries"].append(entry)
        else:
            skip(f, wire, stop)

    values: list[Any] = []
    type_code = meta.get("data_type")
    fmt_by_type = {1: "f", 10: "e", 11: "d", 6: "i", 7: "q", 2: "B", 3: "b", 4: "H", 5: "h"}
    for field_name, payload in payloads:
        fmt = fmt_by_type.get(type_code)
        if fmt is None:
            continue
        width = struct.calcsize("<" + fmt)
        if len(payload) % width:
            raise ValueError("target initializer payload byte count does not match dtype")
        values.extend(struct.unpack("<" + fmt * (len(payload) // width), payload))
    if "typed_varints" in out and type_code in (6, 7, 2, 3, 4, 5):
        vals = out["typed_varints"].get({6: "int32_data", 7: "int64_data", 2: "uint64_data", 3: "int32_data", 4: "int32_data", 5: "int32_data"}.get(type_code, ""), [])
        values.extend(vals)
    expected_elements = 1
    for dim in meta.get("dims", []):
        expected_elements *= dim
    out["expected_elements_from_shape"] = expected_elements
    out["decoded_values"] = values[:32]
    out["decoded_value_count"] = len(values)
    out["decoded_values_truncated"] = len(values) > 32
    out["payload_sha256"] = hashlib.sha256(b"".join(p for _, p in payloads)).hexdigest() if payloads else None
    out["payload_total_bytes"] = sum(len(p) for _, p in payloads)
    is_float = type_code in (1, 10, 11)
    finite = is_float and all(math.isfinite(float(value)) for value in values)
    out["dtype_is_floating"] = is_float
    out["all_decoded_values_finite"] = finite if is_float else None
    if len(values) != expected_elements:
        out["resolution_reasons"].append("DECODED_COUNT_DIFFERS_FROM_SHAPE")
    if expected_elements <= 0 or any(dim < 0 for dim in meta.get("dims", [])):
        out["resolution_reasons"].append("INVALID_OR_EMPTY_TENSOR_SHAPE")
    if not is_float:
        out["resolution_reasons"].append("TARGET_DTYPE_NOT_SUPPORTED_FLOAT")
    elif not finite:
        out["resolution_reasons"].append("NONFINITE_OR_UNRESOLVED_FLOAT_VALUE")
    if out["external_data_entries"] or meta.get("data_location", 0) == 1:
        out["resolution_reasons"].append("EXTERNAL_TENSOR_DATA")
    if out["decoded_values_truncated"]:
        out["resolution_reasons"].append("DECODED_VALUE_LIST_TRUNCATED")
    out["constant_exact"] = not out["resolution_reasons"]
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def analyze(model_path: Path, report_path: Path, verified_full_sha256: str) -> dict[str, Any]:
    mst = model_path.lstat()
    if not stat.S_ISREG(mst.st_mode) or model_path.is_symlink() or mst.st_size != MODEL_BYTES:
        raise ValueError("model path type/size does not match frozen identity")
    if verified_full_sha256 != MODEL_SHA256:
        raise ValueError("independent full-file hash does not match frozen model identity")
    rst = report_path.lstat()
    if not stat.S_ISREG(rst.st_mode) or report_path.is_symlink() or sha256(report_path) != REPORT_SHA256:
        raise ValueError("qualified source report does not match frozen seal")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_and_weights = report.get("source_and_weights")
    if not isinstance(source_and_weights, dict) or source_and_weights.get("source_commit") != "c3e47eee2e626500017a0556b470d6d4182f85e8":
        raise ValueError("qualified source report source commit mismatch")
    weight_records = source_and_weights.get("weights")
    if not isinstance(weight_records, dict):
        raise ValueError("qualified source report weights table missing")
    matching_records = [
        value for value in weight_records.values()
        if isinstance(value, dict) and value.get("path") == MODEL_PATH
    ]
    if len(matching_records) != 1:
        raise ValueError("qualified source report must bind exactly one matching ONNX path")
    record = matching_records[0]
    if record.get("bytes") != MODEL_BYTES or record.get("sha256") != MODEL_SHA256:
        raise ValueError("qualified source report ONNX path/size/hash mismatch")
    if report.get("status") != "PASS_ENGINEERING_ONLY":
        raise ValueError("qualified source report status mismatch")
    with model_path.open("rb") as f:
        nodes, init_spans, constants_by_output = graph_index(f, *model_graph_span(f, mst.st_size))
        sub_nodes = [n for n in nodes if n["op_type"] == "Sub" and n["inputs"][:1] == ["input_values"]]
        div_nodes = [n for n in nodes if n["op_type"] == "Div" and n["inputs"][:1] == ["onnx::Div_424"]]
        unresolved: list[str] = []
        targets: dict[str, str] = {}
        if len(sub_nodes) != 1:
            unresolved.append("RAW_INPUT_SUB_NODE_NOT_UNIQUE")
        elif len(sub_nodes[0]["inputs"]) < 2:
            unresolved.append("RAW_INPUT_SUB_NODE_MISSING_CONSTANT_INPUT")
        else:
            targets["subtract"] = sub_nodes[0]["inputs"][1]
        if len(div_nodes) != 1:
            unresolved.append("RAW_INPUT_DIV_NODE_NOT_UNIQUE")
        elif len(div_nodes[0]["inputs"]) < 2:
            unresolved.append("RAW_INPUT_DIV_NODE_MISSING_CONSTANT_INPUT")
        else:
            targets["divide"] = div_nodes[0]["inputs"][1]
        if set(targets.values()) != TARGETS:
            unresolved.append("RAW_INPUT_TARGET_NAMES_DIFFER_FROM_FROZEN_GRAPH_AUDIT")
        values: dict[str, Any] = {}
        producers: dict[str, Any] = {}

        def summarize_producer(candidate: dict[str, Any], value_name: str) -> dict[str, Any]:
            if candidate["kind"] == "graph_initializer":
                return {"kind": "graph_initializer", "tensor_name": value_name, "tensor_meta": candidate["tensor_meta"]}
            source = candidate["constant"]
            node = source["node"]
            return {
                "kind": "Constant_node_TensorProto" if source["tensor_attributes"] else "Constant_node_without_TensorProto_value",
                "node": {k: node[k] for k in ("index", "name", "op_type", "inputs", "outputs")},
                "attributes": [
                    {k: attr[k] for k in attr if k != "_tensor_value_span"}
                    for attr in node["attributes"]
                ],
            }

        for role in ("subtract", "divide"):
            name = targets.get(role)
            candidates: list[dict[str, Any]] = []
            if name in init_spans:
                candidates.append({"kind": "graph_initializer", "span": init_spans[name][:2], "tensor_meta": init_spans[name][2]})
            candidates.extend(
                {"kind": "constant_node", "constant": source}
                for source in constants_by_output.get(name, [])
            )
            if len(candidates) != 1:
                values[role] = {
                    "tensor_name": name,
                    "constant_exact": False,
                    "resolution_reasons": ["TARGET_PRODUCER_NOT_FOUND" if not candidates else "TARGET_PRODUCER_AMBIGUOUS"],
                    "candidate_producer_count": len(candidates),
                }
                producers[role] = [summarize_producer(candidate, name) for candidate in candidates]
                unresolved.append(f"{role.upper()}_PRODUCER_NOT_UNIQUE")
                continue
            candidate = candidates[0]
            producer = summarize_producer(candidate, name)
            if candidate["kind"] == "graph_initializer":
                a, b = candidate["span"]
                meta = candidate["tensor_meta"]
            else:
                source = candidate["constant"]
                tensor_attrs = source["tensor_attributes"]
                if len(tensor_attrs) != 1:
                    values[role] = {
                        "tensor_name": name,
                        "constant_exact": False,
                        "resolution_reasons": ["CONSTANT_NODE_TENSOR_VALUE_ATTRIBUTE_NOT_UNIQUE"],
                    }
                    producers[role] = producer
                    unresolved.append(f"{role.upper()}_CONSTANT_TENSOR_ATTRIBUTE_NOT_UNIQUE")
                    continue
                attr = tensor_attrs[0]
                a, b = attr["_tensor_value_span"]
                meta = attr["tensor_value_meta"]
                producer["selected_tensor_attribute"] = {k: attr[k] for k in attr if k != "_tensor_value_span"}
            try:
                decoded = decode_target(f, a, b, meta)
            except (ValueError, OSError, struct.error) as exc:
                decoded = {**meta, "constant_exact": False, "resolution_reasons": [f"TARGET_DECODE_{type(exc).__name__}"]}
            values[role] = {"tensor_name": name, **decoded}
            producers[role] = producer
            if not decoded.get("constant_exact"):
                unresolved.extend(
                    f"{role.upper()}_{reason}"
                    for reason in decoded.get("resolution_reasons", ["TARGET_CONSTANT_UNRESOLVED"])
                )
    status = "PASS_METADATA_ONLY_CONSTANTS_READ" if not unresolved and all(v.get("constant_exact") for v in values.values()) else "INCOMPLETE_TARGET_CONSTANT_UNRESOLVED"
    return {
        "schema": "DITTO_HUBERT_ONNX_RAW_INPUT_CONSTANTS_AUDIT_V2",
        "status": status,
        "unresolved_reasons": sorted(set(unresolved)),
        "scope": {
            "model_inference": False,
            "onnx_runtime_loaded": False,
            "audio_read": False,
            "model_full_rehash_performed_by_collector": False,
            "independent_full_rehash_receipt_required": True,
            "independent_full_rehash_matches_frozen_identity": True,
            "only_target_tensor_payloads_read": True,
            "max_target_payload_bytes": MAX_PAYLOAD_BYTES,
        },
        "model": {"path": str(model_path), "bytes": mst.st_size, "sha256_bound_from_qualified_report": MODEL_SHA256, "independent_full_rehash_sha256": verified_full_sha256, "independent_full_rehash_performed": True, "plan_raw_sha256": PLAN_SHA256},
        "source_report": {"path": str(report_path), "bytes": rst.st_size, "sha256": REPORT_SHA256},
        "input_path_nodes": {"subtract": sub_nodes[0] if len(sub_nodes) == 1 else None, "divide": div_nodes[0] if len(div_nodes) == 1 else None},
        "target_producers": producers,
        "input_constants": values,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    ap.add_argument("--source-report", type=Path, default=Path(REPORT_PATH))
    ap.add_argument("--verified-model-full-sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = analyze(args.model, args.source_report, args.verified_model_full_sha256)
    args.out.parent.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    args.out.write_bytes(raw)
    print(json.dumps({"status": result["status"], "out": str(args.out), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "constants": {k: v.get("decoded_values", []) for k, v in result["input_constants"].items()}}, separators=(",", ":")))


if __name__ == "__main__":
    main()
