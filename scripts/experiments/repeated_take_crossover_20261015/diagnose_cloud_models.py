"""One-shot, read-only DashScope compatible-model listing diagnostic.

The response body is inspected in memory and only filtered model IDs and
minimal status metadata are persisted. An empty filtered list is inconclusive.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "repeated_take_crossover_20261015_attempt01"
OUTPUT_PATH = REPO_ROOT / "runs" / RUN_ID / "cloud_model_availability.json"
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/models"
FILTER = re.compile(r"tts|audio|cosy|speech", re.IGNORECASE)
REQUEST_ID_HEADERS = (
    "x-dashscope-request-id",
    "x-request-id",
    "request-id",
)


def _request_id(headers: Any) -> str | None:
    for name in REQUEST_ID_HEADERS:
        value = headers.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _result(
    *,
    status: int | None,
    request_id: str | None,
    listed_count: int | None,
    model_ids: list[str],
    classification: str,
) -> dict[str, Any]:
    return {
        "http_status": status,
        "request_id": request_id,
        "listed_model_count": listed_count,
        "filtered_model_count": len(model_ids),
        "filtered_model_ids": model_ids,
        "catalog_classification": classification,
    }


def main() -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        api_key = getpass.getpass("DashScope API key (input hidden): ")
    if not api_key:
        print("No API key supplied; no request made.", file=sys.stderr)
        return 2

    try:
        response = requests.get(
            ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=(10, 30),
            allow_redirects=False,
        )
    except requests.RequestException:
        result = _result(
            status=None,
            request_id=None,
            listed_count=None,
            model_ids=[],
            classification="request_failed; model availability is undetermined",
        )
    else:
        request_id = _request_id(response.headers)
        if response.status_code != 200:
            result = _result(
                status=response.status_code,
                request_id=request_id,
                listed_count=None,
                model_ids=[],
                classification="non_200_response; model availability is undetermined",
            )
        else:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            listed = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(listed, list):
                all_ids = [
                    item["id"]
                    for item in listed
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ]
                model_ids = [model_id for model_id in all_ids if FILTER.search(model_id)]
                classification = (
                    "account_visible_compatible_model_list; filtered IDs found"
                    if model_ids
                    else "account_visible_compatible_model_list; empty filtered list does not establish unavailability"
                )
                result = _result(
                    status=response.status_code,
                    request_id=request_id,
                    listed_count=len(all_ids),
                    model_ids=model_ids,
                    classification=classification,
                )
            else:
                result = _result(
                    status=response.status_code,
                    request_id=request_id,
                    listed_count=None,
                    model_ids=[],
                    classification="unrecognized_model_list_shape; model availability is undetermined",
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
