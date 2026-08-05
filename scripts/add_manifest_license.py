from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
for record in payload.get("records", []):
    record["license"] = "Apache-2.0"
payload["dataset_license"] = "Apache-2.0"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(len(payload.get("records", [])))
