"""Integration checks for the independent audit against frozen real artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/syncnet_audio_replay20_20261024_attempt01"


class ParentAuditContract(unittest.TestCase):
    @unittest.skipUnless((RUN / "protocol.json").is_file(), "frozen run not available")
    def test_source_manifest_identity_is_not_enriched_manifest_identity(self):
        protocol = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
        source = ROOT / "runs/syncnet_preference_probe_20261024_attempt01/input_manifest.json"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), protocol["input_manifest_sha256"])
        self.assertNotEqual(hashlib.sha256((RUN / "input_manifest.json").read_bytes()).hexdigest(),
                            protocol["input_manifest_sha256"])

    @unittest.skipUnless((RUN / "items/BAC009S0002W0122_N_0/sequence.npz").is_file(),
                         "full frozen cache is remote only")
    def test_full_real_replay_contract(self):
        path = ROOT / "scripts/experiments/syncnet_audio_replay20_parent_audit_20261024.py"
        spec = importlib.util.spec_from_file_location("independent_parent_audit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.audit(RUN)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["count"], 140)
        self.assertEqual(result["windows"], 16612)
        self.assertTrue(all(x["canonical_score_exact"] for x in result["items"]))


if __name__ == "__main__":
    unittest.main()
