"""Source-order regression for the native Ditto attempt02 CPU preflight."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / ".codex" / "scratch"
          / "ditto_spsa_stagea_adapter_attempt11_native_pre_render_attempt02"
          / "validate_ditto11_native_prerender_attempt02.py")
SOURCE_PIN = (49_735, "2a711ca3a42bbe6de21291845fb4bb23bb97cfdc556d53b0198c02da2d8829f9")


class DittoNativePreRenderAttempt02Tests(unittest.TestCase):
    def test_overlay_is_pinned_and_imported_before_any_overlay_attribute_read(self) -> None:
        raw = SOURCE.read_bytes()
        if SOURCE_PIN[0]:
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), SOURCE_PIN)
        tree = ast.parse(raw.decode("utf-8"), filename=str(SOURCE))
        main_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
        self.assertEqual(len(main_nodes), 1)
        main = main_nodes[0]

        def assignment_line(name: str) -> int:
            matches = []
            for node in ast.walk(main):
                if isinstance(node, ast.Assign):
                    if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                        matches.append(node.lineno)
            self.assertEqual(len(matches), 1, name)
            return matches[0]

        overlay_import_line = assignment_line("overlay")
        for pin_name in ("core_pin", "overlay_pin", "resource_pin"):
            self.assertLess(assignment_line(pin_name), overlay_import_line)
        overlay_reads = [node.lineno for node in ast.walk(main)
                         if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                         and node.value.id == "overlay"]
        self.assertTrue(overlay_reads)
        self.assertLess(overlay_import_line, min(overlay_reads))

        imports = [node for node in ast.walk(main)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "import_exact" and node.args
                   and isinstance(node.args[0], ast.Constant)
                   and node.args[0].value == "ditto11_native_pre_render_overlay"]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].lineno, overlay_import_line)


if __name__ == "__main__":
    unittest.main()
