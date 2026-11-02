import importlib.util
from pathlib import Path

import pytest
import os
import subprocess
import sys
import time


spec = importlib.util.spec_from_file_location("lipscontroller", Path(__file__).parents[1] / "scripts/experiments/chinese_lips_controller_20261029.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(tmp_path):
    p = tmp_path / "asset"
    p.touch()
    base = {"audio_path": str(p), "video_path": str(p), "mfcc_path": str(p)}
    return [dict(base, id="A", speaker="one", split="train", frame_count=20*3600*25),
            dict(base, id="B", speaker="two", split="val", frame_count=2*3600*25)]


def test_insufficient_data_and_speaker_leak_fail_closed(tmp_path):
    rows = fixture(tmp_path)
    assert m.validate_data(rows)["hours"] == {"train": 20., "val": 2.}
    rows[1]["speaker"] = "one"
    with pytest.raises(ValueError, match="leakage"):
        m.validate_data(rows)
    rows[1]["speaker"] = "two"
    rows[0]["frame_count"] -= 1
    with pytest.raises(ValueError, match="incomplete"):
        m.validate_data(rows)


def test_locked_test_and_missing_asset_are_not_ignored(tmp_path):
    rows = fixture(tmp_path)
    rows[1]["split"] = "test"
    with pytest.raises(ValueError, match="test locked"):
        m.validate_data(rows)
    rows[1]["split"] = "val"
    rows[0]["video_path"] = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        m.validate_data(rows)


@pytest.mark.skipif(os.name != "posix", reason="Ubuntu process-group controller")
def test_failed_stage_leader_does_not_leave_running_grandchildren(tmp_path):
    marker = tmp_path / "orphan_finished"
    grandchild_code = f"import time,pathlib;time.sleep(.8);pathlib.Path({str(marker)!r}).touch()"
    parent_code = f"import subprocess,sys;subprocess.Popen([sys.executable,'-c',{grandchild_code!r}]);sys.exit(1)"
    child = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    assert child.wait(timeout=10) == 1
    m.terminate_owned_group(child)
    time.sleep(1)
    assert not marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="Ubuntu process-group controller")
def test_failed_leader_cleans_worker_ignoring_sigterm(tmp_path):
    marker, ready = tmp_path / "orphan_finished", tmp_path / "ready"
    code = (f"import time,pathlib,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            f"pathlib.Path({str(ready)!r}).touch();time.sleep(.8);pathlib.Path({str(marker)!r}).touch()")
    parent = (f"import subprocess,sys,time,pathlib;subprocess.Popen([sys.executable,'-c',{code!r}]);"
              f"\nwhile not pathlib.Path({str(ready)!r}).exists(): time.sleep(.01)\nsys.exit(1)")
    child = subprocess.Popen([sys.executable, "-c", parent], start_new_session=True)
    assert child.wait(timeout=10) == 1
    m.terminate_owned_group(child)
    time.sleep(1)
    assert not marker.exists()
