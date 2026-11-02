#!/usr/bin/python3
"""Bounded GPU power policy; never changes fans, clocks, ECC, or resets GPUs."""
import argparse
import csv
import json
import logging
import math
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

SMI = "/usr/bin/nvidia-smi"
FIELDS = "uuid,temperature.gpu,temperature.memory,power.draw,power.limit,power.min_limit,power.max_limit"


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite telemetry")
    return result


def query(uuid):
    output = subprocess.run(
        [SMI, "-i", uuid, "--query-gpu=" + FIELDS, "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout
    rows = list(csv.reader(output.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 7 or rows[0][0].strip() != uuid:
        raise ValueError("GPU identity or telemetry mismatch")
    values = [number(v) for v in rows[0][1:]]
    result = dict(zip(("gpu", "memory", "watts", "limit", "minimum", "maximum"), values))
    if not (0 <= result["gpu"] <= 120 and 0 <= result["memory"] <= 120):
        raise ValueError("Temperature out of range")
    return result


def validate(config):
    if config["uuid"] != "GPU-d12b3309-c299-da83-064d-a3cd619bd560":
        raise ValueError("This installation targets the verified LAN V100 only")
    if not 100 <= config["emergency_watts"] <= config["warm_watts"] <= config["night_watts"] <= config["day_watts"] <= 300:
        raise ValueError("Invalid power limits")
    if not 5 <= config["interval_seconds"] <= 30:
        raise ValueError("Invalid monitor interval")


def policy(config, telemetry, now, held_level=0, cool_count=0):
    """Use both core/HBM temperatures; release thermal guard after 4 cool samples."""
    gpu, memory = telemetry["gpu"], telemetry["memory"]
    if gpu >= 80 or memory >= 82:
        level, cool_count = 2, 0
    elif gpu >= 75 or memory >= 78:
        level, cool_count = max(held_level, 1), 0
    elif held_level:
        cool_count = cool_count + 1 if gpu <= 68 and memory <= 70 else 0
        level = 0 if cool_count >= 4 else held_level
    else:
        level, cool_count = 0, 0
    night = now.astimezone(ZoneInfo("Asia/Shanghai")).hour >= 23 or now.astimezone(ZoneInfo("Asia/Shanghai")).hour < 8
    target = config["night_watts"] if night else config["day_watts"]
    mode = "night" if night else "day"
    if level:
        target = min(target, config["emergency_watts"] if level == 2 else config["warm_watts"])
        mode += "/thermal-" + str(level)
    if not telemetry["minimum"] <= target <= telemetry["maximum"]:
        raise ValueError("Target outside actual device power bounds")
    return target, mode, level, cool_count


def set_limit(uuid, watts):
    output = subprocess.run(
        [SMI, "-i", uuid, "-pl", str(watts)],
        check=True, capture_output=True, text=True, timeout=10,
    )
    logging.info("Power change: %s", output.stdout.strip())
    actual = query(uuid)["limit"]
    if abs(actual - watts) > 0.5:
        raise RuntimeError("Power limit readback did not match requested value")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/etc/gpu-night-quiet.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--at", help="ISO time for dry-run schedule checks only")
    args = parser.parse_args()
    if args.at and not args.dry_run:
        parser.error("--at requires --dry-run")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    validate(config)
    level, cool_count, last_mode, last_log = 0, 0, None, 0.0
    # Fail closed after missing telemetry: require four cool readings before raising power.
    while True:
        try:
            telemetry = query(config["uuid"])
            now = datetime.fromisoformat(args.at) if args.at else datetime.now(ZoneInfo("Asia/Shanghai"))
            if now.tzinfo is None:
                raise ValueError("Time must include an explicit timezone")
            target, mode, level, cool_count = policy(config, telemetry, now, level, cool_count)
            if args.dry_run:
                print(json.dumps({"time": now.isoformat(), "mode": mode, "target_watts": target, "telemetry": telemetry}))
            elif abs(telemetry["limit"] - target) > 0.5:
                set_limit(config["uuid"], target)
            if mode != last_mode or time.monotonic() - last_log >= 60:
                logging.info("mode=%s target=%sW core=%sC HBM=%sC draw=%sW", mode, target, telemetry["gpu"], telemetry["memory"], telemetry["watts"])
                last_mode, last_log = mode, time.monotonic()
        except Exception:
            logging.exception("GPU telemetry/control failed; no automatic reboot or GPU reset")
            level, cool_count = 2, 0
            if not args.dry_run:
                try:
                    set_limit(config["uuid"], config["emergency_watts"])
                except Exception:
                    logging.exception("Emergency power cap failed; inspect GPU/driver/PCIe manually")
            if args.once:
                raise
        if args.once:
            break
        time.sleep(config["interval_seconds"])


if __name__ == "__main__":
    main()
