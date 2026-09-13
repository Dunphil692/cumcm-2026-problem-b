#!/usr/bin/env python3
"""One official rehearsal of fielded E15. Never start 正式测试 from here."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from problem3_belief_struct import run_belief_struct
from problem3_official_robot import OfficialClient, OfficialClosed, OfficialRobot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID", ""))
    parser.add_argument("--base", default="http://127.0.0.1:2026")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "_figs"
        / "official_rehearsal"
        / "e15.jsonl",
    )
    args = parser.parse_args()
    if not args.robot_id:
        raise SystemExit(
            "缺少参赛队号。先 export CUMCM_ROBOT_ID='队号'，或加 --robot-id。"
        )
    print("只走演练。不要点正式测试。决策是上场 E15。", flush=True)
    client = OfficialClient(args.robot_id, base_url=args.base, log_path=args.log)
    robot = OfficialRobot(client)
    try:
        enter = robot.enter()
    except OfficialClosed as err:
        raise SystemExit(str(err)) from err
    print("enter remaining_real_duration_s=", enter.get("remaining_real_duration_s"))
    stats = run_belief_struct(robot.world, fidelity="high", mode="e15", robot=robot)
    try:
        exit_response = robot.exit()
    except (OfficialClosed, Exception) as err:
        exit_response = {"error": str(err)}
    out = {
        "official_simulator": True,
        "rehearsal": True,
        "formal_test": False,
        "cleared_ok": robot.n_clear_ok,
        "clear_fail": robot.n_clear_fail,
        "time_s": robot.time,
        "walk_m": robot.walk_m,
        "measures": robot.n_measure,
        "stop_reason": stats.get("stop_reason"),
        "enter": enter,
        "exit": exit_response,
    }
    summary = args.log.with_suffix(".summary.json")
    summary.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", args.log)
    print("wrote", summary)


if __name__ == "__main__":
    main()
