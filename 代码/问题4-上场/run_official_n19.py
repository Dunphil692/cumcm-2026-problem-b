#!/usr/bin/env python3
"""Official rehearsal of the N19 framework. Never start 正式测试 from here.

v14 combo_opk remains the rollback: run_official_v14.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from problem4_belief_n19 import run_belief
from problem4_official_robot import OfficialClient, OfficialClosed, OfficialRobot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-id", default=os.environ.get("CUMCM_ROBOT_ID", ""))
    parser.add_argument("--base", default="http://127.0.0.1:2026")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("official_rehearsal") / "q4_n19.jsonl",
    )
    args = parser.parse_args()
    if not args.robot_id:
        raise SystemExit(
            "缺少参赛队号。先 set CUMCM_ROBOT_ID=队号，或加 --robot-id。"
        )
    print("只走演练。不要点正式测试。决策是 N19 朝向位框架。", flush=True)
    client = OfficialClient(args.robot_id, base_url=args.base, log_path=args.log)
    robot = OfficialRobot(client)
    try:
        enter = robot.enter()
    except OfficialClosed as err:
        raise SystemExit(str(err)) from err
    print("enter remaining_real_duration_s=", enter.get("remaining_real_duration_s"))
    stats = run_belief(robot.world, fidelity="high", robot=robot)
    try:
        exit_response = robot.exit()
    except (OfficialClosed, Exception) as err:
        exit_response = {"error": str(err)}
    out = {
        "official_simulator": True,
        "rehearsal": True,
        "formal_test": False,
        "policy": "n19",
        "cleared_ok": robot.n_clear_ok,
        "clear_fail": robot.n_clear_fail,
        "time_s": robot.time,
        "walk_m": robot.walk_m,
        "measures": robot.n_measure,
        "stop_reason": stats.get("stop_reason"),
        "cover_visits": stats.get("cover_visits"),
        "heard": stats.get("heard"),
        "enter": enter,
        "exit": exit_response,
    }
    summary = args.log.with_suffix(".summary.json")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", args.log)
    print("wrote", summary)


if __name__ == "__main__":
    main()
