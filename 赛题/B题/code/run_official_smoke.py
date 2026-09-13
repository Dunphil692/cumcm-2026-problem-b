#!/usr/bin/env python3
"""Attachment §11 four-command smoke. Use 问题3演练测试 only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from problem3_official_robot import OfficialClient, OfficialClosed, smoke_actions


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
        / "smoke.jsonl",
    )
    args = parser.parse_args()
    if not args.robot_id:
        raise SystemExit(
            "缺少参赛队号。先 export CUMCM_ROBOT_ID='队号'，或加 --robot-id。"
        )
    client = OfficialClient(args.robot_id, base_url=args.base, log_path=args.log)
    print("只走演练。不要点正式测试。", flush=True)
    try:
        enter = client.post("/enter", client.base("enter-1"))
    except OfficialClosed as err:
        raise SystemExit(str(err)) from err
    print("enter", json.dumps(enter, ensure_ascii=False))
    print("本局可用现实时间", enter.get("remaining_real_duration_s"), "秒")
    for path, x, y, channel, request_id in smoke_actions():
        payload = client.action(request_id, x, y, channel)
        response = client.post(path, payload)
        print(path, json.dumps(response, ensure_ascii=False))
        if path == "/measure":
            kind = response.get("measure_result")
            if kind == "direction":
                print("  示向度", response.get("svd_deg"))
            elif kind == "near":
                print("  距离过近")
            else:
                print("  未测得信号")
        elif response.get("clear_result") == "success":
            print("  清除成功")
        else:
            print("  附近没有目标")
    exit_response = client.post("/exit", client.base("exit-1"))
    print("exit", json.dumps(exit_response, ensure_ascii=False))
    print("wrote", args.log)


if __name__ == "__main__":
    main()
