"""Official HTTP robot. Decision stays v14 combo_opk; this only talks to the simulator.

Does not start a test. The operator must log in and click 问题4演练测试 first.
Never click 问题4正式测试 from these scripts.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from problem4_simulation_model import CHANNELS, Source

DEFAULT_BASE = "http://127.0.0.1:2026"
ARENA_ID = "default"


class OfficialClosed(RuntimeError):
    """Interface not open, or the test already ended."""


class OfficialRejected(RuntimeError):
    """HTTP came back but the action was not accepted."""


def map_measure(payload: dict) -> dict:
    kind = payload["measure_result"]
    if kind == "direction":
        return {"status": "direction", "svd_deg": float(payload["svd_deg"])}
    if kind in {"near", "no_signal"}:
        return {"status": kind}
    raise OfficialRejected(f"unknown measure_result {kind!r}")


class OfficialWorld:
    """No true source list. Dummy live source until that channel is cleared."""

    def __init__(self) -> None:
        self.sources: list[Source] = []
        self._cleared: set[int] = set()

    def source_by_channel(self, channel: int) -> Source | None:
        if channel in self._cleared:
            return None
        return Source(channel, 0.0, 0.0, 1500.0)

    def remaining(self):
        return None

    def n_directional(self) -> int:
        return 0

    def n_directional_cleared(self) -> int:
        return 0

    def mark_cleared(self, channel: int) -> None:
        if channel in self._cleared:
            return
        self._cleared.add(channel)
        self.sources.append(Source(channel, 0.0, 0.0, 1500.0, cleared=True))


class OfficialClient:
    def __init__(
        self,
        robot_id: str,
        base_url: str = DEFAULT_BASE,
        log_path: Path | None = None,
        timeout_s: float = 15.0,
    ) -> None:
        self.robot_id = robot_id
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.log_path = log_path
        self.n_post = 0
        self.last_virtual_s = 0.0
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def base(self, request_id: str) -> dict:
        return {
            "arena_id": ARENA_ID,
            "robot_id": self.robot_id,
            "request_id": request_id,
        }

    def action(self, request_id: str, x: float, y: float, channel: int) -> dict:
        payload = self.base(request_id)
        payload["position"] = {"x": float(x), "y": float(y)}
        payload["channel"] = int(channel)
        return payload

    def post(self, path: str, payload: dict, retries: int = 8) -> dict:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(retries):
            request = Request(
                self.base_url + path,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_s) as http_response:
                    raw = http_response.read().decode("utf-8")
                    status = http_response.status
            except HTTPError as err:
                raw = err.read().decode("utf-8") if err.fp is not None else ""
                try:
                    response = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    response = {}
                self._log(path, payload, err.code, response)
                raise OfficialRejected(f"HTTP {err.code} {path} {response}") from err
            except URLError as err:
                last_err = err
                time.sleep(min(2.0, 0.25 * (attempt + 1)))
                continue
            try:
                response = json.loads(raw)
            except json.JSONDecodeError as err:
                raise OfficialRejected(f"bad JSON from {path}: {raw[:200]}") from err
            self._log(path, payload, status, response)
            if response.get("accepted") is True:
                if "virtual_time_s" in response:
                    self.last_virtual_s = float(response["virtual_time_s"])
                self.n_post += 1
                return response
            raise OfficialRejected(f"accepted=false {path} {response}")
        raise OfficialClosed(
            "机器狗接口连不上。先在模拟器里登录，点「问题4演练测试」，"
            "等倒计时结束、接口就绪后再跑。"
        ) from last_err

    def _log(self, path: str, payload: dict, status: int, response: dict) -> None:
        if self.log_path is None:
            return
        row = {
            "path": path,
            "status": status,
            "request": payload,
            "response": response,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class OfficialRobot:
    def __init__(self, client: OfficialClient) -> None:
        self.client = client
        self.world = OfficialWorld()
        self.x = 0.0
        self.y = 0.0
        self.channel = 1
        self.time = 0.0
        self.walk_m = 0.0
        self.n_measure = 0
        self.n_clear_ok = 0
        self.n_clear_fail = 0
        self.n_replan = 0
        self.log = {channel: [] for channel in CHANNELS}
        self._seq = 0
        self.enter_info: dict | None = None

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq}"

    def _sync_time(self) -> None:
        self.time = self.client.last_virtual_s

    def enter(self) -> dict:
        response = self.client.post("/enter", self.client.base(self._next_id("enter")))
        self.x, self.y, self.channel = 0.0, 0.0, 1
        self._sync_time()
        self.enter_info = response
        return response

    def exit(self) -> dict:
        return self.client.post("/exit", self.client.base(self._next_id("exit")))

    def move_to(self, x: float, y: float) -> None:
        self.walk_m += math.hypot(x - self.x, y - self.y)
        self.x, self.y = float(x), float(y)

    def measure(self, channel: int) -> dict:
        payload = self.client.action(
            self._next_id("measure"), self.x, self.y, channel
        )
        response = self.client.post("/measure", payload)
        result = map_measure(response)
        self.channel = int(channel)
        self.n_measure += 1
        self._sync_time()
        self.log[channel].append({"x": self.x, "y": self.y, **result})
        return result

    def clear(self, channel: int) -> bool:
        payload = self.client.action(
            self._next_id("clear"), self.x, self.y, channel
        )
        response = self.client.post("/clear", payload)
        ok = response.get("clear_result") == "success"
        self._sync_time()
        if ok:
            self.n_clear_ok += 1
            self.world.mark_cleared(int(channel))
            self.log[channel].append(
                {"x": self.x, "y": self.y, "status": "cleared"}
            )
        else:
            self.n_clear_fail += 1
        return ok


def smoke_actions() -> list[tuple[str, float | None, float | None, int | None, str]]:
    """Attachment §10 four live actions after /enter. Not a search policy."""
    return [
        ("/measure", 300.0, 400.0, 1, "measure-1"),
        ("/measure", 300.0, 400.0, 2, "measure-2"),
        ("/clear", 300.0, 0.0, 3, "clear-1"),
        ("/measure", 300.0, 0.0, 2, "measure-3"),
    ]


def expected_smoke_virtual_s() -> float:
    """If the last clear misses, virtual time should be 199 s."""
    return 199.0
