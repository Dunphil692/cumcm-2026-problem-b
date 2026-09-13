#!/usr/bin/env python3
"""Assemble the official supporting-materials zip and patched paper PDF."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(__file__).resolve().parent
STAGE = BUILD / "支撑材料"
OUT_DIR = ROOT / "论文/交卷"
EXTERNAL_DIR = Path(os.environ["CUMCM_EXTERNAL_DIR"]) if os.environ.get("CUMCM_EXTERNAL_DIR") else None
MESSY_ZIP = Path(os.environ["CUMCM_MESSY_ZIP"]) if os.environ.get("CUMCM_MESSY_ZIP") else None
Q1_LOCAL = ROOT / "过程/问题1上场"


def dec(name: str) -> str:
    try:
        return name.encode("cp437").decode("gbk")
    except Exception:
        return name


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_py_dir(src: Path, dest: Path, names: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        src_f = src / name
        if not src_f.is_file():
            raise FileNotFoundError(src_f)
        copy_file(src_f, dest / name)


def extract_from_messy(suffixes: list[str], dest_dir: Path) -> None:
    if MESSY_ZIP is None or not MESSY_ZIP.is_file():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(MESSY_ZIP) as zf:
        for info in zf.infolist():
            name = dec(info.filename)
            if "__MACOSX" in name or name.endswith(".DS_Store"):
                continue
            for suf in suffixes:
                if name.endswith(suf):
                    out = dest_dir / Path(suf).name
                    out.write_bytes(zf.read(info.filename))
                    break


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    OUT_DIR.mkdir(exist_ok=True)

    # --- AI (official filename) ---
    ai_candidates = [BUILD / "AI工具使用详情.pdf"]
    if EXTERNAL_DIR is not None:
        ai_candidates.append(EXTERNAL_DIR / "AI工具使用详情_填写版.pdf")
    ai_pdf = next((p for p in ai_candidates if p.is_file()), None)
    if ai_pdf is None:
        raise FileNotFoundError("AI工具使用详情.pdf (set CUMCM_EXTERNAL_DIR if using an outside copy)")
    copy_file(ai_pdf, STAGE / "AI工具使用详情.pdf")

    # --- Q1 ---
    q1_src = Q1_LOCAL
    if not q1_src.is_dir() and EXTERNAL_DIR is not None:
        q1_src = EXTERNAL_DIR / "问题一 2"
    q1 = STAGE / "问题一"
    copy_file(q1_src / "问题一代码final_修正版.py", q1 / "问题一代码final_修正版.py")
    copy_file(q1_src / "问题一_表格与图像.py", q1 / "问题一_表格与图像.py")
    copy_file(
        ROOT / "过程/问题3执笔材料/02-上场代码/problem1_localize.py",
        q1 / "problem1_localize.py",
    )
    for name in (
        "已知形状验证图.png",
        "测试用例图.png",
        "问题一流程图3.drawio.png",
    ):
        copy_file(q1_src / name, q1 / "figs" / name)
    extract_from_messy(
        [
            "问题一/figs/图1_示向度示意图.png",
            "问题一/figs/图2_交会定位示意图.png",
            "问题一/figs/图4_角形区域.png",
        ],
        q1 / "figs",
    )

    # --- Q2 ---
    q2_src = ROOT / "赛题/B题/问题二"
    q2 = STAGE / "问题二"
    copy_py_dir(
        q2_src / "code",
        q2 / "code",
        [
            "problem2_model.py",
            "problem2_paper.py",
            "problem1_localize.py",
            "test_problem2.py",
            "final_refine.py",
        ],
    )
    copy_file(q2_src / "results/problem2_summary.json", q2 / "results/problem2_summary.json")
    for name in (
        "problem2_candidate_region_center_reference.png",
        "problem2_candidate_region_inside_outward_800.png",
        "problem2_candidate_region_tangent_boundary.png",
    ):
        copy_file(q2_src / "figs" / name, q2 / "figs" / name)

    # --- Q3 ---
    q3_src = ROOT / "过程/问题3执笔材料/02-上场代码"
    q3 = STAGE / "问题三"
    copy_py_dir(
        q3_src,
        q3 / "code",
        [
            "problem1_localize.py",
            "problem2_model.py",
            "problem3_batch_station.py",
            "problem3_belief.py",
            "problem3_belief_struct.py",
            "problem3_fusion_policies.py",
            "problem3_official_robot.py",
            "problem3_route_solver.py",
            "problem3_simulation_model.py",
            "run_official_e15.py",
            "requirements.txt",
        ],
    )
    log_src = ROOT / "结果/支撑材料/问题3正式日志"
    for name in (
        "formal-p3-1-S589-ME5P-AVEG-7FVF.jlog",
        "formal-p3-2-CRNF-M248-R8CD-3QE2.jlog",
        "formal-p3-3-G3WW-HEAT-UQXS-8K7V.jlog",
    ):
        copy_file(log_src / name, q3 / "正式日志" / name)
    write_text(
        q3 / "正式日志/表1-问题3正式三次.txt",
        """问题3 正式测试三次（表1）
平均定位清除时间 = 总虚拟时间 / 清除个数
程序运行时间 = /enter 到 /exit 墙钟

序号  案例编码              清除个数  总虚拟时间/s  平均/s   程序运行时间/s  日志
1     S589-ME5P-AVEG-7FVF  16        3524.51       220.28   2.694           formal-p3-1-S589-ME5P-AVEG-7FVF.jlog
2     CRNF-M248-R8CD-3QE2  13        3362.51       258.65   1.987           formal-p3-2-CRNF-M248-R8CD-3QE2.jlog
3     G3WW-HEAT-UQXS-8K7V  12        3001.98       250.17   2.519           formal-p3-3-G3WW-HEAT-UQXS-8K7V.jlog

三次均为证书停机、清除失败 0、正常退出。
""",
    )
    copy_file(
        ROOT / "赛题/B题/_figs/q3_fullopt_10k/summary.json",
        q3 / "本地结果/fullopt_10000/summary.json",
    )
    copy_file(
        ROOT / "赛题/B题/_figs/q3_fullopt_10k/trials.csv",
        q3 / "本地结果/fullopt_10000/trials.csv",
    )
    copy_file(
        ROOT / "过程/问题3执笔材料/03-本地结果摘要/上场E15_本地1000局.json",
        q3 / "本地结果/e15_1000_summary.json",
    )
    copy_file(
        ROOT / "过程/全题执笔材料/图-问题三典型搜索与清除轨迹.png",
        q3 / "figs/典型搜索与清除轨迹.png",
    )

    # --- Q4 ---
    q4_src = ROOT / "代码/问题4-上场"
    q4 = STAGE / "问题四"
    copy_py_dir(
        q4_src,
        q4 / "code",
        [
            "problem4_belief_n19.py",
            "run_official_n19.py",
            "problem4_official_robot.py",
            "problem4_batch_station.py",
            "problem4_orient_cert.py",
            "problem4_route_solver.py",
            "problem4_simulation_model.py",
            "problem4_geometry.py",
        ],
    )
    write_text(
        q4 / "正式测试成绩.txt",
        """问题4 正式测试三次（表1）
平均定位清除时间 = 总虚拟时间 / 清除个数
程序运行时间 = /enter 到 /exit 墙钟

序号  清除个数  总虚拟时间/s  平均/s   程序运行时间/s  停机
1     12        6902.50       575.21   27.255          epsilon
2     11        6370.99       579.18   12.416          epsilon
3     15        8663.87       577.59   22.349          epsilon

三次均满额全清。本包不附 formal-p4-*.jlog。
""",
    )
    copy_file(
        ROOT / "赛题/B题/_figs/q4_n19_n1000/summary.json",
        q4 / "本地结果/n19_1000/summary.json",
    )
    copy_file(
        ROOT / "赛题/B题/_figs/q4_n19_n1000/trials.csv",
        q4 / "本地结果/n19_1000/trials.csv",
    )
    copy_file(
        ROOT / "过程/全题执笔材料/图-问题四典型搜索与清除轨迹.png",
        q4 / "figs/典型搜索与清除轨迹.png",
    )

    write_text(
        STAGE / "文件列表.txt",
        """支撑材料文件列表（与论文附录 A 一致）

AI工具使用详情.pdf

问题一/
  问题一代码final_修正版.py
  问题一_表格与图像.py
  problem1_localize.py
  figs/

问题二/
  code/problem2_model.py
  code/problem2_paper.py
  code/problem1_localize.py
  code/test_problem2.py
  code/final_refine.py
  results/problem2_summary.json
  figs/

问题三/
  code/ 上场源程序（E15：problem3_belief_struct.py + run_official_e15.py 等）
  正式日志/formal-p3-1-S589-ME5P-AVEG-7FVF.jlog
  正式日志/formal-p3-2-CRNF-M248-R8CD-3QE2.jlog
  正式日志/formal-p3-3-G3WW-HEAT-UQXS-8K7V.jlog
  正式日志/表1-问题3正式三次.txt
  本地结果/fullopt_10000/summary.json
  本地结果/fullopt_10000/trials.csv
  本地结果/e15_1000_summary.json
  figs/典型搜索与清除轨迹.png

问题四/
  code/ 上场源程序（N19：problem4_belief_n19.py + run_official_n19.py 等）
  正式测试成绩.txt
  本地结果/n19_1000/summary.json
  本地结果/n19_1000/trials.csv
  figs/典型搜索与清除轨迹.png
""",
    )

    forbidden = ("rita", "/users/")
    hits = []
    for path in STAGE.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".jlog"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for word in forbidden:
            if word.lower() in text.lower():
                hits.append(f"{path.relative_to(STAGE)}: {word}")
    if hits:
        raise SystemExit("identity leak:\n" + "\n".join(hits))

    zip_path = OUT_DIR / "支撑材料.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path("支撑材料") / path.relative_to(STAGE)))

    print("files", sum(1 for p in STAGE.rglob("*") if p.is_file()))
    print("zip_mb", round(zip_path.stat().st_size / 1024 / 1024, 3))
    print("wrote", zip_path)


if __name__ == "__main__":
    main()
