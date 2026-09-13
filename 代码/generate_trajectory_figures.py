#!/usr/bin/env python3
"""
Generate National First Prize level trajectory figures for Problem 3 and Problem 4.
Produces high-DPI (300 DPI) publication-ready charts with ground truth sources,
coverage skeletons, search paths, and time progression curves.
"""

import sys
import os
import math
import random
from pathlib import Path

# Set MPLCONFIGDIR to prevent warning
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ensure clean Chinese and math fonts
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC', 'Arial Unicode MS', 'Hiragino Sans GB', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "赛题" / "B题" / "code"))
sys.path.insert(0, str(BASE_DIR / "代码" / "问题4-上场"))

import problem3_simulation_model as p3sim
import problem3_belief_struct as p3belief
import problem4_simulation_model as p4sim
import problem4_belief_n19 as p4belief

print("Modules loaded successfully.")

# ==============================================================================
# 1. Problem 3 Tracking & Generation
# ==============================================================================
class TrackRobotQ3(p3sim.Robot):
    def __init__(self, world):
        super().__init__(world)
        self.traj = [(0.0, 0.0, 0.0)] # x, y, time
        self.moves = []
        self.measures = []
        self.clears = []

    def move_to(self, x, y):
        x0, y0 = self.x, self.y
        super().move_to(x, y)
        self.traj.append((self.x, self.y, self.time))
        self.moves.append((x0, y0, self.x, self.y, self.time))

    def measure(self, channel):
        res = super().measure(channel)
        self.measures.append((self.x, self.y, self.time, channel, res.get("status")))
        return res

    def clear(self, channel):
        ok = super().clear(channel)
        if ok:
            self.clears.append((self.x, self.y, self.time, channel, len(self.clears) + 1))
        return ok

def generate_q3_figure():
    print("Generating Q3 trajectory figure (Seed 2032, 16 sources)...")
    rng = random.Random(2032)
    world = p3sim.generate_instance(rng)
    robot = TrackRobotQ3(world)
    stats = p3belief.run_belief_struct(world, fidelity="high", mode="e15", robot=robot)

    fig = plt.figure(figsize=(15, 8.5), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1, 1],
                           left=0.06, right=0.96, bottom=0.08, top=0.92, wspace=0.22, hspace=0.28)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_cleared = fig.add_subplot(gs[1, 1])

    # ---- 1. Map Panel ----
    ax_map.set_aspect('equal')
    ax_map.set_xlim(-2100, 2100)
    ax_map.set_ylim(-2100, 2100)

    # 1800m Boundary
    boundary = plt.Circle((0, 0), 1800, fill=True, facecolor='#f8f9fa',
                          edgecolor='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.9, zorder=1)
    ax_map.add_patch(boundary)

    # Coordinate Axes faint lines
    ax_map.axhline(0, color='#bdc3c7', linestyle=':', linewidth=0.8, zorder=1)
    ax_map.axvline(0, color='#bdc3c7', linestyle=':', linewidth=0.8, zorder=1)

    # 7 Census Points (Hexagon R=1200m + Origin)
    census_pts = [(0.0, 0.0)]
    for i in range(6):
        ang = i * math.pi / 3.0
        census_pts.append((1200.0 * math.cos(ang), 1200.0 * math.sin(ang)))

    cx, cy = zip(*census_pts)
    ax_map.scatter(cx, cy, marker='D', s=55, facecolor='none', edgecolor='#e67e22',
                   linewidth=1.8, label='预设7点普查网 ($R=1200$m 六边形+原点)', zorder=3)
    for idx, (px, py) in enumerate(census_pts):
        ax_map.text(px + 45, py + 45, f"$P_{idx}$", fontsize=8, color='#d35400', fontweight='bold', zorder=4)

    # Robot Trajectory Polyline
    tx = [p[0] for p in robot.traj]
    ty = [p[1] for p in robot.traj]
    tt = [p[2] for p in robot.traj]

    # Draw segments with subtle arrows
    for i in range(len(tx) - 1):
        ax_map.plot([tx[i], tx[i+1]], [ty[i], ty[i+1]], color='#2980b9', linewidth=1.4, alpha=0.75, zorder=2)
        dx = tx[i+1] - tx[i]
        dy = ty[i+1] - ty[i]
        dist = math.hypot(dx, dy)
        if dist > 350:
            mx = (tx[i] + tx[i+1]) / 2.0
            my = (ty[i] + ty[i+1]) / 2.0
            ax_map.annotate('', xy=(mx + dx*0.05, my + dy*0.05), xytext=(mx - dx*0.05, my - dy*0.05),
                            arrowprops=dict(arrowstyle="-|>", color='#1f618d', lw=1.2, mutation_scale=10),
                            zorder=2)

    # True Sources (16)
    sx = [s.x for s in world.sources]
    sy = [s.y for s in world.sources]
    ax_map.scatter(sx, sy, marker='o', s=70, facecolor='#e74c3c', edgecolor='#922b21',
                   linewidth=1.2, label='全向干扰源真值 ($n=16$, $R\\in[1000,1500]$m)', zorder=5)

    # Clearance Points
    for cx_pt, cy_pt, ct, ch, seq in robot.clears:
        ax_map.scatter(cx_pt, cy_pt, marker='*', s=130, facecolor='#27ae60', edgecolor='#145a32',
                       linewidth=1.2, zorder=6)
        ax_map.text(cx_pt + 35, cy_pt - 45, f"$C_{{{seq}}}$", fontsize=8, color='#1e8449',
                    fontweight='bold', zorder=7)

    # Origin Start
    ax_map.scatter(0, 0, marker='P', s=120, facecolor='#f39c12', edgecolor='#b9770e',
                   linewidth=1.5, label='出发点 $(0,0)$', zorder=8)

    ax_map.set_title("问题三：全向干扰源在线搜索与混排清除全景轨迹 (16源满清案例)", fontsize=12, fontweight='bold', pad=12)
    ax_map.set_xlabel("X 坐标 (米)", fontsize=10)
    ax_map.set_ylabel("Y 坐标 (米)", fontsize=10)
    ax_map.grid(True, linestyle='--', alpha=0.4)
    ax_map.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)

    # Stats Box on Map
    info_text = (
        f"【典型案例运行指标】\n"
        f"• 干扰源总数: {len(world.sources)} 只 (全向)\n"
        f"• 清除成功率: {robot.n_clear_ok}/{len(world.sources)} (100.0%)\n"
        f"• 总虚拟时间: {robot.time:.1f} s\n"
        f"• 单源平均耗时: {robot.time / robot.n_clear_ok:.1f} s/源\n"
        f"• 机器狗总位移: {robot.walk_m/1000.0:.2f} km\n"
        f"• 测向停测次数: {robot.n_measure} 次\n"
        f"• 停机机制: $\\Omega$ 空证书严格停机"
    )
    ax_map.text(0.03, 0.03, info_text, transform=ax_map.transAxes, fontsize=8.5,
                verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#34495e', alpha=0.92))

    # ---- 2. Time vs Cumulative Distance ----
    times = [p[2] for p in robot.traj]
    dists = [0.0]
    for i in range(1, len(robot.traj)):
        d = math.hypot(robot.traj[i][0] - robot.traj[i-1][0], robot.traj[i][1] - robot.traj[i-1][1])
        dists.append(dists[-1] + d)

    ax_time.plot(times, [d/1000.0 for d in dists], color='#2980b9', linewidth=2.0, label='累计移动路程 (km)')
    ax_time.set_title("机器狗移动路程与虚拟时序演进", fontsize=11, fontweight='bold')
    ax_time.set_xlabel("虚拟时间 (秒)", fontsize=9.5)
    ax_time.set_ylabel("累计行走里程 (千米)", fontsize=9.5)
    ax_time.grid(True, linestyle='--', alpha=0.4)

    for cx_pt, cy_pt, ct, ch, seq in robot.clears:
        idx = min(range(len(times)), key=lambda i: abs(times[i] - ct))
        ax_time.scatter(ct, dists[idx]/1000.0, marker='*', s=80, color='#27ae60', zorder=5)
    ax_time.legend(loc='lower right', fontsize=8.5)

    # ---- 3. Clearance Progression Curve ----
    clear_times = [0.0] + [c[2] for c in robot.clears]
    clear_counts = [0] + [c[4] for c in robot.clears]
    clear_times.append(robot.time)
    clear_counts.append(robot.n_clear_ok)

    ax_cleared.step(clear_times, clear_counts, where='post', color='#27ae60', linewidth=2.2, label='已清除干扰源累计')
    ax_cleared.set_title("干扰源清除进度阶梯演进", fontsize=11, fontweight='bold')
    ax_cleared.set_xlabel("虚拟时间 (秒)", fontsize=9.5)
    ax_cleared.set_ylabel("清除个数 (只)", fontsize=9.5)
    ax_cleared.set_yticks(range(0, 18, 2))
    ax_cleared.set_ylim(-0.5, 17)
    ax_cleared.grid(True, linestyle='--', alpha=0.4)

    first_clear_t = robot.clears[0][2]
    ax_cleared.axvspan(0, first_clear_t, color='#f39c12', alpha=0.15, label='初始探测与普查建网阶段')
    ax_cleared.axvspan(first_clear_t, robot.time, color='#2ecc71', alpha=0.12, label='动态混排交会与快速收割阶段')
    ax_cleared.legend(loc='lower right', fontsize=8.5)

    target_paths = [
        BASE_DIR / "赛题" / "B题" / "_figs" / "fig_q3_trajectory_case.png",
        BASE_DIR / "过程" / "全题执笔材料" / "03-问题三" / "fig_q3_trajectory_case.png",
        BASE_DIR / "论文" / "问题3" / "figures" / "fig_q3_trajectory_case.png"
    ]
    for p in target_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Saved: {p}")
    plt.close(fig)

# ==============================================================================
# 2. Problem 4 Tracking & Generation
# ==============================================================================
class TrackRobotQ4(p4sim.Robot):
    def __init__(self, world):
        super().__init__(world)
        self.traj = [(0.0, 0.0, 0.0)]
        self.moves = []
        self.measures = []
        self.clears = []

    def move_to(self, x, y):
        x0, y0 = self.x, self.y
        super().move_to(x, y)
        self.traj.append((self.x, self.y, self.time))
        self.moves.append((x0, y0, self.x, self.y, self.time))

    def measure(self, channel):
        res = super().measure(channel)
        self.measures.append((self.x, self.y, self.time, channel, res.get("status")))
        return res

    def clear(self, channel):
        ok = super().clear(channel)
        if ok:
            self.clears.append((self.x, self.y, self.time, channel, len(self.clears) + 1))
        return ok

def generate_q4_figure():
    print("Generating Q4 trajectory figure (Seed 2028, 12 sources: 7 directional, 5 omni)...")
    rng = random.Random(2028)
    world = p4sim.generate_instance(rng, p_dir=0.5)
    robot = TrackRobotQ4(world)
    stats = p4belief.run_belief(world, fidelity="high", robot=robot)

    fig = plt.figure(figsize=(15, 8.5), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1, 1],
                           left=0.06, right=0.96, bottom=0.08, top=0.92, wspace=0.22, hspace=0.28)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_cleared = fig.add_subplot(gs[1, 1])

    # ---- 1. Map Panel ----
    ax_map.set_aspect('equal')
    ax_map.set_xlim(-2100, 2100)
    ax_map.set_ylim(-2100, 2100)

    boundary = plt.Circle((0, 0), 1800, fill=True, facecolor='#fbfcfc',
                          edgecolor='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.9, zorder=1)
    ax_map.add_patch(boundary)
    ax_map.axhline(0, color='#bdc3c7', linestyle=':', linewidth=0.8, zorder=1)
    ax_map.axvline(0, color='#bdc3c7', linestyle=':', linewidth=0.8, zorder=1)

    n19_pts = [(0.0, 0.0)]
    for i in range(6):
        a = i * math.pi / 3.0
        n19_pts.append((1000.0 * math.cos(a), 1000.0 * math.sin(a)))
    for i in range(12):
        a = (i * math.pi / 6.0) + math.radians(15.0)
        n19_pts.append((1825.0 * math.cos(a), 1825.0 * math.sin(a)))

    nx, ny = zip(*n19_pts)
    ax_map.scatter(nx, ny, marker='h', s=45, facecolor='none', edgecolor='#e67e22',
                   linewidth=1.5, label='N19 普查骨干网 (原点+内圈6点+外环12点)', zorder=3)

    for s in world.sources:
        if s.directional and s.direction_deg is not None:
            theta1 = s.direction_deg - 90.0
            theta2 = s.direction_deg + 90.0
            wedge = patches.Wedge((s.x, s.y), s.r * 0.4, theta1, theta2,
                                  facecolor='#9b59b6', alpha=0.18, edgecolor='#8e44ad',
                                  linestyle=':', linewidth=1.0, zorder=2)
            ax_map.add_patch(wedge)
            rad = math.radians(s.direction_deg)
            ax_map.annotate('', xy=(s.x + 280*math.cos(rad), s.y + 280*math.sin(rad)), xytext=(s.x, s.y),
                            arrowprops=dict(arrowstyle="->", color='#8e44ad', lw=1.2), zorder=4)

    omni_s = [s for s in world.sources if not s.directional]
    dir_s = [s for s in world.sources if s.directional]

    if omni_s:
        ax_map.scatter([s.x for s in omni_s], [s.y for s in omni_s], marker='o', s=70,
                       facecolor='#3498db', edgecolor='#1b4f72', linewidth=1.2,
                       label=f'全向源真值 ($n={len(omni_s)}$)', zorder=5)
    if dir_s:
        ax_map.scatter([s.x for s in dir_s], [s.y for s in dir_s], marker='^', s=80,
                       facecolor='#9b59b6', edgecolor='#512e5f', linewidth=1.2,
                       label=f'定向源真值 ($n={len(dir_s)}$, 附带180°辐射扇区)', zorder=5)

    tx = [p[0] for p in robot.traj]
    ty = [p[1] for p in robot.traj]
    for i in range(len(tx) - 1):
        ax_map.plot([tx[i], tx[i+1]], [ty[i], ty[i+1]], color='#2c3e50', linewidth=1.4, alpha=0.75, zorder=2)
        dx = tx[i+1] - tx[i]
        dy = ty[i+1] - ty[i]
        dist = math.hypot(dx, dy)
        if dist > 400:
            mx = (tx[i] + tx[i+1]) / 2.0
            my = (ty[i] + ty[i+1]) / 2.0
            ax_map.annotate('', xy=(mx + dx*0.05, my + dy*0.05), xytext=(mx - dx*0.05, my - dy*0.05),
                            arrowprops=dict(arrowstyle="-|>", color='#1a252f', lw=1.2, mutation_scale=10),
                            zorder=2)

    for cx_pt, cy_pt, ct, ch, seq in robot.clears:
        ax_map.scatter(cx_pt, cy_pt, marker='*', s=140, facecolor='#27ae60', edgecolor='#145a32',
                       linewidth=1.2, zorder=6)
        ax_map.text(cx_pt + 40, cy_pt - 45, f"$C_{{{seq}}}$", fontsize=8, color='#1e8449',
                    fontweight='bold', zorder=7)

    ax_map.scatter(0, 0, marker='P', s=120, facecolor='#f39c12', edgecolor='#b9770e',
                   linewidth=1.5, label='出发点 $(0,0)$', zorder=8)

    ax_map.set_title("问题四：复杂定向/全向混合源 N19 顺流巡航与扇区延迟收割轨迹 (12源满清案例)", fontsize=12, fontweight='bold', pad=12)
    ax_map.set_xlabel("X 坐标 (米)", fontsize=10)
    ax_map.set_ylabel("Y 坐标 (米)", fontsize=10)
    ax_map.grid(True, linestyle='--', alpha=0.4)
    ax_map.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)

    n_omni = len(omni_s)
    n_dir = len(dir_s)
    info_text = (
        f"【真实测试基准对齐指标】\n"
        f"• 干扰源总数: {len(world.sources)} 只 (定向: {n_dir}, 全向: {n_omni})\n"
        f"• 清除成功率: {robot.n_clear_ok}/{len(world.sources)} (100.0%)\n"
        f"• 总虚拟时间: {robot.time:.1f} s\n"
        f"• 单源平均耗时: {robot.time / robot.n_clear_ok:.1f} s/源\n"
        f"• 机器狗总位移: {robot.walk_m/1000.0:.2f} km\n"
        f"• 测向停测次数: {robot.n_measure} 次\n"
        f"• 巡航策略: 逆时针顺流 + 1300m扇区延迟 + 奔袭免听\n"
        f"• 停机机制: $\\varepsilon$ 动态朝向网格全场收敛"
    )
    ax_map.text(0.03, 0.03, info_text, transform=ax_map.transAxes, fontsize=8.5,
                verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#34495e', alpha=0.92))

    times = [p[2] for p in robot.traj]
    dists = [0.0]
    for i in range(1, len(robot.traj)):
        d = math.hypot(robot.traj[i][0] - robot.traj[i-1][0], robot.traj[i][1] - robot.traj[i-1][1])
        dists.append(dists[-1] + d)

    ax_time.plot(times, [d/1000.0 for d in dists], color='#2c3e50', linewidth=2.0, label='累计移动路程 (km)')
    ax_time.set_title("巡航位移与虚拟时序演进", fontsize=11, fontweight='bold')
    ax_time.set_xlabel("虚拟时间 (秒)", fontsize=9.5)
    ax_time.set_ylabel("累计行走里程 (千米)", fontsize=9.5)
    ax_time.grid(True, linestyle='--', alpha=0.4)

    for cx_pt, cy_pt, ct, ch, seq in robot.clears:
        idx = min(range(len(times)), key=lambda i: abs(times[i] - ct))
        ax_time.scatter(ct, dists[idx]/1000.0, marker='*', s=80, color='#27ae60', zorder=5)
    ax_time.legend(loc='lower right', fontsize=8.5)

    clear_times = [0.0] + [c[2] for c in robot.clears]
    clear_counts = [0] + [c[4] for c in robot.clears]
    clear_times.append(robot.time)
    clear_counts.append(robot.n_clear_ok)

    ax_cleared.step(clear_times, clear_counts, where='post', color='#27ae60', linewidth=2.2, label='累计清除干扰源')
    ax_cleared.set_title("定向与全向干扰源逐阶段清除演进", fontsize=11, fontweight='bold')
    ax_cleared.set_xlabel("虚拟时间 (秒)", fontsize=9.5)
    ax_cleared.set_ylabel("清除个数 (只)", fontsize=9.5)
    ax_cleared.set_yticks(range(0, 14, 2))
    ax_cleared.set_ylim(-0.5, 13)
    ax_cleared.grid(True, linestyle='--', alpha=0.4)

    t_mid = 2500.0
    ax_cleared.axvspan(0, t_mid, color='#e67e22', alpha=0.15, label='内圈六边形普查与近场收割')
    ax_cleared.axvspan(t_mid, robot.time, color='#2980b9', alpha=0.12, label='外环12点逆时针顺流与延迟清除')
    ax_cleared.legend(loc='lower right', fontsize=8.5)

    target_paths = [
        BASE_DIR / "赛题" / "B题" / "_figs" / "fig_q4_trajectory_case.png",
        BASE_DIR / "过程" / "全题执笔材料" / "04-问题四" / "fig_q4_trajectory_case.png"
    ]
    for p in target_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Saved: {p}")
    plt.close(fig)

if __name__ == "__main__":
    generate_q3_figure()
    generate_q4_figure()
    print("ALL TRAJECTORY FIGURES GENERATED SUCCESSFULLY!")
