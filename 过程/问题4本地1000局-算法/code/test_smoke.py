import sys, time
import random
from pathlib import Path

from problem4_simulation_model import generate_instance
from problem4_belief_n19 import run_belief

print("Testing 1 trial...", flush=True)
rng = random.Random(2027)
world = generate_instance(rng, p_dir=0.5)
world.remaining = lambda: None

t0 = time.time()
res = run_belief(world, fidelity="fast")
dt = time.time() - t0

print(f"PASS! 1 trial completed in {dt:.1f}s.")
print(f"Cleared: {res['cleared']}/{len(world.sources)} in {res['time_s']:.1f}s (walk: {res['walk_m']:.0f}m)")
