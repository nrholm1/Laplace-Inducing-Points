import time
from statistics import mean, pstdev
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
import jax
from jax import tree_util

@dataclass
class JaxTiming:
    compile_and_first_s: float          # first call (includes JIT compile)
    warmup_calls: int                   # additional warmup calls (not timed)
    runs: int                           # timed runs
    times_s: List[float]                # per-run durations (steady state)
    mean_s: float
    std_s: float
    min_s: float
    max_s: float

def _block_until_ready(x: Any) -> None:
    """Block on JAX arrays inside nested structures without copying to host."""
    def _maybe_block(v):
        return v.block_until_ready() if hasattr(v, "block_until_ready") else v
    tree_util.tree_map(_maybe_block, x)

def time_jax(
    fn: Callable[..., Any],
    *args,
    warmup: int = 1,
    runs: int = 10,
    **kwargs,
) -> JaxTiming:
    """
    Time a JAX function `fn(*args, **kwargs)`.

    - Separately measures the first call (which can include JIT compile time).
    - Performs `warmup` additional calls (not timed) to stabilize caches.
    - Then measures `runs` steady-state calls.
    - Uses `block_until_ready` to account for async dispatch.

    Returns a JaxTiming dataclass with detailed stats.
    """

    # --- First call: includes possible JIT compilation ---
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _block_until_ready(out)
    t1 = time.perf_counter()
    compile_and_first = t1 - t0

    # --- Extra warmups (not timed) ---
    for _ in range(max(0, warmup)):
        _block_until_ready(fn(*args, **kwargs))

    # --- Timed steady-state runs ---
    times: List[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        _block_until_ready(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return JaxTiming(
        compile_and_first_s=compile_and_first,
        warmup_calls=max(0, warmup),
        runs=max(1, runs),
        times_s=times,
        mean_s=mean(times),
        std_s=pstdev(times) if len(times) > 1 else 0.0,
        min_s=min(times),
        max_s=max(times),
    )
