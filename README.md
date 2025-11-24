# tpt_jax_moog_filter
```python
# tpt_moog_filter.py
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Saturator: tanh or hard-clip
# ============================================================

def saturate(x, drive, mode):
    """
    mode = 0.0 → tanh
    mode = 1.0 → hard clip
    """
    xd = drive * x
    sat_tanh = jnp.tanh(xd)
    sat_clip = jnp.clip(xd, -1.0, 1.0)
    return jnp.where(mode < 0.5, sat_tanh, sat_clip)


# ============================================================
# TPT / BLT one-pole: (y, z_new)
# ============================================================

def tpt_onepole(x, z, G):
    """
    TPT 1-pole:
        v = (x - z) * G
        y = v + z
        z_new = y + v
    """
    v = (x - z) * G
    y = v + z
    z_new = y + v
    return y, z_new


# ============================================================
# Ladder evaluation given u and states (no feedback solving)
# ============================================================

def ladder_eval(u, z1, z2, z3, z4, G, drive, mode):
    # Stage 1
    s1 = saturate(u, drive, mode)
    y1, z1n = tpt_onepole(s1, z1, G)

    # Stage 2
    s2 = saturate(y1, drive, mode)
    y2, z2n = tpt_onepole(s2, z2, G)

    # Stage 3
    s3 = saturate(y2, drive, mode)
    y3, z3n = tpt_onepole(s3, z3, G)

    # Stage 4
    s4 = saturate(y3, drive, mode)
    y4, z4n = tpt_onepole(s4, z4, G)

    return y4, (z1n, z2n, z3n, z4n)


# ============================================================
# Filter state = (fs, z1, z2, z3, z4, prev_y4, mode)
# ============================================================

def ladder_init(fs, mode="tanh"):
    fs = jnp.asarray(fs, jnp.float32)
    z0 = jnp.asarray(0.0, jnp.float32)
    prev = jnp.asarray(0.0, jnp.float32)
    mode_val = 0.0 if mode == "tanh" else 1.0
    mode_j = jnp.asarray(mode_val, jnp.float32)
    return (fs, z0, z0, z0, z0, prev, mode_j)


# ============================================================
# Tick (one sample)
# ============================================================

def ladder_tick(state, x, cutoff_hz, resonance, drive, n_iter=2):
    """
    One-sample ZDF Moog Ladder.
    All parameters may be scalars or arrays.
    """

    fs, z1, z2, z3, z4, prev_y4, mode = state

    # g = tan(pi * fc / fs), G = g / (1+g)
    fc = jnp.clip(cutoff_hz, 0.0, 0.49 * fs)
    g = jnp.tan(jnp.pi * fc / fs)
    G = g / (1.0 + g + 1e-12)

    k = resonance

    # === initial guess for feedback input u
    u0 = x - k * saturate(prev_y4, drive, mode)

    # === ZDF fixed-point iteration
    def fp_body(i, u):
        y4_tmp, _ = ladder_eval(u, z1, z2, z3, z4, G, drive, mode)
        u_next = x - k * saturate(y4_tmp, drive, mode)
        return u_next

    u_final = lax.fori_loop(0, n_iter, fp_body, u0)

    # === final evaluation with converged u
    y4, (z1n, z2n, z3n, z4n) = ladder_eval(u_final, z1, z2, z3, z4, G, drive, mode)

    # Output (OTA saturator at the last stage)
    y = saturate(y4, drive, mode)

    new_state = (fs, z1n, z2n, z3n, z4n, y4, mode)
    return new_state, y


# ============================================================
# Block processing
# ============================================================

def ladder_process(state, x_block, cutoff_block, res_block, drive_block, n_iter=2):
    """
    Apply ladder_tick across a whole block.
    """

    def step_fn(state, inputs):
        x, fc, r, d = inputs
        return ladder_tick(state, x, fc, r, d, n_iter=n_iter)

    inputs = (x_block, cutoff_block, res_block, drive_block)
    final_state, y_block = lax.scan(step_fn, state, inputs)
    return final_state, y_block


# ============================================================
# Smoke test with simple plot
# ============================================================

if __name__ == "__main__":
    fs = 48_000.0
    dur = 0.02
    n = int(fs * dur)

    t = jnp.arange(n, dtype=jnp.float32) / fs

    # Test signal: single impulse
    x = jnp.where(t == 0.0, 1.0, 0.0)

    cutoff = jnp.ones_like(x) * 1000.0
    res = jnp.ones_like(x) * 3.5
    drive = jnp.ones_like(x) * 1.2

    state = ladder_init(fs, mode="tanh")

    state, y = ladder_process(state, x, cutoff, res, drive)

    y_np = np.asarray(y)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(n)/fs, y_np, label="Output")
    plt.title("ZDF Moog Ladder (functional) — Impulse Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("First 12 samples:", y_np[:12])

```
