"""
Fig 3: Learned density snapshots p(x|t) at key protocol phases.

2x3 panels showing p(x|t) at h = -2, -1, 0, +1, +2 and one repeat,
with V(x,h)/kT overlay (rescaled) showing how density tracks the tilting potential.
"""

import numpy as np
import matplotlib.pyplot as plt
from pub_style import apply_style, save_pub, COLORS, panel_label
from physics import LangevinParams, V, h_protocol, simulate_ensemble, boltzmann_density
from flows_1d import train_planar_flow

apply_style()


def main():
    # FIXME: At h=0 the NF density deviates notably from the Boltzmann reference,
    # unlike h=+/-1, +/-2 where agreement is close. Likely physical (nonequilibrium
    # vs quasi-static reference) but may also reflect an optimization issue with
    # planar flows on the symmetric bimodal target. Investigate before submission.
    # Tracked: https://gitlab.eif.urjc.es/afcaball/entropy-cromatines/-/issues/1
    # --- Simulate one full cycle ---
    params = LangevinParams()
    rng = np.random.default_rng(42)
    n_particles = 5000
    n_cycles = 6

    print("Simulating ensemble...")
    time_grid, ensemble_x = simulate_ensemble(n_particles, n_cycles, params, rng)

    # Extract last cycle
    steps_per_cycle = int(params.Period / params.dt)
    start_idx = len(time_grid) - steps_per_cycle
    time_cycle = time_grid[start_idx:] - time_grid[start_idx]
    ensemble_cycle = ensemble_x[:, start_idx:]

    # Target h values and find closest time indices
    h_targets = [-2.0, -1.0, 0.0, 1.0, 2.0]
    h_cycle = h_protocol(time_cycle, params)

    # For h=0, pick the ascending zero crossing (t ~ 0)
    # For others, find closest match in the first half-cycle
    snapshots = []
    for h_target in h_targets:
        diffs = np.abs(h_cycle - h_target)
        # Find first occurrence
        idx = np.argmin(diffs)
        t = time_cycle[idx]
        h_actual = h_cycle[idx]
        samples = ensemble_cycle[:, idx]
        snapshots.append((t, h_actual, samples))
        print(f"  h_target={h_target:+.1f}: t={t:.3f}, h_actual={h_actual:.3f}")

    # --- Train flows and plot ---
    x_grid = np.linspace(-3, 3, 400)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes_flat = axes.flatten()
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for i, (t, h_val, samples) in enumerate(snapshots):
        ax = axes_flat[i]
        panel_label(ax, labels[i])

        # Train flow on this snapshot
        print(f"  Training flow for h={h_val:.2f}...")
        flow = train_planar_flow(
            samples,
            n_layers=14,
            n_base_components=2,
            seed=42 + i,
            maxiter=400,
            verbose=False,
        )

        # NF density
        log_p = flow.log_prob(x_grid)
        p_nf = np.exp(log_p)

        # Boltzmann reference (quasi-static)
        p_boltz = boltzmann_density(x_grid, h_val, params.kT)

        # Rescaled potential for overlay
        v = V(x_grid, h_val) / params.kT
        v_rescaled = (v - v.min()) / (v.max() - v.min()) * p_nf.max() * 0.6

        # Plot
        ax.fill_between(x_grid, 0, v_rescaled, color=COLORS["potential"], alpha=0.15)
        ax.plot(x_grid, v_rescaled, color=COLORS["potential"], lw=1, ls=":", alpha=0.6)
        ax.hist(samples, bins=60, density=True, alpha=0.2, color="gray")
        ax.plot(x_grid, p_boltz, color=COLORS["boltzmann"], lw=1.5, label="Boltzmann")
        ax.plot(x_grid, p_nf, color=COLORS["flow_density"], lw=2, ls="--", label="NF")
        ax.set_xlim(-3, 3)
        ax.set_xlabel("$x$")
        if i % 3 == 0:
            ax.set_ylabel("$p(x)$")
        ax.set_title(f"$h = {h_val:+.1f}$", fontsize=11)
        if i == 0:
            ax.legend(fontsize=9, loc="upper right")

    # Use last panel for legend / annotation
    ax = axes_flat[5]
    ax.set_visible(False)

    plt.tight_layout()
    save_pub(fig, "fig3_learned_density_snapshots")
    print("Fig 3 complete.")


if __name__ == "__main__":
    main()
