"""
Fig 1: Equilibrium validation — NF density vs Boltzmann at h=0.

Three panels:
  (a) p_NF(x) vs p_Boltzmann(x)
  (b) Score function comparison: d(log p)/dx
  (c) EPR ~ 0 at equilibrium (sanity check)
"""

import numpy as np
import matplotlib.pyplot as plt
from pub_style import apply_style, save_pub, COLORS, panel_label
from physics import LangevinParams, simulate_ensemble, boltzmann_density, grad_V
from physics import fokker_planck_current, epr_density
from flows_1d import train_planar_flow

apply_style()


def main():
    # --- Parameters ---
    params = LangevinParams(h_max=0.0)  # static potential, no driving
    rng = np.random.default_rng(42)
    n_particles = 5000
    n_cycles = 6

    # --- Simulate ---
    print("Simulating equilibrium ensemble (h=0)...")
    time_grid, ensemble_x = simulate_ensemble(n_particles, n_cycles, params, rng)

    # Extract last-cycle samples (equilibrated)
    steps_per_cycle = int(params.Period / params.dt)
    last_cycle_samples = ensemble_x[:, -1]  # final time snapshot
    print(
        f"  {n_particles} samples, range [{last_cycle_samples.min():.2f}, {last_cycle_samples.max():.2f}]"
    )

    # --- Train flow ---
    print("Training planar flow (12 layers)...")
    flow = train_planar_flow(
        last_cycle_samples, n_layers=12, n_base_components=2, seed=42, maxiter=500
    )

    # --- Evaluate ---
    x_grid = np.linspace(-3, 3, 500)
    h = 0.0

    # Boltzmann reference
    p_boltz = boltzmann_density(x_grid, h, params.kT)

    # Flow density
    log_p_nf = flow.log_prob(x_grid)
    p_nf = np.exp(log_p_nf)

    # Score functions
    score_nf = flow.score(x_grid)
    score_boltz = -grad_V(x_grid, h) / params.kT  # d(log p_B)/dx = -V'(x)/kT

    # EPR from flow
    dp_dx_nf = p_nf * score_nf
    J = fokker_planck_current(x_grid, p_nf, dp_dx_nf, h, params)
    sigma = epr_density(J, p_nf, params.D)
    epr_total = np.trapz(sigma, x_grid)

    # KL divergence
    mask = (p_nf > 1e-10) & (p_boltz > 1e-10)
    kl = np.trapz(p_nf[mask] * np.log(p_nf[mask] / p_boltz[mask]), x_grid[mask])

    print(f"  KL(NF || Boltzmann) = {kl:.6f}")
    print(f"  Integrated EPR = {epr_total:.6f} (should be ~0)")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # (a) Density comparison
    ax = axes[0]
    panel_label(ax, "(a)")
    ax.hist(
        last_cycle_samples,
        bins=80,
        density=True,
        alpha=0.3,
        color="gray",
        label="Histogram",
    )
    ax.plot(x_grid, p_boltz, color=COLORS["boltzmann"], lw=2, label="Boltzmann")
    ax.plot(x_grid, p_nf, color=COLORS["flow_density"], lw=2, ls="--", label="NF")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")
    ax.legend(fontsize=10)
    ax.set_xlim(-3, 3)

    # (b) Score comparison
    ax = axes[1]
    panel_label(ax, "(b)")
    ax.plot(
        x_grid, score_boltz, color=COLORS["boltzmann"], lw=2, label=r"$-V'(x)/k_BT$"
    )
    ax.plot(
        x_grid, score_nf, color=COLORS["flow_density"], lw=2, ls="--", label="NF score"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\partial \log p / \partial x$")
    ax.legend(fontsize=10)
    ax.set_xlim(-3, 3)

    # (c) EPR ~ 0
    ax = axes[2]
    panel_label(ax, "(c)")
    ax.plot(x_grid, sigma, color=COLORS["entropy"], lw=2)
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\sigma(x)$ (EPR density)")
    ax.set_xlim(-3, 3)
    # Annotation
    ax.text(
        0.95,
        0.95,
        f"$\\Sigma = {epr_total:.4f}$\nKL = {kl:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8, ec="gray"),
    )

    plt.tight_layout()
    save_pub(fig, "fig1_equilibrium_validation")
    print("Fig 1 complete.")


if __name__ == "__main__":
    main()
