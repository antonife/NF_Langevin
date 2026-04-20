"""
Fig 5: Base distribution comparison — standard normal vs GMM.

Demonstrates why a Gaussian mixture base is essential for bimodal targets.
At h=0 the Landau potential has two wells, producing a bimodal equilibrium density.
A planar flow with standard normal base cannot capture this (monotone composition
of a unimodal base stays unimodal). GMM base solves this.

Three panels:
  (a) Density: GMM base vs standard base vs Boltzmann
  (b) Score function comparison
  (c) KL divergence vs number of layers (both bases)
"""

import numpy as np
import matplotlib.pyplot as plt
from pub_style import apply_style, save_pub, COLORS, panel_label
from physics import LangevinParams, simulate_ensemble, boltzmann_density, grad_V
from flows_1d import train_planar_flow

apply_style()


def main():
    # --- Generate bimodal equilibrium data (h=0) ---
    params = LangevinParams(h_max=0.0)
    rng = np.random.default_rng(42)
    n_particles = 5000
    n_cycles = 6

    print("Simulating equilibrium ensemble (h=0, bimodal)...")
    time_grid, ensemble_x = simulate_ensemble(n_particles, n_cycles, params, rng)
    samples = ensemble_x[:, -1]

    x_grid = np.linspace(-3, 3, 400)
    h = 0.0
    p_boltz = boltzmann_density(x_grid, h, params.kT)

    # --- Train with standard normal base (should fail) ---
    print("\nTraining Planar Flow with standard normal base (12 layers)...")
    flow_std = train_planar_flow(
        samples, n_layers=12, n_base_components=1, seed=42, maxiter=500
    )

    # --- Train with GMM base (should succeed) ---
    print("\nTraining Planar Flow with GMM base (12 layers, K=2)...")
    flow_gmm = train_planar_flow(
        samples, n_layers=12, n_base_components=2, seed=42, maxiter=500
    )

    # --- Evaluate ---
    p_std = np.exp(flow_std.log_prob(x_grid))
    p_gmm = np.exp(flow_gmm.log_prob(x_grid))
    score_std = flow_std.score(x_grid)
    score_gmm = flow_gmm.score(x_grid)
    score_boltz = -grad_V(x_grid, h) / params.kT

    # KL divergences
    mask_s = (p_std > 1e-10) & (p_boltz > 1e-10)
    kl_std = np.trapz(
        p_std[mask_s] * np.log(p_std[mask_s] / p_boltz[mask_s]), x_grid[mask_s]
    )
    mask_g = (p_gmm > 1e-10) & (p_boltz > 1e-10)
    kl_gmm = np.trapz(
        p_gmm[mask_g] * np.log(p_gmm[mask_g] / p_boltz[mask_g]), x_grid[mask_g]
    )
    print(f"\n  KL (standard base): {kl_std:.6f}")
    print(f"  KL (GMM base):      {kl_gmm:.6f}")

    # --- Panel (c): KL vs number of layers ---
    layer_counts = [4, 8, 12, 16, 20]
    kl_std_layers = []
    kl_gmm_layers = []

    print("\nScanning layer counts...")
    for nl in layer_counts:
        f_s = train_planar_flow(
            samples,
            n_layers=nl,
            n_base_components=1,
            seed=42,
            maxiter=400,
            verbose=False,
        )
        f_g = train_planar_flow(
            samples,
            n_layers=nl,
            n_base_components=2,
            seed=42,
            maxiter=400,
            verbose=False,
        )
        p_s = np.exp(f_s.log_prob(x_grid))
        p_g = np.exp(f_g.log_prob(x_grid))
        ms = (p_s > 1e-10) & (p_boltz > 1e-10)
        mg = (p_g > 1e-10) & (p_boltz > 1e-10)
        ks = np.trapz(p_s[ms] * np.log(p_s[ms] / p_boltz[ms]), x_grid[ms])
        kg = np.trapz(p_g[mg] * np.log(p_g[mg] / p_boltz[mg]), x_grid[mg])
        kl_std_layers.append(ks)
        kl_gmm_layers.append(kg)
        print(f"  L={nl:2d}: KL_std={ks:.4f}, KL_gmm={kg:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # (a) Density comparison
    ax = axes[0]
    panel_label(ax, "(a)")
    ax.hist(samples, bins=80, density=True, alpha=0.2, color="gray", label="Histogram")
    ax.plot(x_grid, p_boltz, color=COLORS["boltzmann"], lw=2, label="Boltzmann")
    ax.plot(
        x_grid,
        p_std,
        color=COLORS["epr_cg"],
        lw=2,
        ls=":",
        label=f"Std. base (KL={kl_std:.3f})",
    )
    ax.plot(
        x_grid,
        p_gmm,
        color=COLORS["flow_density"],
        lw=2,
        ls="--",
        label=f"GMM base (KL={kl_gmm:.4f})",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")
    ax.legend(fontsize=10)
    ax.set_xlim(-3, 3)

    # (b) Score function comparison
    ax = axes[1]
    panel_label(ax, "(b)")
    ax.plot(
        x_grid,
        score_boltz,
        color=COLORS["boltzmann"],
        lw=2,
        label=r"$-V'(x)/k_BT$",
    )
    ax.plot(
        x_grid,
        score_std,
        color=COLORS["epr_cg"],
        lw=1.5,
        ls=":",
        label="Std. base",
    )
    ax.plot(
        x_grid,
        score_gmm,
        color=COLORS["flow_density"],
        lw=2,
        ls="--",
        label="GMM base",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\partial \log p / \partial x$")
    ax.legend(fontsize=10)
    ax.set_xlim(-3, 3)

    # (c) KL vs layers
    ax = axes[2]
    panel_label(ax, "(c)")
    ax.semilogy(
        layer_counts,
        kl_std_layers,
        "o-",
        color=COLORS["epr_cg"],
        lw=2,
        label="Std. normal base",
    )
    ax.semilogy(
        layer_counts,
        kl_gmm_layers,
        "s--",
        color=COLORS["flow_density"],
        lw=2,
        label="GMM base ($K=2$)",
    )
    ax.set_xlabel("Number of layers $L$")
    ax.set_ylabel("KL divergence")
    ax.legend(fontsize=10)

    plt.tight_layout()
    save_pub(fig, "fig5_architecture_comparison")
    print("Fig 5 complete.")


if __name__ == "__main__":
    main()
