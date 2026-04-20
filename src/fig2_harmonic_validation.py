"""
Fig 2: Harmonic potential validation — NF density vs exact Gaussian.

V(x) = k*x^2/2, exact p(x) = N(0, kT/k).
Two panels:
  (a) Densities: NF vs Gaussian
  (b) Residuals: p_NF - p_exact
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pub_style import apply_style, save_pub, COLORS, panel_label
from flows_1d import train_planar_flow

apply_style()


def main():
    # --- Parameters ---
    k = 2.0  # spring constant
    kT = 0.5  # thermal energy
    sigma_eq = np.sqrt(kT / k)  # equilibrium std dev
    n_samples = 5000
    rng = np.random.default_rng(42)

    # --- Generate exact samples from harmonic potential ---
    # At equilibrium: p(x) = N(0, kT/k)
    data = rng.normal(0, sigma_eq, n_samples)
    print(f"Harmonic potential: k={k}, kT={kT}, sigma_eq={sigma_eq:.4f}")
    print(f"  Sample std = {data.std():.4f} (expect {sigma_eq:.4f})")

    # --- Train flow ---
    print("Training planar flow (10 layers)...")
    flow = train_planar_flow(data, n_layers=10, seed=42, maxiter=400)

    # --- Evaluate ---
    x_grid = np.linspace(-3, 3, 500)

    # Exact Gaussian
    p_exact = norm.pdf(x_grid, 0, sigma_eq)

    # Flow density
    log_p_nf = flow.log_prob(x_grid)
    p_nf = np.exp(log_p_nf)

    # KL divergence
    mask = (p_nf > 1e-10) & (p_exact > 1e-10)
    kl = np.trapezoid(p_nf[mask] * np.log(p_nf[mask] / p_exact[mask]), x_grid[mask])
    print(f"  KL(NF || Gaussian) = {kl:.6f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # (a) Densities
    ax = axes[0]
    panel_label(ax, "(a)")
    ax.hist(data, bins=60, density=True, alpha=0.3, color="gray", label="Samples")
    ax.plot(
        x_grid,
        p_exact,
        color=COLORS["boltzmann"],
        lw=2,
        label=r"$\mathcal{N}(0, k_BT/k)$",
    )
    ax.plot(x_grid, p_nf, color=COLORS["flow_density"], lw=2, ls="--", label="NF")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")
    ax.legend(fontsize=8)
    ax.set_xlim(-3, 3)
    ax.text(
        0.95,
        0.95,
        f"KL = {kl:.5f}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8, ec="gray"),
    )

    # (b) Residuals
    ax = axes[1]
    panel_label(ax, "(b)")
    residual = p_nf - p_exact
    ax.plot(x_grid, residual, color=COLORS["residual"], lw=1.5)
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.fill_between(x_grid, residual, 0, alpha=0.2, color=COLORS["residual"])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p_{NF}(x) - p_{exact}(x)$")
    ax.set_xlim(-3, 3)

    plt.tight_layout()
    save_pub(fig, "fig2_harmonic_validation")
    print("Fig 2 complete.")


if __name__ == "__main__":
    main()
