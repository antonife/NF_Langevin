"""
Fig 4: EPR comparison — NF vs Sekimoto (total) vs Schnakenberg (coarse-grained).

Three panels matching Landau_Theory fig4 style:
  (a) Protocol h(t) and ensemble response <x>(t)
  (b) EPR_Sekimoto (black), EPR_NF (blue), EPR_Schnakenberg (red dashed)
  (c) Integrated Sigma values + hidden entropy fractions
"""

import numpy as np
import matplotlib.pyplot as plt
from pub_style import apply_style, save_pub, COLORS, panel_label
from physics import LangevinParams, h_protocol, simulate_ensemble
from epr import (
    extract_time_bins,
    train_flows_per_bin,
    compute_epr_from_flows,
    sekimoto_epr,
    schnakenberg_epr,
)

apply_style()


def main():
    # --- Simulate ---
    params = LangevinParams()
    rng = np.random.default_rng(42)
    n_particles = 5000
    n_cycles = 6

    print("Simulating ensemble (6 cycles, 5000 particles)...")
    time_grid, ensemble_x = simulate_ensemble(n_particles, n_cycles, params, rng)

    # Extract last cycle
    steps_per_cycle = int(params.Period / params.dt)
    start_idx = len(time_grid) - steps_per_cycle
    time_cycle = time_grid[start_idx:] - time_grid[start_idx]
    ensemble_cycle = ensemble_x[:, start_idx:]

    h_cycle = h_protocol(time_cycle, params)
    xi_cycle = np.mean(ensemble_cycle, axis=0)

    # --- Sekimoto EPR ---
    print("Computing Sekimoto EPR...")
    EPR_sek, Sigma_sek = sekimoto_epr(time_cycle, ensemble_cycle, params)

    # Smooth for plotting
    smooth_win = 50
    kernel = np.ones(smooth_win) / smooth_win
    EPR_sek_filt = np.convolve(EPR_sek, kernel, mode="same")

    # --- Schnakenberg EPR ---
    print("Computing Schnakenberg EPR...")
    EPR_cg, Sigma_cg = schnakenberg_epr(time_cycle, ensemble_cycle, params)

    # --- NF EPR ---
    print("Training flows per time bin (40 bins)...")
    bins = extract_time_bins(time_cycle, ensemble_cycle, n_bins=40)
    flows = train_flows_per_bin(
        bins, n_layers=14, n_base_components=2, seed=42, maxiter=400, verbose=True
    )

    x_grid = np.linspace(-3, 3, 300)
    print("Computing NF EPR...")
    t_nf, EPR_nf, sigma_field = compute_epr_from_flows(flows, params, x_grid)
    Sigma_nf = np.trapezoid(EPR_nf, t_nf)

    # --- Fractions ---
    hidden_sek_cg = (Sigma_sek - Sigma_cg) / Sigma_sek * 100 if Sigma_sek > 0 else 0
    hidden_sek_nf = (Sigma_sek - Sigma_nf) / Sigma_sek * 100 if Sigma_sek > 0 else 0
    recovered_by_nf = (
        (Sigma_nf - Sigma_cg) / (Sigma_sek - Sigma_cg) * 100
        if (Sigma_sek - Sigma_cg) > 0
        else 0
    )

    print("\n--- Results ---")
    print(f"  Sigma_Sekimoto  = {Sigma_sek:.4f}")
    print(f"  Sigma_NF        = {Sigma_nf:.4f}")
    print(f"  Sigma_CG        = {Sigma_cg:.4f}")
    print(f"  Hidden (Sek-CG) = {hidden_sek_cg:.1f}%")
    print(f"  Hidden (Sek-NF) = {hidden_sek_nf:.1f}%")
    print(f"  NF recovers     = {recovered_by_nf:.1f}% of hidden entropy")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # (a) Protocol and response
    ax = axes[0]
    panel_label(ax, "(a)")
    ax.plot(
        time_cycle,
        h_cycle,
        color=COLORS["field"],
        ls="--",
        lw=1.5,
        label="Protocol $h(t)$",
    )
    ax.plot(
        time_cycle,
        xi_cycle,
        color=COLORS["order_param"],
        lw=2,
        label=r"Response $\langle x \rangle(t)$",
    )
    ax.set_ylabel("Amplitude")
    ax.legend(loc="lower left")

    # (b) EPR comparison
    ax = axes[1]
    panel_label(ax, "(b)")
    ax.plot(
        time_cycle,
        EPR_sek_filt,
        color=COLORS["epr_total"],
        lw=2,
        label="Total EPR (Sekimoto)",
    )
    ax.fill_between(
        time_cycle,
        0,
        EPR_sek_filt,
        color=COLORS["epr_hidden_fill"],
        alpha=0.5,
        label="Hidden entropy",
    )
    ax.plot(
        t_nf,
        EPR_nf,
        color=COLORS["epr_nf"],
        lw=2.5,
        marker="o",
        ms=3,
        label="NF EPR (this work)",
    )
    ax.plot(
        time_cycle,
        EPR_cg,
        color=COLORS["epr_cg"],
        lw=1.8,
        ls="--",
        label="CG EPR (Schnakenberg)",
    )
    ax.fill_between(
        time_cycle,
        0,
        EPR_cg,
        color=COLORS["epr_cg_fill"],
        alpha=0.5,
        label="CG entropy",
    )
    ax.set_ylabel(r"EPR ($k_B/\mathrm{s}$)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(bottom=0)

    # (c) Integrated values as bar chart
    ax = axes[2]
    panel_label(ax, "(c)")
    bar_labels = ["Sekimoto\n(total)", "NF\n(this work)", "Schnakenberg\n(CG)"]
    bar_values = [Sigma_sek, Sigma_nf, Sigma_cg]
    bar_colors = [COLORS["epr_total"], COLORS["epr_nf"], COLORS["epr_cg"]]
    bars = ax.bar(
        bar_labels,
        bar_values,
        color=bar_colors,
        alpha=0.8,
        width=0.5,
        edgecolor="k",
        linewidth=0.8,
    )

    # Add value labels on bars
    for bar, val in zip(bars, bar_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * max(bar_values),
            f"$\\Sigma = {val:.2f}$",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel(r"$\Sigma$ (integrated EPR)")
    ax.set_xlabel("Time $t$ (s)")

    # Annotation box
    textstr = (
        f"Hidden (Sek$-$CG): {hidden_sek_cg:.1f}%\n"
        f"NF recovers: {recovered_by_nf:.1f}% of hidden"
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
    ax.text(
        0.98,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=props,
    )

    # Remove x-axis sharing for bar chart
    axes[2].set_xlim(auto=True)

    plt.tight_layout()
    save_pub(fig, "fig4_epr_nf_vs_sekimoto")
    print("Fig 4 complete.")


if __name__ == "__main__":
    main()
