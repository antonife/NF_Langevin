"""
Fig 6: Force-free EPR validation — continuity equation vs known dynamics.

Four panels:
  (a) KM-learned D₁(x,t) vs known drift −V'(x,h)/γ at 5 h-values
  (b) KM-learned D₂(x,t) vs known D = 0.5
  (c) EPR time series: NF (known force) vs continuity equation (force-free)
  (d) Cycle-integrated EPR comparison bar chart

The continuity equation approach computes J(x,t) = −∫ ∂p/∂t dx directly
from NF-learned densities, bypassing noisy KM drift estimation entirely.
Only the diffusion coefficient D is needed (learned from KM D₂).
"""

import numpy as np
import matplotlib.pyplot as plt
from pub_style import apply_style, save_pub, COLORS, panel_label
from physics import LangevinParams, h_protocol, simulate_ensemble, grad_V
from epr import (
    extract_time_bins,
    train_flows_per_bin,
    compute_epr_from_flows,
    compute_epr_continuity,
    sekimoto_epr,
)
from km_inference import KMInferenceTimeSeries

apply_style()


def main():
    # --- Simulation parameters ---
    params = LangevinParams()
    rng = np.random.default_rng(42)
    n_particles = 5000
    n_cycles = 6
    n_flow_bins = 20

    print("=" * 60)
    print("Force-Free EPR Validation: KM vs Known Dynamics")
    print("=" * 60)

    # --- Simulate ---
    print("\n[1/5] Simulating ensemble (6 cycles, 5000 particles)...")
    time_grid, ensemble_x = simulate_ensemble(n_particles, n_cycles, params, rng)

    # Extract last cycle for analysis
    steps_per_cycle = int(params.Period / params.dt)
    start_idx = len(time_grid) - steps_per_cycle
    time_cycle = time_grid[start_idx:] - time_grid[start_idx]
    ensemble_cycle = ensemble_x[:, start_idx:]

    # --- Kramers-Moyal inference ---
    print("\n[2/5] Running Kramers-Moyal inference per snapshot...")
    x_grid = np.linspace(-3, 3, 100)
    km = KMInferenceTimeSeries(
        time_cycle,
        ensemble_cycle,
        x_grid,
        n_snapshots=n_flow_bins,
        bandwidth=0.08,
        smooth_sigma=0.05,
    )
    print(f"  KM inference complete: {len(km.t_snapshots)} snapshots")
    print(f"  Mean D₂ = {km.get_mean_D2():.4f} (expected: {params.D:.4f})")

    # --- Train normalizing flows ---
    print("\n[3/5] Training normalizing flows per time bin...")
    bins = extract_time_bins(time_cycle, ensemble_cycle, n_bins=n_flow_bins)
    flows = train_flows_per_bin(
        bins, n_layers=8, n_base_components=2, seed=42, maxiter=200, verbose=False
    )
    print("  Flow training complete")

    # --- Compute EPR with known force ---
    print("\n[4/5] Computing EPR (known force)...")
    x_epr = np.linspace(-3, 3, 300)
    t_nf, EPR_nf, _ = compute_epr_from_flows(flows, params, x_epr)
    Sigma_nf = np.trapz(EPR_nf, t_nf)
    print(f"  Σ_NF (known force) = {Sigma_nf:.4f}")

    # --- Compute EPR via continuity equation (force-free) ---
    print("\n[5/5] Computing EPR (force-free, continuity equation)...")
    D2_learned = km.get_mean_D2()
    t_cont, EPR_cont, _ = compute_epr_continuity(flows, D2_learned, x_epr)
    Sigma_cont = np.trapz(EPR_cont, t_cont)
    print(f"  Σ_continuity (force-free) = {Sigma_cont:.4f}")

    # --- Sekimoto reference ---
    print("\nComputing Sekimoto reference...")
    _, Sigma_sek = sekimoto_epr(time_cycle, ensemble_cycle, params)
    print(f"  Σ_Sekimoto = {Sigma_sek:.4f}")

    # --- Error metrics ---
    err_nf = abs(Sigma_nf - Sigma_sek) / Sigma_sek * 100
    err_cont = abs(Sigma_cont - Sigma_sek) / Sigma_sek * 100
    print("\n--- Validation Metrics ---")
    print(f"  |Σ_NF - Σ_Sek| / Σ_Sek   = {err_nf:.1f}%")
    print(f"  |Σ_cont - Σ_Sek| / Σ_Sek = {err_cont:.1f}%")
    print(f"  Target: < 15%  -->  {'PASS' if err_cont < 15 else 'FAIL'}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # === Panel (a): D₁ comparison at 5 h-values ===
    ax = axes[0, 0]
    panel_label(ax, "(a)")

    # Select 5 snapshots spanning h-range
    h_values = h_protocol(km.t_snapshots, params)
    snapshot_indices = np.linspace(0, n_flow_bins - 1, 5, dtype=int).tolist()
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(snapshot_indices)))

    for idx, color in zip(snapshot_indices, colors):
        h = h_values[idx]
        t = km.t_snapshots[idx]

        # Learned D₁
        D1_learned = km.get_D1_at_grid(idx)

        # Known drift -V'(x,h)/γ
        drift_true = -grad_V(x_grid, h) / params.gamma

        ax.plot(x_grid, D1_learned, color=color, lw=2, label=f"$h={h:.1f}$ (learned)")
        ax.plot(x_grid, drift_true, color=color, lw=1.5, ls="--", alpha=0.7)

    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$D_1(x)$ (drift)")
    ax.set_title("Learned drift vs known $-V'(x,h)/\\gamma$")
    ax.set_xlim(-2.5, 2.5)
    ax.legend(fontsize=10, ncol=2, loc="upper right")

    # === Panel (b): D₂ comparison ===
    ax = axes[0, 1]
    panel_label(ax, "(b)")

    # Plot D₂ for same snapshots
    for idx, color in zip(snapshot_indices, colors):
        D2_learned = km.get_D2_at_grid(idx)
        ax.plot(x_grid, D2_learned, color=color, lw=1.5, alpha=0.8)

    # True diffusion (constant)
    ax.axhline(params.D, color="k", lw=2, ls="--", label=f"True $D = {params.D}$")

    # Mean learned D₂
    mean_D2 = km.get_mean_D2()
    ax.axhline(
        mean_D2,
        color=COLORS["epr_nf"],
        lw=2,
        ls=":",
        label=f"Mean learned $D_2 = {mean_D2:.3f}$",
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$D_2(x)$ (diffusion)")
    ax.set_title("Learned diffusion vs constant $D = k_BT/\\gamma$")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0.3, 0.7)
    ax.legend(fontsize=9)

    # D₂ error
    D2_err = abs(mean_D2 - params.D) / params.D * 100
    ax.text(
        0.02,
        0.98,
        f"Rel. error: {D2_err:.1f}%",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # === Panel (c): EPR time series comparison ===
    ax = axes[1, 0]
    panel_label(ax, "(c)")

    ax.plot(
        t_nf,
        EPR_nf,
        color=COLORS["epr_nf"],
        lw=2.5,
        marker="o",
        ms=3,
        label="NF (known force)",
    )
    ax.plot(
        t_cont,
        EPR_cont,
        color="#FF7F00",
        lw=2.5,
        marker="s",
        ms=3,
        label="Continuity (force-free)",
    )

    # Difference
    ax2 = ax.twinx()
    diff = EPR_cont - EPR_nf
    ax2.fill_between(t_cont, 0, diff, alpha=0.3, color="gray")
    ax2.set_ylabel("Difference (cont. $-$ NF)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    max_diff = max(abs(diff)) if max(abs(diff)) > 0 else 1.0
    ax2.set_ylim(-max_diff * 2, max_diff * 2)

    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel(r"EPR ($k_B/\mathrm{s}$)")
    ax.set_title("EPR time series: known force vs force-free")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(bottom=0)

    # === Panel (d): Integrated EPR bar chart ===
    ax = axes[1, 1]
    panel_label(ax, "(d)")

    bar_labels = [
        "Sekimoto\n(reference)",
        "NF\n(known force)",
        "Continuity\n(force-free)",
    ]
    bar_values = [Sigma_sek, Sigma_nf, Sigma_cont]
    bar_colors = [COLORS["epr_total"], COLORS["epr_nf"], "#FF7F00"]
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
            fontsize=10,
        )

    ax.set_ylabel(r"$\Sigma$ (integrated EPR)")
    ax.set_title("Cycle-integrated EPR comparison")

    # Validation annotation
    textstr = f"NF error: {err_nf:.1f}%\nCont. error: {err_cont:.1f}%\nTarget: < 15%"
    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
    ax.text(
        0.98,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox=props,
    )

    # Pass/Fail indicator
    status = "PASS" if err_cont < 15 else "FAIL"
    status_color = "green" if err_cont < 15 else "red"
    ax.text(
        0.98,
        0.60,
        status,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
        color=status_color,
    )

    plt.tight_layout()
    save_pub(fig, "fig6_km_epr_comparison")

    print("\n" + "=" * 60)
    print("Fig 6 complete: figures/fig6_km_epr_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
