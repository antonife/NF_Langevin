"""
EPR computation from learned normalizing flows, plus Sekimoto and Schnakenberg references.

Three EPR methods:
  1. NF-EPR: Train flow per time bin, compute J and sigma from learned p(x|t)
  2. Sekimoto (total): Work/energy balance from ensemble trajectories
  3. Schnakenberg (coarse-grained): Inter-well transition flux counting
"""

import numpy as np
from typing import List, Tuple
from physics import LangevinParams, V, h_protocol
from physics import (
    fokker_planck_current,
    epr_density,
    fokker_planck_current_km,
    epr_density_km,
)
from flows_1d import PlanarFlow1D, train_planar_flow


# ============================================================================
# 1. Time-binned flow training
# ============================================================================


def extract_time_bins(
    time_grid: np.ndarray,
    ensemble_x: np.ndarray,
    n_bins: int = 50,
) -> List[Tuple[float, np.ndarray]]:
    """Bin ensemble snapshots into n_bins time windows.

    Parameters
    ----------
    time_grid : ndarray, shape (n_steps,)
    ensemble_x : ndarray, shape (n_particles, n_steps)
    n_bins : int
        Number of time bins within the last cycle.

    Returns
    -------
    bins : list of (t_center, samples)
        Each entry has the center time and particle positions at that snapshot.
    """
    n_steps = time_grid.shape[0]
    # Use evenly spaced snapshots
    indices = np.linspace(0, n_steps - 1, n_bins, dtype=int)
    bins = []
    for idx in indices:
        t = time_grid[idx]
        samples = ensemble_x[:, idx]
        bins.append((t, samples))
    return bins


def train_flows_per_bin(
    bins: List[Tuple[float, np.ndarray]],
    n_layers: int = 12,
    n_base_components: int = 2,
    seed: int = 42,
    maxiter: int = 400,
    verbose: bool = False,
) -> List[Tuple[float, PlanarFlow1D]]:
    """Train one planar flow per time bin.

    Parameters
    ----------
    bins : list of (t, samples)
    n_layers : int
    n_base_components : int
        GMM base components (2 for bimodal Landau potential).
    seed : int
    maxiter : int
    verbose : bool

    Returns
    -------
    flows : list of (t, flow)
    """
    flows = []
    for i, (t, samples) in enumerate(bins):
        if verbose:
            print(f"  Bin {i + 1}/{len(bins)}: t = {t:.3f}")
        flow = train_planar_flow(
            samples,
            n_layers=n_layers,
            n_base_components=n_base_components,
            seed=seed + i,
            maxiter=maxiter,
            verbose=False,
        )
        flows.append((t, flow))
    return flows


# ============================================================================
# 2. NF-based EPR
# ============================================================================


def compute_epr_from_flows(
    flows: List[Tuple[float, PlanarFlow1D]],
    params: LangevinParams,
    x_grid: np.ndarray,
    p_threshold: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute EPR(t) from trained flows via Fokker-Planck current.

    For each time bin:
      p(x) = exp(log_prob(x))
      dp/dx = p(x) * score(x)
      J = [-V'(x,h)/gamma]*p - D*dp/dx
      sigma(x) = J^2 / (D*p)
      EPR(t) = integral sigma(x) dx

    Parameters
    ----------
    flows : list of (t, flow)
    params : LangevinParams
    x_grid : ndarray
        Spatial grid for integration.
    p_threshold : float
        Probability threshold below which EPR density is zeroed to avoid
        boundary artifacts from low-density regions.

    Returns
    -------
    t_arr : ndarray, shape (n_bins,)
    epr_arr : ndarray, shape (n_bins,)
        Integrated EPR at each time.
    sigma_field : ndarray, shape (n_bins, len(x_grid))
        EPR density field.
    """
    n_bins = len(flows)
    nx = len(x_grid)
    t_arr = np.zeros(n_bins)
    epr_arr = np.zeros(n_bins)
    sigma_field = np.zeros((n_bins, nx))

    for i, (t, flow) in enumerate(flows):
        h = h_protocol(np.array([t]), params)[0]

        # Density and score from flow
        log_p = flow.log_prob(x_grid)
        p = np.exp(log_p)
        s = flow.score(x_grid)
        dp_dx = p * s

        # Fokker-Planck current and EPR density
        J = fokker_planck_current(x_grid, p, dp_dx, h, params)
        sigma = epr_density(J, p, params.D)

        # Zero out EPR in low-density regions to avoid boundary artifacts
        sigma[p < p_threshold] = 0.0

        t_arr[i] = t
        sigma_field[i] = sigma
        epr_arr[i] = np.trapz(sigma, x_grid)

    return t_arr, epr_arr, sigma_field


def compute_epr_continuity(
    flows: List[Tuple[float, PlanarFlow1D]],
    D: float,
    x_grid: np.ndarray,
    p_threshold: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Force-free EPR via the continuity equation: J = -∫ ∂p/∂t dx.

    Instead of computing the Fokker-Planck current from the (unknown) drift,
    exploit the continuity equation ∂p/∂t = -∂J/∂x to obtain J directly
    from the time derivative of the NF-learned densities:

        J(x, t_k) = -∫_{-∞}^{x} [∂p/∂t](x', t_k) dx'

    This requires NO knowledge of the force field — only the diffusion
    coefficient D and the trained normalizing flows.

    Parameters
    ----------
    flows : list of (t, flow)
        Trained normalizing flows per time bin (sorted by time).
    D : float
        Diffusion coefficient (constant). Can be learned from KM or known.
    x_grid : ndarray
        Spatial grid for integration.
    p_threshold : float
        Probability threshold below which EPR density is zeroed.

    Returns
    -------
    t_arr : ndarray, shape (n_bins,)
    epr_arr : ndarray, shape (n_bins,)
        Integrated EPR at each time bin.
    sigma_field : ndarray, shape (n_bins, len(x_grid))
        EPR density field σ(x, t).
    """
    from scipy.integrate import cumulative_trapezoid

    n_bins = len(flows)
    nx = len(x_grid)

    # Step 1: evaluate p(x, t_k) for all bins on the shared x_grid
    p_all = np.zeros((n_bins, nx))
    t_arr = np.zeros(n_bins)
    for i, (t, flow) in enumerate(flows):
        t_arr[i] = t
        p_all[i] = np.exp(flow.log_prob(x_grid))

    # Step 2: compute ∂p/∂t via centered finite differences
    dpdt = np.zeros_like(p_all)
    for k in range(n_bins):
        if k == 0:
            # Forward difference at first bin
            dt = t_arr[1] - t_arr[0]
            dpdt[k] = (p_all[1] - p_all[0]) / dt
        elif k == n_bins - 1:
            # Backward difference at last bin
            dt = t_arr[-1] - t_arr[-2]
            dpdt[k] = (p_all[-1] - p_all[-2]) / dt
        else:
            # Centered difference
            dt = t_arr[k + 1] - t_arr[k - 1]
            dpdt[k] = (p_all[k + 1] - p_all[k - 1]) / dt

    # Step 3: enforce conservation (∫ ∂p/∂t dx = 0) by subtracting residual
    for k in range(n_bins):
        residual = np.trapz(dpdt[k], x_grid)
        dpdt[k] -= residual / (x_grid[-1] - x_grid[0])

    # Step 4: compute J(x, t_k) = -∫_{-∞}^{x} ∂p/∂t dx'
    epr_arr = np.zeros(n_bins)
    sigma_field = np.zeros((n_bins, nx))

    for k in range(n_bins):
        J = -cumulative_trapezoid(dpdt[k], x_grid, initial=0.0)

        p = p_all[k]
        sigma = np.zeros(nx)
        valid = p > p_threshold
        sigma[valid] = J[valid] ** 2 / (D * p[valid])

        sigma_field[k] = sigma
        epr_arr[k] = np.trapz(sigma, x_grid)

    return t_arr, epr_arr, sigma_field


# ============================================================================
# 3. Sekimoto EPR (total dissipation from work/energy balance)
# ============================================================================


def sekimoto_epr(
    time_grid: np.ndarray,
    ensemble_x: np.ndarray,
    params: LangevinParams,
    smooth_win: int = 100,
) -> Tuple[np.ndarray, float]:
    """Total EPR via Sekimoto stochastic energetics.

    EPR_total = (dW/dt - dE/dt) / kT
    where dW/dt = <x> * dh/dt (input power) and dE/dt is the internal energy rate.

    Parameters
    ----------
    time_grid : ndarray, shape (n_steps,)
    ensemble_x : ndarray, shape (n_particles, n_steps)
    params : LangevinParams
    smooth_win : int
        Smoothing window for dE/dt.

    Returns
    -------
    EPR_total : ndarray, shape (n_steps,)
        Instantaneous total EPR.
    Sigma_total : float
        Integrated total entropy production over the time window.
    """
    dt = time_grid[1] - time_grid[0]
    h_t = h_protocol(time_grid, params)

    # Ensemble-averaged position and internal energy
    xi = np.mean(ensemble_x, axis=0)
    E = np.mean(V(ensemble_x, h_t[np.newaxis, :]), axis=0)

    # Protocol rate
    h_dot = (
        params.h_max
        * (2 * np.pi / params.Period)
        * np.cos(2 * np.pi * time_grid / params.Period)
    )

    # dE/dt with smoothing
    dE_dt = np.gradient(E, dt)
    kernel = np.ones(smooth_win) / smooth_win
    dE_dt_smooth = np.convolve(dE_dt, kernel, mode="same")

    # Dissipation rate and EPR
    Q_diss_rate = xi * h_dot - dE_dt_smooth
    EPR_total = Q_diss_rate / params.kT

    Sigma_total = np.trapz(EPR_total, time_grid)
    return EPR_total, Sigma_total


# ============================================================================
# 4. Schnakenberg EPR (coarse-grained, inter-well transitions)
# ============================================================================


def schnakenberg_epr(
    time_grid: np.ndarray,
    ensemble_x: np.ndarray,
    params: LangevinParams,
    window_sec: float = 0.4,
) -> Tuple[np.ndarray, float]:
    """Coarse-grained EPR via Schnakenberg formula on L/R well transitions.

    EPR_CG = (J_net / dt) * ln(J_01 / J_10)

    Parameters
    ----------
    time_grid : ndarray, shape (n_steps,)
    ensemble_x : ndarray, shape (n_particles, n_steps)
    params : LangevinParams
    window_sec : float
        Smoothing window in seconds.

    Returns
    -------
    EPR_cg : ndarray, shape (n_steps,)
        Coarse-grained EPR.
    Sigma_cg : float
        Integrated CG entropy production.
    """
    dt = time_grid[1] - time_grid[0]
    n_steps = len(time_grid)
    n_particles = ensemble_x.shape[0]

    window_steps = int(window_sec / dt)
    kernel = np.ones(window_steps) / window_steps

    # Transition counting
    is_R = ensemble_x >= 0
    flux_01 = np.sum((~is_R[:, :-1]) & is_R[:, 1:], axis=0)  # L -> R
    flux_10 = np.sum(is_R[:, :-1] & (~is_R[:, 1:]), axis=0)  # R -> L

    # Pad to n_steps
    flux_01 = np.append(flux_01, 0)
    flux_10 = np.append(flux_10, 0)

    # Smooth
    J_01 = np.convolve(flux_01, kernel, mode="same")
    J_10 = np.convolve(flux_10, kernel, mode="same")

    J_net = (J_01 - J_10) / n_particles
    flux_force = np.zeros(n_steps)
    mask = (J_01 > 1e-9) & (J_10 > 1e-9)
    flux_force[mask] = np.log(J_01[mask] / J_10[mask])

    EPR_cg = (J_net / dt) * flux_force

    # Integrate excluding edge effects
    crop = window_steps
    if crop < n_steps // 2:
        EPR_valid = EPR_cg[crop:-crop]
        t_valid = time_grid[crop:-crop]
        Sigma_cg = np.trapz(EPR_valid, t_valid)
        # Scale to full period
        Sigma_cg *= (time_grid[-1] - time_grid[0]) / (t_valid[-1] - t_valid[0])
    else:
        Sigma_cg = np.trapz(EPR_cg, time_grid)

    return EPR_cg, Sigma_cg


# ============================================================================
# 5. Force-free EPR via Kramers-Moyal learned coefficients
# ============================================================================


def compute_epr_from_flows_km(
    flows: List[Tuple[float, PlanarFlow1D]],
    km_inference,
    x_grid: np.ndarray,
    use_mean_D2: bool = True,
    p_threshold: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute EPR(t) from trained flows using LEARNED drift and diffusion.

    This is the FORCE-FREE variant: instead of requiring -V'(x,h)/γ,
    we use D₁(x,t) and D₂(x,t) estimated from trajectory data via
    Kramers-Moyal.

    For each time bin:
      p(x) = exp(log_prob(x))  [from normalizing flow]
      dp/dx = p(x) * score(x)   [from normalizing flow]
      J = D₁(x,t)·p - D₂(x,t)·dp/dx  [using LEARNED coefficients]
      σ(x) = J² / (D₂·p)
      EPR(t) = ∫ σ(x) dx

    Parameters
    ----------
    flows : list of (t, flow)
        Trained normalizing flows per time bin.
    km_inference : KMInferenceTimeSeries
        Object providing D₁(x,t) and D₂(x,t) via .get_D1_at_grid(t_idx)
        and .get_D2_at_grid(t_idx).
    x_grid : ndarray
        Spatial grid for integration.
    use_mean_D2 : bool
        If True, use time-averaged mean D₂ (more stable, appropriate for
        constant-diffusion systems like overdamped Langevin).
    p_threshold : float
        Probability threshold below which EPR density is zeroed to avoid
        boundary artifacts from low-density regions.

    Returns
    -------
    t_arr : ndarray, shape (n_bins,)
    epr_arr : ndarray, shape (n_bins,)
        Integrated EPR at each time.
    sigma_field : ndarray, shape (n_bins, len(x_grid))
        EPR density field.
    """
    n_bins = len(flows)
    nx = len(x_grid)
    t_arr = np.zeros(n_bins)
    epr_arr = np.zeros(n_bins)
    sigma_field = np.zeros((n_bins, nx))

    # Use constant D₂ for stability (physically correct for overdamped Langevin)
    D2_const = km_inference.get_mean_D2() if use_mean_D2 else None

    for i, (t, flow) in enumerate(flows):
        # Density and score from flow
        log_p = flow.log_prob(x_grid)
        p = np.exp(log_p)
        s = flow.score(x_grid)
        dp_dx = p * s

        # Get learned drift at this snapshot
        D1 = km_inference.get_D1_at_grid(i)
        D1_interp = np.interp(x_grid, km_inference.x_grid, D1)

        # Use constant D₂ or interpolated D₂
        if use_mean_D2:
            D2_interp = np.full(nx, D2_const)
        else:
            D2 = km_inference.get_D2_at_grid(i)
            D2_interp = np.interp(x_grid, km_inference.x_grid, D2)

        # Fokker-Planck current and EPR density using learned coefficients
        J = fokker_planck_current_km(x_grid, p, dp_dx, D1_interp, D2_interp)
        sigma = epr_density_km(J, p, D2_interp)

        # Zero out EPR in low-density regions to avoid boundary artifacts
        sigma[p < p_threshold] = 0.0

        t_arr[i] = t
        sigma_field[i] = sigma
        epr_arr[i] = np.trapz(sigma, x_grid)

    return t_arr, epr_arr, sigma_field


# ============================================================================
# 6. Force-free EPR via pooled Kramers-Moyal + cross-validated debiasing
# ============================================================================


def compute_epr_ensemble_km(
    flows: List[Tuple[float, PlanarFlow1D]],
    time_grid: np.ndarray,
    ensemble_x: np.ndarray,
    D2: float,
    x_km_grid: np.ndarray,
    n_avg: int = 100,
    bandwidth: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute EPR(t) via pooled multi-step KM drift + NF score (ensemble avg).

    For each time bin, pools n_avg consecutive timesteps of (x, dx) pairs
    to reduce KM drift variance, then computes EPR via the ensemble-averaged
    irreversible velocity formula with split-half cross-validation to
    eliminate noise bias.

    EPR(t) = <v_irrev_A * v_irrev_B / D2>_particles

    where v_irrev = D1(x) - D2*score(x) and A, B are independent half-samples.

    Parameters
    ----------
    flows : list of (t, flow)
        Trained normalizing flows per time bin.
    time_grid : ndarray, shape (n_steps,)
        Simulation time grid (same cycle used for flow training).
    ensemble_x : ndarray, shape (n_particles, n_steps)
        Ensemble trajectories for the cycle.
    D2 : float
        Diffusion coefficient (constant, learned or known).
    x_km_grid : ndarray
        Spatial grid for KM kernel regression.
    n_avg : int
        Number of consecutive timesteps to pool per snapshot.
    bandwidth : float
        Kernel bandwidth for KM estimation.

    Returns
    -------
    t_arr : ndarray, shape (n_bins,)
    epr_arr : ndarray, shape (n_bins,)
        Cross-validated EPR rate at each time bin.
    """
    from km_inference import km_estimate_snapshot

    n_bins = len(flows)
    n_steps = len(time_grid)
    dt = time_grid[1] - time_grid[0]
    t_arr = np.zeros(n_bins)
    epr_arr = np.zeros(n_bins)

    # Snapshot indices matching flow time bins
    flow_times = np.array([t for t, _ in flows])
    snap_indices = np.array([np.argmin(np.abs(time_grid - ft)) for ft in flow_times])

    rng = np.random.default_rng(123)

    for i, (t_flow, flow) in enumerate(flows):
        idx = snap_indices[i]

        # Pool n_avg consecutive timesteps centered on snapshot
        half_win = n_avg // 2
        start = max(0, idx - half_win)
        end = min(n_steps - 1, start + n_avg)
        start = max(0, end - n_avg)

        # Collect all (x, dx) pairs
        all_x = []
        all_dx = []
        for step in range(start, end):
            all_x.append(ensemble_x[:, step])
            all_dx.append(ensemble_x[:, step + 1] - ensemble_x[:, step])
        all_x = np.concatenate(all_x)
        all_dx = np.concatenate(all_dx)

        # Split-half cross-validation
        n_total = len(all_x)
        perm = np.arange(n_total)
        rng.shuffle(perm)
        A, B = perm[: n_total // 2], perm[n_total // 2 :]

        # KM drift from each half
        D1_A, _, _, _ = km_estimate_snapshot(
            all_x[A], all_x[A] + all_dx[A], dt, x_km_grid, bandwidth=bandwidth
        )
        D1_B, _, _, _ = km_estimate_snapshot(
            all_x[B], all_x[B] + all_dx[B], dt, x_km_grid, bandwidth=bandwidth
        )
        D1_A = np.nan_to_num(D1_A, nan=0.0)
        D1_B = np.nan_to_num(D1_B, nan=0.0)

        # Evaluate at particle positions for ensemble averaging
        particles = ensemble_x[:, idx]
        score_p = flow.score(particles)
        D1_A_p = np.interp(particles, x_km_grid, D1_A)
        D1_B_p = np.interp(particles, x_km_grid, D1_B)

        # Cross-validated irreversible velocities
        v_A = D1_A_p - D2 * score_p
        v_B = D1_B_p - D2 * score_p

        # Unbiased EPR: E[v_A * v_B] = v_true^2 (no noise^2 term)
        epr_cross = np.mean(v_A * v_B) / D2

        t_arr[i] = t_flow
        epr_arr[i] = max(epr_cross, 0.0)  # EPR is non-negative

    return t_arr, epr_arr
