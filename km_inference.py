"""
Kramers-Moyal inference for time-dependent ensemble trajectories.

Estimates drift D₁(x,t) and diffusion D₂(x,t) coefficients from ensemble
particle trajectories at each time snapshot, enabling FORCE-FREE EPR computation.

This module adapts the langevin-inference-validation pipeline for driven systems
where the force field varies with time via a protocol h(t).

Key difference from stationary inference:
  - Stationary: Learn D₁(x), D₂(x) from a single long trajectory
  - Time-dependent: Learn D₁(x,t), D₂(x,t) from ensemble snapshots at each t

Reference: Honisch & Friedrich (2011), Friedrich, Peinke & Sahimi (2011)
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import interp1d


def km_estimate_snapshot(
    x_t: np.ndarray,
    x_t_next: np.ndarray,
    dt: float,
    x_grid: np.ndarray,
    bandwidth: Optional[float] = None,
    min_count: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate Kramers-Moyal coefficients D₁(x) and D₂(x) from a single snapshot.

    Uses kernel-weighted local averaging to estimate:
      D₁(x) = ⟨Δx/Δt | X(t)=x⟩  (drift)
      D₂(x) = ⟨(Δx)²/(2Δt) | X(t)=x⟩  (diffusion)

    Parameters
    ----------
    x_t : ndarray, shape (n_particles,)
        Particle positions at time t.
    x_t_next : ndarray, shape (n_particles,)
        Particle positions at time t + dt.
    dt : float
        Time step between snapshots.
    x_grid : ndarray, shape (n_grid,)
        Spatial grid for evaluation.
    bandwidth : float, optional
        Kernel bandwidth. If None, uses Silverman's rule.
    min_count : int
        Minimum effective count per bin (for error estimation).

    Returns
    -------
    D1 : ndarray, shape (n_grid,)
        Estimated drift at each grid point.
    D2 : ndarray, shape (n_grid,)
        Estimated diffusion at each grid point.
    D1_err : ndarray, shape (n_grid,)
        Standard error of D1 estimate.
    D2_err : ndarray, shape (n_grid,)
        Standard error of D2 estimate.
    """
    # Increments
    dx = x_t_next - x_t

    # Bandwidth via Silverman's rule if not provided
    if bandwidth is None:
        sigma = np.std(x_t)
        n = len(x_t)
        bandwidth = 1.06 * sigma * n ** (-1 / 5)
        bandwidth = max(bandwidth, 0.05)  # floor to avoid numerical issues

    n_grid = len(x_grid)
    D1 = np.zeros(n_grid)
    D2 = np.zeros(n_grid)
    D1_err = np.full(n_grid, np.nan)
    D2_err = np.full(n_grid, np.nan)

    for i, x0 in enumerate(x_grid):
        # Gaussian kernel weights
        w = np.exp(-0.5 * ((x_t - x0) / bandwidth) ** 2)
        w_sum = w.sum()

        if w_sum < min_count:
            D1[i] = np.nan
            D2[i] = np.nan
            continue

        # Weighted moments
        w_norm = w / w_sum
        D1[i] = np.sum(w_norm * dx) / dt
        D2[i] = np.sum(w_norm * dx**2) / (2 * dt)

        # Effective sample size for error estimation
        n_eff = w_sum**2 / np.sum(w**2)
        if n_eff > 2:
            # Weighted variance of dx/dt
            var_dx = np.sum(w_norm * (dx / dt - D1[i]) ** 2)
            D1_err[i] = np.sqrt(var_dx / n_eff)

            # Weighted variance of dx^2/(2dt)
            var_dx2 = np.sum(w_norm * (dx**2 / (2 * dt) - D2[i]) ** 2)
            D2_err[i] = np.sqrt(var_dx2 / n_eff)

    return D1, D2, D1_err, D2_err


def km_estimate_per_snapshot(
    time_grid: np.ndarray,
    ensemble_x: np.ndarray,
    x_grid: np.ndarray,
    snapshot_indices: Optional[np.ndarray] = None,
    bandwidth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate Kramers-Moyal coefficients for each time snapshot.

    Parameters
    ----------
    time_grid : ndarray, shape (n_steps,)
        Full simulation time grid.
    ensemble_x : ndarray, shape (n_particles, n_steps)
        Ensemble particle trajectories.
    x_grid : ndarray, shape (n_x,)
        Spatial grid for D₁, D₂ evaluation.
    snapshot_indices : ndarray, optional
        Indices into time_grid for snapshots. If None, uses evenly spaced.
    bandwidth : float, optional
        Kernel bandwidth. If None, uses adaptive Silverman's rule.

    Returns
    -------
    t_snapshots : ndarray, shape (n_snapshots,)
        Time values at each snapshot.
    D1_grid : ndarray, shape (n_snapshots, n_x)
        Drift coefficient D₁(x,t) at each snapshot and grid point.
    D2_grid : ndarray, shape (n_snapshots, n_x)
        Diffusion coefficient D₂(x,t) at each snapshot and grid point.
    D1_err : ndarray, shape (n_snapshots, n_x)
        Standard error of D1.
    D2_err : ndarray, shape (n_snapshots, n_x)
        Standard error of D2.
    """
    n_steps = time_grid.shape[0]
    dt = time_grid[1] - time_grid[0]

    if snapshot_indices is None:
        # Default: 40 evenly spaced snapshots (matching NF bins)
        snapshot_indices = np.linspace(0, n_steps - 2, 40, dtype=int)

    n_snapshots = len(snapshot_indices)
    n_x = len(x_grid)

    t_snapshots = np.zeros(n_snapshots)
    D1_grid = np.zeros((n_snapshots, n_x))
    D2_grid = np.zeros((n_snapshots, n_x))
    D1_err_grid = np.zeros((n_snapshots, n_x))
    D2_err_grid = np.zeros((n_snapshots, n_x))

    for j, idx in enumerate(snapshot_indices):
        t_snapshots[j] = time_grid[idx]

        # Get particle positions at t and t+dt
        x_t = ensemble_x[:, idx]
        x_t_next = ensemble_x[:, idx + 1]

        D1, D2, D1_err, D2_err = km_estimate_snapshot(
            x_t, x_t_next, dt, x_grid, bandwidth=bandwidth
        )

        D1_grid[j] = D1
        D2_grid[j] = D2
        D1_err_grid[j] = D1_err
        D2_err_grid[j] = D2_err

    return t_snapshots, D1_grid, D2_grid, D1_err_grid, D2_err_grid


def smooth_km_coefficients(
    x_grid: np.ndarray,
    D1_raw: np.ndarray,
    D2_raw: np.ndarray,
    smooth_sigma: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Gaussian smoothing to KM coefficients and interpolate NaN values.

    Parameters
    ----------
    x_grid : ndarray, shape (n_x,)
        Spatial grid.
    D1_raw : ndarray, shape (n_x,)
        Raw drift estimate (may contain NaN).
    D2_raw : ndarray, shape (n_x,)
        Raw diffusion estimate (may contain NaN).
    smooth_sigma : float
        Smoothing bandwidth in x-units.

    Returns
    -------
    D1_smooth : ndarray, shape (n_x,)
    D2_smooth : ndarray, shape (n_x,)
    """
    from scipy.ndimage import gaussian_filter1d

    dx = x_grid[1] - x_grid[0]
    sigma_pix = smooth_sigma / dx

    # Interpolate NaN values
    def fill_nan(arr):
        valid = ~np.isnan(arr)
        if valid.sum() < 2:
            return np.zeros_like(arr)
        interp = interp1d(
            x_grid[valid], arr[valid], kind="linear", fill_value="extrapolate"
        )
        return interp(x_grid)

    D1_filled = fill_nan(D1_raw)
    D2_filled = fill_nan(D2_raw)

    # Apply Gaussian smoothing
    D1_smooth = gaussian_filter1d(D1_filled, sigma_pix)
    D2_smooth = gaussian_filter1d(D2_filled, sigma_pix)

    # Ensure D2 > 0 (physical constraint)
    D2_smooth = np.maximum(D2_smooth, 1e-6)

    return D1_smooth, D2_smooth


class KMInferenceTimeSeries:
    """
    Full Kramers-Moyal inference pipeline for time-dependent ensemble data.

    Provides D₁(x,t) and D₂(x,t) as interpolated functions for use in
    force-free EPR computation.
    """

    def __init__(
        self,
        time_grid: np.ndarray,
        ensemble_x: np.ndarray,
        x_grid: np.ndarray,
        n_snapshots: int = 40,
        bandwidth: Optional[float] = None,
        smooth_sigma: float = 0.1,
    ):
        """
        Parameters
        ----------
        time_grid : ndarray, shape (n_steps,)
            Simulation time grid.
        ensemble_x : ndarray, shape (n_particles, n_steps)
            Ensemble trajectories.
        x_grid : ndarray, shape (n_x,)
            Spatial grid for coefficient evaluation.
        n_snapshots : int
            Number of time snapshots (default 40, matching NF bins).
        bandwidth : float, optional
            Kernel bandwidth for KM estimation.
        smooth_sigma : float
            Spatial smoothing bandwidth.
        """
        self.time_grid = time_grid
        self.ensemble_x = ensemble_x
        self.x_grid = x_grid
        self.n_snapshots = n_snapshots
        self.bandwidth = bandwidth
        self.smooth_sigma = smooth_sigma

        # Run inference
        self._run_inference()

    def _run_inference(self):
        """Run full KM inference pipeline."""
        n_steps = len(self.time_grid)
        snapshot_indices = np.linspace(0, n_steps - 2, self.n_snapshots, dtype=int)

        # Raw KM estimation
        self.t_snapshots, D1_raw, D2_raw, self.D1_err, self.D2_err = (
            km_estimate_per_snapshot(
                self.time_grid,
                self.ensemble_x,
                self.x_grid,
                snapshot_indices=snapshot_indices,
                bandwidth=self.bandwidth,
            )
        )

        # Smooth each snapshot
        n_snap = len(self.t_snapshots)
        n_x = len(self.x_grid)
        self.D1_grid = np.zeros((n_snap, n_x))
        self.D2_grid = np.zeros((n_snap, n_x))

        for j in range(n_snap):
            self.D1_grid[j], self.D2_grid[j] = smooth_km_coefficients(
                self.x_grid, D1_raw[j], D2_raw[j], self.smooth_sigma
            )

        # Store raw for comparison
        self.D1_raw = D1_raw
        self.D2_raw = D2_raw

    def get_D1(self, x: np.ndarray, t_idx: int) -> np.ndarray:
        """Get drift D₁(x) at snapshot index t_idx."""
        return np.interp(x, self.x_grid, self.D1_grid[t_idx])

    def get_D2(self, x: np.ndarray, t_idx: int) -> np.ndarray:
        """Get diffusion D₂(x) at snapshot index t_idx."""
        return np.interp(x, self.x_grid, self.D2_grid[t_idx])

    def get_D1_at_grid(self, t_idx: int) -> np.ndarray:
        """Get drift D₁ on the full x_grid at snapshot t_idx."""
        return self.D1_grid[t_idx].copy()

    def get_D2_at_grid(self, t_idx: int) -> np.ndarray:
        """Get diffusion D₂ on the full x_grid at snapshot t_idx."""
        return self.D2_grid[t_idx].copy()

    def get_mean_D2(self) -> float:
        """Get time-averaged, spatially-averaged diffusion coefficient."""
        return np.nanmean(self.D2_grid)
