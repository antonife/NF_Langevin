"""
Langevin dynamics in a bistable Landau potential V(x,h) = x^4 - 2x^2 + hx.

Refactored from src/langevin_sim.py and Landau_Theory/scripts/fig4_epr_decomposition_series.py.
Provides ensemble simulation, Boltzmann density, and Fokker-Planck current/EPR utilities.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class LangevinParams:
    """Physical and numerical parameters for the Langevin simulation."""

    gamma: float = 1.0  # friction coefficient
    kT: float = 0.5  # thermal energy
    dt: float = 0.002  # Euler-Maruyama time step
    h_max: float = 2.0  # protocol amplitude
    Period: float = 10.0  # driving period

    @property
    def D(self) -> float:
        """Diffusion coefficient D = kT/gamma."""
        return self.kT / self.gamma


def V(x: np.ndarray, h: float) -> np.ndarray:
    """Landau potential V(x,h) = x^4 - 2x^2 + hx."""
    return x**4 - 2 * x**2 + h * x


def grad_V(x: np.ndarray, h: float) -> np.ndarray:
    """Gradient dV/dx = 4x^3 - 4x + h."""
    return 4 * x**3 - 4 * x + h


def h_protocol(t: np.ndarray, params: LangevinParams) -> np.ndarray:
    """Sinusoidal tilting protocol h(t) = h_max * sin(2*pi*t/Period)."""
    return params.h_max * np.sin(2 * np.pi * t / params.Period)


def simulate_ensemble(
    n_particles: int,
    n_cycles: int,
    params: LangevinParams,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Euler-Maruyama ensemble simulation.

    Parameters
    ----------
    n_particles : int
        Number of particles in the ensemble.
    n_cycles : int
        Number of driving periods to simulate.
    params : LangevinParams
        Physical and numerical parameters.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    time_grid : ndarray, shape (n_steps,)
    ensemble_x : ndarray, shape (n_particles, n_steps)
    """
    total_time = n_cycles * params.Period
    n_steps = int(total_time / params.dt)
    time_grid = np.linspace(0, total_time, n_steps)

    ensemble_x = np.zeros((n_particles, n_steps))
    # Start particles in both wells
    ensemble_x[:, 0] = rng.choice([-1.0, 1.0], size=n_particles)

    sqrt_dt = np.sqrt(params.dt)
    noise_scale = np.sqrt(2 * params.gamma * params.kT)

    for i in range(1, n_steps):
        h = h_protocol(time_grid[i - 1], params)
        x = ensemble_x[:, i - 1]
        force = -grad_V(x, h)
        noise = rng.normal(0, 1, n_particles)
        ensemble_x[:, i] = (
            x + (force / params.gamma) * params.dt + noise_scale * noise * sqrt_dt
        )

    return time_grid, ensemble_x


def boltzmann_density(x_grid: np.ndarray, h: float, kT: float) -> np.ndarray:
    """Equilibrium Boltzmann density p(x) = exp(-V(x,h)/kT) / Z.

    Parameters
    ----------
    x_grid : ndarray
        Evaluation points.
    h : float
        Tilt parameter.
    kT : float
        Thermal energy.

    Returns
    -------
    p : ndarray
        Normalized density on x_grid.
    """
    log_p = -V(x_grid, h) / kT
    log_p -= log_p.max()  # numerical stability
    p = np.exp(log_p)
    # Trapezoidal normalization
    Z = np.trapz(p, x_grid)
    return p / Z


def fokker_planck_current(
    x: np.ndarray,
    p: np.ndarray,
    dp_dx: np.ndarray,
    h: float,
    params: LangevinParams,
) -> np.ndarray:
    """Fokker-Planck probability current J(x,t).

    J = [-V'(x,h)/gamma] * p(x) - D * dp/dx

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    p : ndarray
        Probability density p(x).
    dp_dx : ndarray
        Spatial derivative of p.
    h : float
        Tilt parameter at current time.
    params : LangevinParams

    Returns
    -------
    J : ndarray
        Probability current.
    """
    drift = -grad_V(x, h) / params.gamma
    return drift * p - params.D * dp_dx


def epr_density(
    J: np.ndarray, p: np.ndarray, D: float, p_min: float = 1e-10
) -> np.ndarray:
    """Local entropy production rate density sigma(x) = J^2 / (D * p).

    Parameters
    ----------
    J : ndarray
        Probability current.
    p : ndarray
        Probability density.
    D : float
        Diffusion coefficient.
    p_min : float
        Floor to prevent division by zero.

    Returns
    -------
    sigma : ndarray
        EPR density field.
    """
    p_safe = np.maximum(p, p_min)
    return J**2 / (D * p_safe)


# ============================================================================
# Force-free (Kramers-Moyal) variants
# ============================================================================


def fokker_planck_current_km(
    x: np.ndarray,
    p: np.ndarray,
    dp_dx: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
) -> np.ndarray:
    """Fokker-Planck probability current using LEARNED drift and diffusion.

    J = D₁(x)·p(x) - D₂(x)·∂p/∂x

    This is the FORCE-FREE variant: instead of computing drift from
    -V'(x,h)/γ (which requires knowing V), we use D₁ estimated directly
    from trajectory data via Kramers-Moyal.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    p : ndarray
        Probability density p(x).
    dp_dx : ndarray
        Spatial derivative of p.
    D1 : ndarray
        Learned drift coefficient D₁(x) = ⟨Δx/Δt | x⟩.
    D2 : ndarray
        Learned diffusion coefficient D₂(x) = ⟨(Δx)²/(2Δt) | x⟩.

    Returns
    -------
    J : ndarray
        Probability current.
    """
    return D1 * p - D2 * dp_dx


def epr_density_km(
    J: np.ndarray, p: np.ndarray, D2: np.ndarray, p_min: float = 1e-10
) -> np.ndarray:
    """Local EPR density using spatially-varying diffusion D₂(x).

    σ(x) = J²(x) / (D₂(x)·p(x))

    Parameters
    ----------
    J : ndarray
        Probability current.
    p : ndarray
        Probability density.
    D2 : ndarray
        Diffusion coefficient field D₂(x).
    p_min : float
        Floor to prevent division by zero.

    Returns
    -------
    sigma : ndarray
        EPR density field.
    """
    p_safe = np.maximum(p, p_min)
    D2_safe = np.maximum(D2, 1e-10)
    return J**2 / (D2_safe * p_safe)
