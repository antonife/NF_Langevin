"""
1D Normalizing Flows for density estimation on Langevin trajectory data.

Implements:
  1. PlanarFlow1D — stacked g(x) = x + u*tanh(w*x+b), invertible via Newton
  2. NeuralODE1D — dx/dt = F(x,t), vectorized over N samples

Both support a Gaussian mixture base distribution (essential for bimodal targets).
Trained with scipy.optimize.minimize (L-BFGS-B). Pure NumPy/SciPy.

Reference: Kobyzev, Prince, Brubaker (2020), arXiv:1908.09257v4
  - Planar: Sec. 3.3.1, Eqs. 10-11
  - Neural ODE: Sec. 3.6.1, Eqs. 33, 36
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
from typing import Tuple


# ============================================================================
# Gaussian Mixture Base Distribution
# ============================================================================


def _gmm_log_prob(z: np.ndarray, base_params: np.ndarray, K: int) -> np.ndarray:
    """Log-probability of a K-component Gaussian mixture.

    Parameters
    ----------
    z : ndarray, shape (N,)
    base_params : ndarray
        [logit_weights(K-1), means(K), log_stds(K)] = 3K - 1 params.
    K : int

    Returns
    -------
    log_p : ndarray, shape (N,)
    """
    logit_w = base_params[: K - 1]
    mus = base_params[K - 1 : 2 * K - 1]
    log_sigmas = base_params[2 * K - 1 : 3 * K - 1]
    sigmas = np.exp(log_sigmas)

    # Softmax weights
    logit_full = np.concatenate([logit_w, [0.0]])  # last component is reference
    log_w = logit_full - logsumexp(logit_full)

    # log p(z) = logsumexp_k [ log w_k + log N(z; mu_k, sigma_k^2) ]
    N = z.shape[0]
    log_components = np.zeros((K, N))
    for k in range(K):
        log_components[k] = (
            log_w[k]
            - 0.5 * np.log(2 * np.pi)
            - log_sigmas[k]
            - 0.5 * ((z - mus[k]) / sigmas[k]) ** 2
        )
    return logsumexp(log_components, axis=0)


# ============================================================================
# 1. Stacked Planar Flow (1D)
# ============================================================================


class PlanarFlow1D:
    """Stacked 1D planar flow: g(x) = x + u*tanh(w*x + b).

    In 1D the Jacobian is a scalar: dg/dx = 1 + u*w*sech^2(w*x + b).
    Monotonicity requires u*w > -1.

    Supports an optional K-component Gaussian mixture base distribution
    (essential for bimodal targets like the Landau potential at h~0).

    Parameters are stored as a flat vector for scipy optimization:
      [u_0, w_0, b_0, ..., u_{L-1}, w_{L-1}, b_{L-1}, base_params...]
    """

    def __init__(self, n_layers: int = 12, n_base_components: int = 1, seed: int = 42):
        self.n_layers = n_layers
        self.n_base = n_base_components
        rng = np.random.default_rng(seed)

        # Flow layer params: 3 per layer
        n_flow = 3 * n_layers
        flow_params = np.zeros(n_flow)
        for k in range(n_layers):
            flow_params[3 * k] = rng.normal(0, 0.1)  # u
            flow_params[3 * k + 1] = rng.normal(0, 0.3)  # w
            flow_params[3 * k + 2] = rng.normal(0, 0.05)  # b

        # Base distribution params (only if GMM)
        if n_base_components > 1:
            K = n_base_components
            n_base_params = 3 * K - 1
            base_params = np.zeros(n_base_params)
            # Init: equal weights (logits=0), means spread, moderate stds
            # logit_weights: K-1 zeros (equal weights via softmax)
            # means: linearly spaced from -1 to +1
            means = np.linspace(-1.0, 1.0, K)
            base_params[K - 1 : 2 * K - 1] = means
            # log_stds: log(0.5) for each component
            base_params[2 * K - 1 : 3 * K - 1] = np.log(0.5)
            self._params = np.concatenate([flow_params, base_params])
        else:
            self._params = flow_params

    @property
    def n_flow_params(self) -> int:
        return 3 * self.n_layers

    @property
    def n_base_params(self) -> int:
        if self.n_base > 1:
            return 3 * self.n_base - 1
        return 0

    def get_params(self) -> np.ndarray:
        return self._params.copy()

    def set_params(self, params: np.ndarray) -> None:
        """Set parameters with monotonicity enforcement: u*w > -1 + eps."""
        self._params = params.copy()
        eps = 0.01
        for k in range(self.n_layers):
            u = self._params[3 * k]
            w = self._params[3 * k + 1]
            if u * w <= -1 + eps:
                if w != 0:
                    self._params[3 * k] = (1.0 + eps) / abs(w) * np.sign(w)
                else:
                    self._params[3 * k] = 0.1

    def _get_layer(self, k: int) -> Tuple[float, float, float]:
        return self._params[3 * k], self._params[3 * k + 1], self._params[3 * k + 2]

    def _base_log_prob(self, z: np.ndarray) -> np.ndarray:
        """Log-probability under the base distribution."""
        if self.n_base > 1:
            base_p = self._params[self.n_flow_params :]
            return _gmm_log_prob(z, base_p, self.n_base)
        else:
            return -0.5 * z**2 - 0.5 * np.log(2 * np.pi)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass (generative): z -> y.

        Returns (y, log_det_jacobian) where both have shape (N,).
        """
        y = x.copy()
        log_det = np.zeros_like(x)
        for k in range(self.n_layers):
            u, w, b = self._get_layer(k)
            a = w * y + b
            t = np.tanh(a)
            y = y + u * t
            sech2 = 1 - t**2
            dg_dx = 1 + u * w * sech2
            log_det += np.log(np.abs(dg_dx))
        return y, log_det

    def inverse(
        self, y: np.ndarray, max_iter: int = 50, tol: float = 1e-12
    ) -> np.ndarray:
        """Inverse pass (normalizing): y -> z via Newton's method per layer (reversed)."""
        x = y.copy()
        for k in range(self.n_layers - 1, -1, -1):
            u, w, b = self._get_layer(k)
            target = x.copy()
            x_k = target.copy()
            for _ in range(max_iter):
                a = w * x_k + b
                t = np.tanh(a)
                f = x_k + u * t - target
                sech2 = 1 - t**2
                df = 1 + u * w * sech2
                step = f / df
                x_k = x_k - step
                if np.max(np.abs(step)) < tol:
                    break
            x = x_k
        return x

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        """log p(y) = log p_Z(f^{-1}(y)) - log|dy/dz|."""
        z = self.inverse(y)
        _, log_det_fwd = self.forward(z)
        log_pz = self._base_log_prob(z)
        return log_pz - log_det_fwd

    def score(self, y: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Score function d(log p)/dy via central finite differences."""
        lp_plus = self.log_prob(y + eps)
        lp_minus = self.log_prob(y - eps)
        return (lp_plus - lp_minus) / (2 * eps)


# ============================================================================
# 2. Neural ODE Flow (1D)
# ============================================================================


class NeuralODE1D:
    """1D Neural ODE flow: dx/dt = F(x, t).

    In 1D: d(log p)/dt = -dF/dx (scalar, Eq. 36).
    Vectorized: packs N samples as state[0:N] = x, state[N:2N] = log_det.

    Supports an optional K-component Gaussian mixture base distribution.

    MLP architecture: (x, t) -> hidden -> F(x,t)
      W1: (2, hidden_dim), b1: (hidden_dim,)
      W2: (hidden_dim, 1), b2: (1,)
    """

    def __init__(
        self, hidden_dim: int = 16, n_base_components: int = 1, seed: int = 42
    ):
        self.hidden_dim = hidden_dim
        self.n_base = n_base_components
        rng = np.random.default_rng(seed)
        scale = 0.1
        self.W1 = rng.normal(0, scale, (2, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, scale, (hidden_dim, 1))
        self.b2 = np.zeros(1)

    @property
    def n_mlp_params(self) -> int:
        return 2 * self.hidden_dim + self.hidden_dim + 1

    @property
    def n_base_params(self) -> int:
        if self.n_base > 1:
            return 3 * self.n_base - 1
        return 0

    @property
    def n_params(self) -> int:
        return self.n_mlp_params + self.n_base_params

    def get_params(self) -> np.ndarray:
        mlp = np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])
        if self.n_base > 1:
            return np.concatenate([mlp, self._base_params])
        return mlp

    def set_params(self, params: np.ndarray) -> None:
        hd = self.hidden_dim
        idx = 0
        self.W1 = params[idx : idx + 2 * hd].reshape(2, hd)
        idx += 2 * hd
        self.b1 = params[idx : idx + hd].copy()
        idx += hd
        self.W2 = params[idx : idx + hd].reshape(hd, 1)
        idx += hd
        self.b2 = params[idx : idx + 1].copy()
        idx += 1
        if self.n_base > 1:
            self._base_params = params[idx:].copy()

    def _init_base_params(self) -> None:
        """Initialize GMM base params if needed."""
        if self.n_base > 1 and not hasattr(self, "_base_params"):
            K = self.n_base
            self._base_params = np.zeros(3 * K - 1)
            means = np.linspace(-1.0, 1.0, K)
            self._base_params[K - 1 : 2 * K - 1] = means
            self._base_params[2 * K - 1 : 3 * K - 1] = np.log(0.5)

    def _base_log_prob(self, z: np.ndarray) -> np.ndarray:
        if self.n_base > 1:
            self._init_base_params()
            return _gmm_log_prob(z, self._base_params, self.n_base)
        else:
            return -0.5 * z**2 - 0.5 * np.log(2 * np.pi)

    def _F_batch(self, x: np.ndarray, t: float) -> np.ndarray:
        """Compute F(x, t) for a batch of x values. Returns shape (N,)."""
        N = x.shape[0]
        xt = np.column_stack([x, np.full(N, t)])
        h = np.tanh(xt @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).ravel()

    def _dF_dx_batch(self, x: np.ndarray, t: float) -> np.ndarray:
        """Analytical dF/dx for batch. Returns shape (N,)."""
        N = x.shape[0]
        xt = np.column_stack([x, np.full(N, t)])
        a = xt @ self.W1 + self.b1
        sech2 = 1 - np.tanh(a) ** 2
        w1_x = self.W1[0, :]
        w2 = self.W2[:, 0]
        return (sech2 * w1_x * w2).sum(axis=1)

    def _augmented_dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Augmented ODE: [dx/dt, d(logdet)/dt]."""
        N = state.shape[0] // 2
        x = state[:N]
        dxdt = self._F_batch(x, t)
        dlogdet_dt = -self._dF_dx_batch(x, t)
        return np.concatenate([dxdt, dlogdet_dt])

    def forward(
        self, z: np.ndarray, atol: float = 1e-5, rtol: float = 1e-5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generative: z -> y (integrate t=0 to t=1). Returns (y, log_det)."""
        N = z.shape[0]
        state0 = np.concatenate([z, np.zeros(N)])
        sol = solve_ivp(
            self._augmented_dynamics,
            [0, 1],
            state0,
            method="RK45",
            atol=atol,
            rtol=rtol,
        )
        y = sol.y[:N, -1]
        log_det = sol.y[N:, -1]
        return y, log_det

    def inverse(
        self, y: np.ndarray, atol: float = 1e-5, rtol: float = 1e-5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalizing: y -> z (integrate t=1 to t=0). Returns (z, log_det)."""
        N = y.shape[0]
        state0 = np.concatenate([y, np.zeros(N)])
        sol = solve_ivp(
            self._augmented_dynamics,
            [1, 0],
            state0,
            method="RK45",
            atol=atol,
            rtol=rtol,
        )
        z = sol.y[:N, -1]
        log_det = sol.y[N:, -1]
        return z, log_det

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        """log p(y) via inverse pass."""
        z, log_det = self.inverse(y)
        log_pz = self._base_log_prob(z)
        return log_pz + log_det

    def score(self, y: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Score d(log p)/dy via central finite differences."""
        lp_plus = self.log_prob(y + eps)
        lp_minus = self.log_prob(y - eps)
        return (lp_plus - lp_minus) / (2 * eps)


# ============================================================================
# 3. Training Functions
# ============================================================================


def train_planar_flow(
    data: np.ndarray,
    n_layers: int = 12,
    n_base_components: int = 1,
    seed: int = 42,
    maxiter: int = 500,
    verbose: bool = True,
) -> PlanarFlow1D:
    """Train a PlanarFlow1D by maximizing log-likelihood via L-BFGS-B.

    Parameters
    ----------
    data : ndarray, shape (N,)
        Training samples.
    n_layers : int
        Number of planar layers.
    n_base_components : int
        Number of GMM base components (1 = standard normal, 2 = bimodal).
    seed : int
        RNG seed for initialization.
    maxiter : int
        Maximum optimizer iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    flow : PlanarFlow1D
        Trained flow.
    """
    flow = PlanarFlow1D(
        n_layers=n_layers, n_base_components=n_base_components, seed=seed
    )
    iter_count = [0]
    last_nll = [np.inf]

    def neg_log_likelihood(params: np.ndarray) -> float:
        flow.set_params(params)
        ll = flow.log_prob(data)
        nll = -np.mean(ll)
        if np.isnan(nll) or np.isinf(nll):
            return 1e10
        last_nll[0] = nll
        return nll

    def callback(params: np.ndarray) -> None:
        iter_count[0] += 1
        if verbose and iter_count[0] % 50 == 0:
            print(f"  iter {iter_count[0]:4d}: NLL = {last_nll[0]:.4f}")

    p0 = flow.get_params()
    result = minimize(
        neg_log_likelihood,
        p0,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
        callback=callback,
    )
    flow.set_params(result.x)
    if verbose:
        print(f"  Optimization: {result.message}, NLL = {result.fun:.4f}")
    return flow


def train_neural_ode(
    data: np.ndarray,
    hidden_dim: int = 16,
    n_base_components: int = 1,
    seed: int = 42,
    maxiter: int = 300,
    verbose: bool = True,
) -> NeuralODE1D:
    """Train a NeuralODE1D by maximizing log-likelihood via L-BFGS-B.

    Parameters
    ----------
    data : ndarray, shape (N,)
        Training samples.
    hidden_dim : int
        MLP hidden layer width.
    n_base_components : int
        Number of GMM base components (1 = standard normal, 2 = bimodal).
    seed : int
        RNG seed.
    maxiter : int
        Maximum optimizer iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    flow : NeuralODE1D
        Trained flow.
    """
    flow = NeuralODE1D(
        hidden_dim=hidden_dim, n_base_components=n_base_components, seed=seed
    )
    if n_base_components > 1:
        flow._init_base_params()
    n_calls = [0]

    def neg_log_likelihood(params: np.ndarray) -> float:
        flow.set_params(params)
        ll = flow.log_prob(data)
        nll = -np.mean(ll)
        if np.isnan(nll) or np.isinf(nll):
            return 1e10
        if verbose and n_calls[0] % 20 == 0:
            print(f"  iter {n_calls[0]:4d}: NLL = {nll:.4f}")
        n_calls[0] += 1
        return nll

    def nll_grad(params: np.ndarray) -> np.ndarray:
        eps = 1e-5
        grad = np.zeros_like(params)
        f0 = neg_log_likelihood(params)
        n_calls[0] -= 1
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            grad[i] = (neg_log_likelihood(params_plus) - f0) / eps
            n_calls[0] -= 1
        return grad

    p0 = flow.get_params()
    result = minimize(
        neg_log_likelihood,
        p0,
        jac=nll_grad,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
    )
    flow.set_params(result.x)
    if verbose:
        print(f"  Optimization: {result.message}, NLL = {result.fun:.4f}")
    return flow
