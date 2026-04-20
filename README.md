# NF_Langevin

**Recovering Hidden Entropy Production via Normalizing Flows in Driven Langevin Dynamics**

## Overview

This project uses normalizing flows to estimate entropy production rate (EPR) in periodically driven bistable Langevin systems. The key insight is that coarse-grained methods (Schnakenberg) capture only ~5% of the total dissipation, while the remaining ~95% is "hidden" in intrawell probability currents.

We develop two methods:
1. **NF-EPR** (known force): Learn p(x,t) with NFs, reconstruct J(x,t) via Fokker-Planck
2. **Force-Free** (continuity equation): Compute J from ∂p/∂t = -∂J/∂x without knowing V'(x,h)

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core implementation | Complete | src/physics.py, src/flows_1d.py, src/epr.py, src/km_inference.py |
| Figures 1-6 | Complete | All validation and comparison plots |
| Manuscript | Draft complete | manuscript/main.tex (20 pages) |
| Talk | Complete | Talks/seminar_talk.tex (34 slides) |
| Force-free method | Working | 11% error vs Sekimoto reference |

## Key Results

- **NF-EPR recovers ~95% hidden entropy** that Schnakenberg misses
- **Force-free method works** with 11.3% error (no knowledge of potential needed)
- **Computational cost**: ~40s/model NF training, ~27 min for 40 snapshots (single CPU)
- **GMM base essential**: Unimodal base cannot learn bimodal densities (monotone composition theorem)

## Project Structure

```
NF_Langevin/
├── src/                    # Python sources
│   ├── physics.py          # Langevin simulation, Fokker-Planck current, parameters
│   ├── flows_1d.py         # 1D planar normalizing flows with GMM base
│   ├── epr.py              # EPR methods: NF, continuity, Sekimoto, Schnakenberg
│   ├── km_inference.py     # Kramers-Moyal coefficient estimation
│   ├── pub_style.py        # Matplotlib publication style
│   └── fig[1-6]_*.py       # Figure generation scripts
│
├── figures/                # Generated figures (PDF + PNG)
│   ├── fig1_equilibrium_validation.*
│   ├── fig2_harmonic_validation.*
│   ├── fig3_learned_density_snapshots.*
│   ├── fig4_epr_nf_vs_sekimoto.*
│   ├── fig5_architecture_comparison.*
│   └── fig6_km_epr_comparison.*
│
├── manuscript/             # Manuscript (LaTeX)
│   ├── main.tex            # Main document
│   ├── introduction.tex
│   ├── methods_results.tex
│   ├── discussion.tex
│   ├── references.bib
│   └── figures/            # Symlinks to ../figures/
│
└── Talks/
    ├── seminar_talk.tex    # 34-slide presentation
    ├── seminar_talk.pdf
    └── figures/            # Symlinks to ../figures/
```

## Dependencies

```bash
# Pure NumPy/SciPy implementation (no PyTorch/JAX required)
pip install numpy scipy matplotlib
```

Python 3.9+ recommended. Uses `numpy.trapezoid` (not deprecated `trapz`).

## Quick Start

```bash
# Clone (GitHub mirror)
git clone git@github.com:antonife/NF_Langevin.git
cd NF_Langevin

# Or clone from GitLab (entropy-cromatines, branch normalizing-flows)
git clone -b normalizing-flows git@gitlab.eif.urjc.es:afcaball/entropy-cromatines.git

# Generate all figures (takes ~30 min total)
cd src
python fig1_equilibrium_validation.py
python fig2_harmonic_validation.py
python fig3_learned_density_snapshots.py
python fig4_epr_nf_vs_sekimoto.py
python fig5_architecture_comparison.py
python fig6_km_epr_comparison.py
cd ..

# Compile manuscript
cd manuscript && latexmk -pdf main.tex && cd ..

# Compile talk
cd Talks && latexmk -pdf seminar_talk.tex && cd ..
```

## Core API

### Simulation
```python
from physics import LangevinParams, simulate_ensemble

params = LangevinParams()  # Default: D=0.5, gamma=1, Period=10s
time_grid, ensemble_x = simulate_ensemble(n_particles=5000, n_cycles=3, params=params, rng=rng)
```

### Density Learning
```python
from epr import extract_time_bins, train_flows_per_bin

bins = extract_time_bins(time_cycle, ensemble_cycle, n_bins=40)
flows = train_flows_per_bin(bins, n_layers=8, n_base_components=2)
# flows is List[(t_center, PlanarFlow1D)]
```

### EPR Computation
```python
from epr import compute_epr_from_flows, compute_epr_continuity, sekimoto_epr

# Method 1: Known force (requires V'(x,h))
t_arr, EPR_arr, sigma_field = compute_epr_from_flows(flows, params, x_grid)

# Method 2: Force-free (continuity equation)
t_arr, EPR_arr, sigma_field = compute_epr_continuity(flows, D=0.5, x_grid)

# Reference: Sekimoto (energy balance)
EPR_sek, Sigma_sek = sekimoto_epr(time_cycle, ensemble_cycle, params)
```

## Physical Model

Overdamped Langevin dynamics in periodically tilted Landau potential:

```
dx/dt = -V'(x,h)/γ + √(2D) ξ(t)

V(x,h) = x⁴ - 2x² + h·x       (bistable Landau)
h(t) = h₀·sin(2πt/T)          (periodic tilt)
```

Default parameters: D=0.5, γ=1.0, h₀=2.0, T=10s

## Methods Summary

| Method | Requires | Accuracy | Use Case |
|--------|----------|----------|----------|
| Sekimoto | Trajectories + V(x,h) | Reference | Ground truth |
| NF-EPR | Learned p(x,t) + V'(x,h) | ~4% error | Known potential |
| Continuity | Learned p(x,t) + D only | ~11% error | Unknown potential |
| Schnakenberg | Transition counts | Captures 5% | Coarse-grained only |

## Computational Cost (single CPU)

| Stage | Time |
|-------|------|
| Langevin simulation (5000 particles, 3 cycles) | ~7s |
| NF training (40 models × 40s/model) | ~27 min |
| EPR computation (both methods) | <0.5s |
| Force-free overhead (KM + continuity) | 0.4s |

## Next Steps / TODOs

- [ ] Apply force-free method to experimental data (colloidal particles, molecular motors)
- [ ] Extend to 2D using coupling layers (Real-NVP)
- [ ] GPU acceleration via PyTorch/JAX for higher dimensions
- [ ] Conditional flows p(x|t) to exploit temporal smoothness
- [ ] Submit manuscript to Physical Review E or Journal of Chemical Physics

## Known Issues

- **Fig 3 — `h=0` panel**: NF-learned density deviates notably from the Boltzmann reference at h=0, while the h=±1, ±2 panels agree visually. Likely physical (nonequilibrium vs. quasi-static reference) but may also reflect an NF optimization issue on the symmetric bimodal target. Needs investigation before submission. See GitLab issue [#1](https://gitlab.eif.urjc.es/afcaball/entropy-cromatines/-/issues/1).

## References

- Kobyzev et al. (2020) "Normalizing Flows: An Introduction and Review" — IEEE TPAMI (suggested by collaborator)
- Sekimoto (2010) "Stochastic Energetics" — Springer
- Schnakenberg (1976) "Network theory of microscopic and macroscopic behavior" — Rev. Mod. Phys.

## Related Work

Companion paper on topological hysteresis bounds in the same Landau system (separate repo).

## Author

Antonio Fernández-Caballero  
Departamento de Teoría de la Señal y Comunicaciones  
Universidad Rey Juan Carlos, Fuenlabrada, Spain
