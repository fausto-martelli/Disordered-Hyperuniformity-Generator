# Disordered-Hyperuniformity-Generator

A fixed Poisson configuration of $N$ particles in a periodic square box is
displaced by a bounded vector field produced by a small multilayer perceptron.
The field is optimized — not the particles — against a loss combining three
terms: suppression of the structure factor at low wavenumbers, short-range
repulsion, and a penalty on the variance of $S(k)$ across reciprocal space. The
result is a configuration that is hyperuniform over the sampled reciprocal-space
window while remaining disordered.

The essential point about the parametrization is that the particle coordinates
are a deterministic function of the network weights:

$$X(t) = \left( X_0 + s\,L \tanh\left[ g_\theta(z) \right] \right) \bmod L$$

so the optimization is a reparametrized search over displacement fields. The
gradient updates $\theta$; the particles follow. The base configuration $X_0$ is
drawn once and registered as a buffer — it is never trained.

The trajectory $X(t)$ is a path in configuration space indexed by the epoch
counter. It is **not** a dynamical trajectory: there is no equation of motion,
no thermostat and no time step, so no kinetic meaning attaches to the rate at
which quantities evolve with $t$.

---

## Requirements

```
python >= 3.9
torch
numpy
matplotlib
```

## Usage

```bash
python dhu_generator_training.py
```

The device is selected automatically — Apple MPS if available, otherwise CPU.
For CUDA, edit the `device` line near the top of the file.

Runtime is dominated by the two $N \times N$ pair tensors built each epoch (see
[Memory](#memory)), not by the network, which is tiny.

At start-up the script reports the number of constrained modes and the
constrained-mode fraction $\chi = m/(dN)$. Every 50 epochs it prints the total
loss with its three components, which is the quickest way to see which term
dominates at a given stage:

```
Constrained modes: 44 raw (+/-k pairs), 22 independent
chi = m/(d*N) = 0.001100
Epoch 0: Loss=1.2345e+00  (hyper=..., rep=..., smooth=...)
```

---

## Output

| Path | Contents |
|---|---|
| `points_poisson_reference.csv` | The untouched initial configuration $X_0$. Two columns `x,y`, one header line. |
| `snapshots/points_epoch_XXXXX.csv` | Particle coordinates every `snapshot_every` epochs, same format, wrapped into $[0, L)$. |
| `snapshots/points_epoch_XXXXX.png` | Scatter plot of that snapshot. |
| `snapshots/Sk_epoch_XXXXX.png` | Radially averaged $S(k)$ for that snapshot. |

The loss history is accumulated in memory and **never written to disk**.

---

## Parameters

All parameters sit in a single block near the top of the file. Defaults are the
values used in the paper.

| Name | Default | Meaning |
|---|---|---|
| `N_points` | `10000` | Number of particles, $N$. |
| `box_size` | `1000.0` | Box side $L$. Mean spacing $\ell = L/\sqrt{N} = 10$. |
| `dim` | `2` | Spatial dimension. The analysis scripts assume 2. |
| `latent_dim` | `16` | Dimension of the latent vector $z$. |
| `n_k` | `120` | Reciprocal grid is `n_k` × `n_k`, a square of half-width $60 \cdot 2\pi/L \approx 0.377$. |
| `S_target` | `1e-4` | Target value of $S(k)$ below the cutoff. Not zero — the stealthy condition is enforced softly. |
| `cutoff` | `25.0` | Real-space cutoff $r_c$ for the repulsion term, $2.5\,\ell$. |
| `scale_factor` | `0.05` | Displacement bound $s$. Caps each Cartesian component at $sL = 5\ell$. |
| `noise` | `0.005` | Gaussian perturbation added to coordinates each epoch, $5 \times 10^{-4}\,\ell$. |
| `epochs` | `15000` | Number of optimization steps. |
| `lr` | `1e-4` | Adam learning rate. |
| `batch_size` | `1` | Latent samples per epoch. |
| `snapshot_every` | `100` | Snapshot cadence. |
| `lambda_hyper`, `lambda_rep`, `lambda_smooth` | `1.0` | Loss weights. |

---

## The loss

Each epoch draws a fresh latent vector, builds the coordinates, and evaluates:

| Term | Definition | Acts on |
|---|---|---|
| `L_hyper` | $\langle (S(k) - S_\mathrm{target})^2 \rangle$, $0 < \|k\| < k_c$ | the $m = 22$ constrained modes |
| `L_rep` | $\langle 1/r_{ij}^2 \rangle$, $r_{ij} < r_c$ | real space, minimum image |
| `L_smooth` | $\langle (S(k) - \bar{S})^2 \rangle_k$ | the whole reciprocal grid |

`L_rep` is normalized by the number of qualifying pairs, not by $N^2$, so it is
a genuine conditional average. `L_smooth` penalizes sharp features in $S(k)$ and
therefore acts as a regularizer against crystallization.

The three are summed with the $\lambda$ weights and minimized with Adam, with
gradient-norm clipping at `1.0`.

---

## Implementation notes

### Repulsion uses `torch.where`, not boolean indexing

> [!WARNING]
> Selecting the qualifying pairs with `dist2[neighbor_mask]` is a
> variable-length `masked_select` whose backward pass is unreliable on the MPS
> backend once the neighbour count changes between epochs. It fails inside
> `loss.backward()` with a shape-mismatch error. The `torch.where` form keeps
> the tensor shape fixed and behaves identically on MPS, CPU and CUDA. Please
> keep it.

### Memory

The pair calculation builds an $N \times N \times 2$ displacement tensor and an
$N \times N$ distance tensor each epoch — about 1.2 GB at $N = 10^4$ in single
precision, with several such tensors alive during the backward pass. The phase
matrix for $S(k)$ adds roughly another 1.2 GB. Both scale quadratically, so
$N = 2\times10^4$ is already demanding on a laptop.

---
