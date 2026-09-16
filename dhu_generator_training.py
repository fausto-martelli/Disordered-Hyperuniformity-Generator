import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Device setup
# ============================================================
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ============================================================
# Parameters
# ============================================================
N_points = 10000
#N_points = 17000
dim = 2
box_size = 1000.0
#box_size = 1300.0
latent_dim = 16
n_k = 120
epochs = 15000
lr = 1e-4
batch_size = 1
S_target = 0.0001
noise = 0.005
cutoff = 25.0
snapshot_every = 100

# Max per-particle displacement as a fraction of box_size (was hardcoded
# inside PeriodicGenerator.forward(); see note at top of file).
scale_factor = 0.05

# Relative weights of the three loss contributions (Sec. III of the paper).
# Defaults reproduce the paper's unweighted sum L = L_hyper + L_rep + L_smooth.
lambda_hyper = 1.0
lambda_rep = 1.0
lambda_smooth = 1.0

# Create snapshots folder
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# ============================================================
# Minimum image displacement
# ============================================================
def mic_displacement(d, box_size):
    return d - torch.round(d / box_size) * box_size

# ============================================================
# Generator network
# ============================================================
class PeriodicGenerator(nn.Module):
    def __init__(self, latent_dim, N_points, dim, box_size=1.0, scale_factor=0.05):
        super().__init__()
        self.N_points = N_points
        self.dim = dim
        self.box_size = box_size
        self.scale_factor = scale_factor

        # Fixed base positions
        base = torch.rand(1, N_points, dim) * box_size
        self.register_buffer("base", base)

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, N_points * dim)
        )

    def forward(self, z):
        batch = z.size(0)
        base = self.base.expand(batch, -1, -1)
        delta_raw = self.net(z).view(batch, self.N_points, self.dim)
        delta = self.scale_factor * self.box_size * torch.tanh(delta_raw)
        coords = (base + delta) % self.box_size
        return coords

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.05)
        nn.init.constant_(m.bias, 0.0)

G = PeriodicGenerator(latent_dim, N_points, dim, box_size=box_size, scale_factor=scale_factor).to(device)
G.apply(init_weights)
optimizer = optim.Adam(G.parameters(), lr=lr)

# ============================================================
# Save unperturbed Poisson initialization (reference only)
# ============================================================
with torch.no_grad():
    coords_poisson = G.base[0].cpu().numpy()
    np.savetxt(
        "points_poisson_reference.csv",
        coords_poisson,
        delimiter=",",
        header="x,y",
        comments=""
    )


# ============================================================
# k-vectors for S(k)
# ============================================================
freqs = [torch.fft.fftfreq(n_k, d=1.0/n_k) * 2*np.pi/box_size for _ in range(dim)]
mesh = torch.meshgrid(*freqs, indexing="ij")
kvecs = torch.stack([m.flatten() for m in mesh], dim=1).to(device)
mask_nonzero = ~(torch.all(kvecs == 0, dim=1))
kvecs = kvecs[mask_nonzero]
k_mag = torch.sqrt((kvecs**2).sum(dim=1))
mask_smallk = k_mag < (2*np.pi*4/box_size)

# Report the constrained-mode fraction chi = m/(d*N) referenced in the paper
# (Introduction / Conclusions). mask_smallk counts +k and -k separately,
# but S(k)=S(-k) makes them redundant, so the number of independent
# constrained modes is half the raw count.
m_raw = int(mask_smallk.sum().item())
m_independent = m_raw // 2
chi = m_independent / (dim * N_points)
print(f"Constrained modes: {m_raw} raw (+/-k pairs), {m_independent} independent")
print(f"chi = m/(d*N) = {chi:.6f}")

# ============================================================
# Structure factor (NumPy)
# ============================================================
def structure_factor_np(positions, n_k=40, box_size=1.0):
    dim = positions.shape[1]
    freqs = [np.fft.fftfreq(n_k, d=1.0/n_k) * 2*np.pi/box_size for _ in range(dim)]
    mesh = np.meshgrid(*freqs, indexing="ij")
    kvecs = np.stack([m.flatten() for m in mesh], axis=1)
    mask_nonzero = ~(np.all(kvecs == 0, axis=1))
    kvecs = kvecs[mask_nonzero]
    k_mag = np.sqrt((kvecs**2).sum(axis=1))
    phase = np.exp(-1j * positions @ kvecs.T)
    rho_k = phase.sum(axis=0)
    S_k = (np.abs(rho_k)**2 / len(positions)).real
    return k_mag, S_k

def radial_average(k_mag, S_k, bins=60):
    k_bins = np.linspace(0, k_mag.max(), bins)
    S_rad = np.zeros(len(k_bins)-1)
    for i in range(len(k_bins)-1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.any(mask):
            S_rad[i] = S_k[mask].mean()
    k_centers = 0.5*(k_bins[1:] + k_bins[:-1])
    return k_centers, S_rad

# ============================================================
# Training loop (vectorized)
# ============================================================
loss_history = []

for epoch in range(epochs):
    z = torch.randn(batch_size, latent_dim, device=device)
    coords = G(z)[0] + noise * torch.randn(N_points, dim, device=device)
    coords = coords % box_size

    # Structure factor
    phase = torch.exp(-1j * (coords @ kvecs.T))
    rho_k = phase.sum(dim=0)
    S_k = (rho_k.abs()**2 / N_points).real
    L_hyper = ((S_k[mask_smallk] - S_target)**2).mean()

    # ==============================
    # Local repulsion: <1/r_ij^2>_{r_ij<r_c}
    # Normalized over qualifying pairs only (matching the paper's
    # conditional average), not over the full N x N pair matrix.
    # ==============================
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    diff = mic_displacement(diff, box_size)
    dist2 = (diff**2).sum(dim=-1) + 1e-6
    neighbor_mask = dist2 < cutoff**2
    neighbor_mask.fill_diagonal_(False)
    # Use torch.where (fixed N x N shape) rather than boolean fancy-indexing
    # (dist2[neighbor_mask], a variable-length masked_select) -- the latter
    # triggers a known MPS-backend autograd bug where the backward pass loses
    # sync with the forward pass's mask size once the neighbor count changes
    # between epochs (RuntimeError: shape mismatch ... cannot be broadcast).
    # torch.where keeps the tensor shape fixed and is safe on MPS/CPU/CUDA
    # alike; only the normalization (sum / actual neighbor count, instead of
    # sum / N^2) differs from the original script.
    inv_d2 = torch.where(neighbor_mask, 1.0 / dist2, torch.zeros_like(dist2))
    n_neighbors = neighbor_mask.sum()
    if n_neighbors > 0:
        L_repulsion = inv_d2.sum() / n_neighbors
    else:
        L_repulsion = torch.zeros((), device=device, dtype=dist2.dtype)

    # Smoothness term
    L_smooth = ((S_k - S_k.mean())**2).mean()

    # Total loss -- now combines all three contributions, matching Sec. III
    # of the paper ("L = L_hyper + L_rep + L_smooth").
    loss = lambda_hyper * L_hyper + lambda_rep * L_repulsion + lambda_smooth * L_smooth

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
    optimizer.step()
    loss_history.append(loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4e}  "
              f"(hyper={L_hyper.item():.4e}, rep={L_repulsion.item():.4e}, smooth={L_smooth.item():.4e})")

    # -----------------------------
    # Snapshots every 200 epochs
    # -----------------------------
    if epoch % snapshot_every == 0:
        coords_sample = coords.detach().cpu().numpy()
        np.savetxt(f"snapshots/points_epoch_{epoch:05d}.csv", coords_sample, delimiter=",", header="x,y", comments="")
        plt.figure(figsize=(5,5))
        plt.scatter(coords_sample[:,0], coords_sample[:,1], s=1)
        plt.title(f"Points (epoch {epoch})")
        plt.xlim(0, box_size)
        plt.ylim(0, box_size)
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f"snapshots/points_epoch_{epoch:05d}.png", dpi=160)
        plt.close()

        k_mag_vals, S_k_vals = structure_factor_np(coords_sample, n_k=n_k, box_size=box_size)
        k_centers, S_rad = radial_average(k_mag_vals, S_k_vals)
        plt.figure(figsize=(5,4))
        plt.plot(k_centers, S_rad, "o-", label="Generated")
        plt.xlabel("k")
        plt.ylabel("S(k)")
        plt.title(f"S(k) (epoch {epoch})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"snapshots/Sk_epoch_{epoch:05d}.png", dpi=160)
        plt.close()
