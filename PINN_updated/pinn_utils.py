"""
Utility functions for the PINN implementation

Image and Sound Processing Lab - Politecnico di Milano

Daniele Ugo Leonzio
Paolo Bestagini

IMPROVEMENTS:
- Laplacian raised to 4th order (less numerical dispersion)
- CFL check (raises ValueError if problem occurs)
- Added frequency bandpass utility for multi-scale FWI (to test if better result)
"""

import os
from typing import Optional, Tuple

import GPUtil
import numpy as np
import torch
from torch import nn


# Device Utils

def set_gpu(id: int = -1) -> None:
    """
    Select the compute device.

    Args:
        id: -1 → GPU with lowest memory usage (auto),
             None → CPU only,
             int ≥ 0 → specific GPU index.
    """
    if id is None:
        print("GPU not selected — using CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        device = id if id != -1 else GPUtil.getFirstAvailable(order="memory")[0]
        try:
            name = GPUtil.getGPUs()[device].name
        except IndexError:
            print("Selected GPU does not exist. Switching to most available one.")
            device = GPUtil.getFirstAvailable(order="memory")[0]
            name = GPUtil.getGPUs()[device].name
        print(f"GPU selected: {device} — {name}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


# CFL check

def check_cfl(dt_s: float, dh_km: float, vp_max: float, strict: bool = True) -> None:
    """
    Check the Courant–Friedrichs–Lewy (CFL) stability condition.

    The condition for the 2D acoustic wave equation is:
        dt ≤ dh / (sqrt(2) * vp_max)

    Args:
        dt_s:    time step in seconds.
        dh_km:   grid spacing in km.
        vp_max:  maximum velocity in the model (km/s).
        strict:  if True raise ValueError on violation, else just warn.

    Raises:
        ValueError: if the CFL condition is violated and strict=True.
    """
    dt_max = dh_km / (np.sqrt(2) * vp_max)
    if dt_s > dt_max:
        msg = (
            f"CFL condition violated! "
            f"dt={dt_s*1000:.4f} ms > dt_max={dt_max*1000:.4f} ms  "
            f"(dh={dh_km*1000:.1f} m, vp_max={vp_max*1000:.0f} m/s)"
        )
        if strict:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg}")


# PML coefficients

def generate_pml_coefficients_2d(
    domain_shape: Tuple[int, int],
    N: int = 50,
    multiple: bool = False,
) -> torch.Tensor:
    """
    Generate 2-D Perfectly Matched Layer (PML) damping coefficients.

    Args:
        domain_shape: (Nz, Nx) — full domain size including PML padding.
        N:            number of PML cells on each side.
        multiple:     if True, do NOT apply PML on the top boundary
                      (useful for free-surface condition).

    Returns:
        Tensor of shape (Nz, Nx) with damping values ≥ 0.
    """
    Nx, Ny = domain_shape

    R = 1e-6
    order = 2
    cp = 1000.0
    d0 = (1.5 * cp / N) * np.log10(R ** -1)
    d_vals = d0 * torch.linspace(0.0, 1.0, N + 1) ** order
    d_vals = torch.flip(d_vals, [0])

    d_x = torch.zeros(Ny, Nx)
    d_y = torch.zeros(Ny, Nx)

    if N > 0:
        d_x[0 : N + 1, :] = d_vals.repeat(Nx, 1).transpose(0, 1)
        d_x[(Ny - N - 1) : Ny, :] = torch.flip(d_vals, [0]).repeat(Nx, 1).transpose(0, 1)
        if not multiple:
            d_y[:, 0 : N + 1] = d_vals.repeat(Ny, 1)
        d_y[:, (Nx - N - 1) : Nx] = torch.flip(d_vals, [0]).repeat(Ny, 1)

    _d = torch.sqrt(d_x ** 2 + d_y ** 2).transpose(0, 1)
    _d = _corners(domain_shape, N, _d, d_x.T, d_y.T, multiple)
    return _d


def _corners(
    domain_shape: Tuple[int, int],
    abs_N: int,
    d: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    multiple: bool = False,
) -> torch.Tensor:
    Nx, Ny = domain_shape
    for j in range(Ny):
        for i in range(Nx):
            if not multiple:
                if i < abs_N + 1 and j < abs_N + 1:
                    d[i, j] = dy[i, j] if i < j else dx[i, j]
            if i > (Nx - abs_N - 2) and j < abs_N + 1:
                d[i, j] = dx[i, j] if i + j < Nx else dy[i, j]
            if i > (Nx - abs_N - 2) and j > (Ny - abs_N - 2):
                d[i, j] = dy[i, j] if i - j > Nx - Ny else dx[i, j]
            if not multiple:
                if i < abs_N + 1 and j > (Ny - abs_N - 2):
                    d[i, j] = dy[i, j] if i + j < Ny else dx[i, j]
    return d


def absorbing_boundaries(nx: int, ny: int, nb: int, u: float) -> np.ndarray:
    """
    Simple exponential absorbing boundary mask (alternative to PML).

    Args:
        nx, ny: domain size in x and z.
        nb:     number of absorbing boundary cells.
        u:      exponential decay factor.

    Returns:
        2-D numpy array of shape (ny, nx) with values in (0, 1].
    """
    bound_coeffs = np.exp(-u * (nb - np.arange(nb)) ** 2)
    central_x = np.ones(nx - 2 * nb)
    mask_x = np.tile(np.concatenate([bound_coeffs, central_x, bound_coeffs[::-1]]), (ny, 1))
    central_y = np.ones(ny - 2 * nb)
    mask_y = np.tile(
        np.concatenate([bound_coeffs, central_y, bound_coeffs[::-1]])[:, np.newaxis], (1, nx)
    )
    return mask_x * mask_y


# Wave equation numerics

def laplace(u: torch.Tensor, h: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """
    Compute the 2-D Laplacian of field *u* using a 4th-order finite-difference
    stencil (reduced numerical dispersion compared to 2nd order).

    The stencil coefficients for the 4th-order scheme are (5 stencils):
        [-1/12,  4/3,  -5/2,  4/3,  -1/12]

    Args:
        u:   field tensor of shape (batch, 1, Nz, Nx).
        h:   grid spacing (scalar tensor).
        dev: compute device.

    Returns:
        Laplacian of *u*, same shape as *u*.
    """
    # IMPROVEMENT: upgraded from 2nd to 4th order — LESS dispersion, better accuracy
    kernel_order = 4
    kernel_size = 5
    coeffs = torch.tensor([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12], dtype=torch.float32)

    kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
    kernel[kernel_size // 2, :] += coeffs
    kernel[:, kernel_size // 2] += coeffs
    kernel = kernel.view(1, 1, kernel_size, kernel_size).to(dev)

    return torch.nn.functional.conv2d(u, kernel, padding=kernel_size // 2) / (h ** 2)


def step(
    u_pre: torch.Tensor,
    u_now: torch.Tensor,
    dev: torch.device,
    c: torch.Tensor,
    dt: torch.Tensor,
    h: torch.Tensor,
    b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    
    """
    Advance the acoustic wavefield by one time step using the
    2nd-order-in-time leapfrog scheme with PML damping.

    Discretisation of:
        ∂²u/∂t² + b·∂u/∂t = c²·∇²u

    Args:
        u_pre: wavefield at t-dt,  shape (batch, 1, Nz, Nx).
        u_now: wavefield at t,     shape (batch, 1, Nz, Nx).
        dev:   compute device.
        c:     velocity model,     shape (1, Nz, Nx) or broadcastable.
        dt:    time step (scalar tensor, seconds).
        h:     grid spacing (scalar tensor, km).
        b:     PML damping coefficients, shape broadcastable to u_now.

    Returns:
        u_next: wavefield at t+dt, same shape as u_now.
    """
    lap = laplace(u_now, h, dev)
    u_next = torch.mul(
        (dt ** -2 + b * dt ** -1).pow(-1),
        (
            2.0 / dt ** 2 * u_now
            - torch.mul((dt ** -2 - b * dt ** -1), u_pre)
            + torch.mul(c.pow(2), lap)
        ),
    )

    if torch.isnan(u_next).any():
        raise RuntimeError("NaN values detected in the wavefield — check CFL condition and PML parameters.")

    return u_next


def forward(
    wave: torch.Tensor,
    c: torch.Tensor,
    b: torch.Tensor,
    src_list: np.ndarray,
    domain: Tuple[int, int],
    dt: float,
    h: float,
    dev: torch.device,
    recz: int,
    pmln: int,
) -> torch.Tensor:
    """
    Compute the acoustic forward modelling for a batch of simultaneous shots.

    All shots are propagated in parallel along the batch dimension — this is
    the key GPU-efficiency trick in this codebase.

    Args:
        wave:     source wavelet, shape (nt,).
        c:        velocity model (padded), shape (Nz_pad, Nx_pad).
        b:        PML coefficients,        shape (Nz_pad, Nx_pad).
        src_list: array of (x, z) source grid indices, shape (nshots, 2).
        domain:   (Nz_pad, Nx_pad).
        dt:       time step in seconds.
        h:        grid spacing in km.
        dev:      compute device.
        recz:     receiver depth in grid samples (including PML offset).
        pmln:     number of PML cells (used to strip PML from receiver data).

    Returns:
        rec: recorded data, shape (nshots, nt, Nx_physical).
    """
    nt = wave.shape[0]
    nz, nx = domain
    nshots = len(src_list)

    # Initialise wavefields
    u_pre = torch.zeros(nshots, 1, *domain, device=dev)
    u_now = torch.zeros(nshots, 1, *domain, device=dev)
    rec = torch.zeros(nshots, nt, nx - 2 * pmln, device=dev)

    # Unsqueeze: (1, 1, Nz, Nx)
    b_bc = b.unsqueeze(0).to(dev)
    c_bc = c.unsqueeze(0)

    shots = torch.arange(nshots, device=dev)
    srcx, srcz = zip(*src_list)

    h_t = torch.tensor([h], device=dev)
    dt_t = torch.tensor([dt], device=dev)

    # One source mask per shot — injected simultaneously
    source_mask = torch.zeros_like(u_now)
    source_mask[shots, :, srcz, srcx] = 1.0

    for it in range(nt):
        u_now = u_now + source_mask * wave[it]
        u_next = step(u_pre, u_now, dev, c_bc, dt_t, h_t, b_bc)
        u_pre, u_now = u_now, u_next
        rec[:, it, :] = u_now[:, 0, recz, pmln:-pmln]

    return rec


# ---------------------------------------------------------------------------
# Multi-scale FWI utility - TO TEST
# ---------------------------------------------------------------------------

def bandpass_shots(
    shots: np.ndarray,
    f_low: float,
    f_high: float,
    dt_s: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to a shot gather array.

    Used for multi-scale (frequency continuation) FWI: start with low
    frequencies only, progressively increase the upper cutoff.

    Args:
        shots:  array of shape (nshots, nt, nrec).
        f_low:  lower cutoff frequency in Hz (use a small value, e.g. 1.0,
                rather than 0 to avoid numerical issues).
        f_high: upper cutoff frequency in Hz.
        dt_s:   time step in seconds.
        order:  Butterworth filter order.

    Returns:
        Filtered shots array, same shape as input.
    """
    from scipy.signal import butter, filtfilt

    fs = 1.0 / dt_s
    nyq = fs / 2.0
    f_low_n = max(f_low / nyq, 1e-4)
    f_high_n = min(f_high / nyq, 0.999)

    b, a = butter(order, [f_low_n, f_high_n], btype="band")
    filtered = filtfilt(b, a, shots, axis=1)
    return filtered.astype(np.float32)


# SIREN neural network definition

class SineLayer(nn.Module):
    """
    A single linear layer followed by a sine activation.

    From Sitzmann et al. (2020) "Implicit Neural Representations with
    Periodic Activation Functions".

    Args:
        in_features:  input dimension.
        out_features: output dimension.
        bias:         whether to use bias.
        is_first:     True for the first layer (different weight init scale).
        omega_0:      frequency multiplier.  First layer: controls bandwidth.
                      Hidden layers: controls gradient magnitude.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1.0 / self.in_features, 1.0 / self.in_features
                )
            else:
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    """
    Sinusoidal Representation Network (SIREN) used as an implicit neural
    representation of the 2-D velocity model.

    The network maps normalised spatial coordinates (x, z) ∈ [-1,1]² to a
    scalar velocity value vp(x,z).  Its smooth inductive bias acts as an
    implicit regulariser for the FWI, preventing high-frequency artefacts.

    Args:
        in_features:      input dimension (2 for 2-D).
        hidden_features:  neurons per hidden layer.
        hidden_layers:    number of hidden SineLayers.
        out_features:     output dimension (1 for scalar vp).
        outermost_linear: if True the last layer is a plain Linear (no sine).
        pretrained:       path to a saved state_dict (.pth or .npz).
        first_omega_0:    omega_0 for the first layer.
        hidden_omega_0:   omega_0 for hidden layers.
        domain_shape:     (Nz, Nx) of the physical (non-padded) domain.
        dh:               grid spacing (unused currently, reserved).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = False,
        pretrained: Optional[str] = None,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        domain_shape: Optional[Tuple[int, ...]] = None,
        dh: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.domain_shape = domain_shape

        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(
                SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0)
            )

        if outermost_linear:
            final = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6.0 / hidden_features) / hidden_omega_0
                final.weight.uniform_(-bound, bound)
            layers.append(final)
        else:
            layers.append(
                SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0)
            )

        self.net = nn.Sequential(*layers)
        self.coords = self._generate_mesh(domain_shape)
        self._load_pretrained(pretrained)

    def _load_pretrained(self, pretrained: Optional[str]) -> None:
        path = pretrained or ""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))
            print(f"Loaded pretrained weights from: {path}")
        else:
            print(f"Pretrained model not found at '{path}'. Using random initialisation.")

    def _generate_mesh(self, mshape: Tuple[int, ...]) -> torch.Tensor:
        """Generate a flattened [-1, 1]^d coordinate grid."""
        axes = [torch.linspace(-1, 1, steps=s) for s in mshape]
        mgrid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return mgrid.reshape(-1, len(mshape))

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords).view(self.domain_shape)
        return output, coords


# Elastic wave equation utilities (uguale a prima)

def gradient(input: torch.Tensor, dim: int = -1, forward: bool = True, padding_value: float = 0.0) -> torch.Tensor:
    def forward_diff(x, dim, padding_value):
        diff = x - torch.roll(x, shifts=1, dims=dim)
        if dim == 1:
            diff[:, 0] = padding_value
        elif dim == 2:
            diff[..., 0] = padding_value
        return diff

    def backward_diff(x, dim, padding_value):
        diff = torch.roll(x, shifts=-1, dims=dim) - x
        if dim == 1:
            diff[:, -1] = padding_value
        elif dim == 2:
            diff[..., -1] = padding_value
        return diff

    return forward_diff(input, dim, padding_value) if forward else backward_diff(input, dim, padding_value)


def step_elastic(parameters, wavefields, geometry):
    vp, vs, rho = parameters
    vx, vz, txx, tzz, txz = wavefields
    dt, h, d = geometry

    lame_lambda = rho * (vp.pow(2) - 2 * vs.pow(2))
    lame_mu = rho * vs.pow(2)
    c = 0.5 * dt * d

    vx_x = gradient(vx, 2)
    vz_z = gradient(vz, 1, False)
    vx_z = gradient(vx, 1)
    vz_x = gradient(vz, 2, False)

    y_txx = (1 + c) ** -1 * (dt * h.pow(-1) * ((lame_lambda + 2 * lame_mu) * vx_x + lame_lambda * vz_z) + (1 - c) * txx)
    y_tzz = (1 + c) ** -1 * (dt * h.pow(-1) * ((lame_lambda + 2 * lame_mu) * vz_z + lame_lambda * vx_x) + (1 - c) * tzz)
    y_txz = (1 + c) ** -1 * (dt * lame_mu * h.pow(-1) * (vz_x + vx_z) + (1 - c) * txz)

    txx_x = gradient(y_txx, 2, False)
    txz_z = gradient(y_txz, 1, False)
    tzz_z = gradient(y_tzz, 1)
    txz_x = gradient(y_txz, 2)

    y_vx = (1 + c) ** -1 * (dt * rho.pow(-1) * h.pow(-1) * (txx_x + txz_z) + (1 - c) * vx)
    y_vz = (1 + c) ** -1 * (dt * rho.pow(-1) * h.pow(-1) * (txz_x + tzz_z) + (1 - c) * vz)

    return y_vx, y_vz, y_txx, y_tzz, y_txz


def forward_elastic(wave, parameters, pmlc, src_list, domain, dt, h, dev, npml=50, recz=0):
    nt = wave.shape[0]
    nz, nx = domain
    nshots = len(src_list)

    dt = torch.tensor(dt, dtype=torch.float32, device=dev)
    h = torch.tensor(h, dtype=torch.float32, device=dev)

    vx  = torch.zeros(nshots, *domain, device=dev)
    vz  = torch.zeros(nshots, *domain, device=dev)
    txx = torch.zeros(nshots, *domain, device=dev)
    tzz = torch.zeros(nshots, *domain, device=dev)
    txz = torch.zeros(nshots, *domain, device=dev)

    wavefields = [vx, vz, txx, tzz, txz]
    geoms = [dt, h, pmlc]
    rec = torch.zeros(nshots, nt, nx - 2 * npml, device=dev)

    shots = torch.arange(nshots, device=dev)
    srcx, srcz = zip(*src_list)
    src_mask = torch.zeros_like(vx)
    src_mask[shots, srcz, srcx] = 1.0

    for it in range(nt):
        wavefields[1] = wavefields[1] + src_mask * wave[it]
        wavefields = list(step_elastic(parameters, wavefields, geoms))
        rec[:, it, :] = wavefields[1][:, recz, npml:-npml]

    return rec
