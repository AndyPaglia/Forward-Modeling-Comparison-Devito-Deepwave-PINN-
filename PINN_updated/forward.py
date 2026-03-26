"""
Compute the forward modelling in a 2-D domain using PyTorch.

Image and Sound Processing Lab - Politecnico di Milano

Daniele Ugo Leonzio
Paolo Bestagini

IMPROVEMENTS:
- CFL check is defined as an hard ValueError
- Cleaner argument parsing and docstrings
- Removed dead / redundant code paths
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_utils import (
    check_cfl,
    forward,
    generate_pml_coefficients_2d,
    set_gpu,
)

os.nice(10)
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

# Main

def main(args: argparse.Namespace) -> None:
    """
    Run 2-D acoustic forward modelling and save the shot gathers.

    Steps
    -----
    1. Load and pad the velocity model.
    2. Check numerical stability (CFL, max frequency).
    3. Build source / receiver geometry.
    4. Define the Ricker source wavelet.
    5. Build PML damping coefficients.
    6. Run batched forward modelling on GPU.
    7. Save results to disk.
    """

    # Device
    set_gpu(-1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load velocity model
    npzfile = np.load(args.vp_model_path)
    vp = npzfile["vp"].T
    spacing = npzfile["spacing"] / 1000.0   # → km
    dh = float(spacing[0])
    domain = np.array(vp.shape)             # (Nz, Nx) physical

    nbl = args.nbl
    vp_pad = np.pad(vp, ((nbl, nbl), (nbl, nbl)), mode="edge")
    domain_pad = np.array(vp_pad.shape)

    # Numerical stability checks
    dt_s = args.dt / 1000.0   # ms → s
    tn_s = args.tn / 1000.0   # ms → s

    # IMPROVEMENT: CFL is now a hard stop — avoids silent errors
    check_cfl(dt_s=dt_s, dh_km=dh, vp_max=float(np.max(vp_pad)), strict=True)

    f_nyquist   = 1.0 / (2.0 * dt_s)
    f_max_space = float(np.min(vp_pad)) / (2.0 * dh)
    f_max       = min(f_nyquist, f_max_space)
    print(f"Maximum usable frequency: {f_max:.1f} Hz")

    # Optional: plot velocity model
    if args.plot:
        plt.figure()
        plt.imshow(vp, cmap="RdBu_r", clim=[1.0, 4.5])
        plt.title("Velocity model (km/s)")
        plt.colorbar(label="vp [km/s]")
        plt.tight_layout()
        plt.show()

    # Source geometry
    src_spacing_km = args.src_spacing / 1000.0
    src_depth_km   = args.src_depth   / 1000.0
    domain_size    = (domain - 1) * dh  # physical size in km

    n_src = int(np.ceil(domain_size[1] / src_spacing_km))
    src_coords = np.column_stack([
        np.arange(0, src_spacing_km * n_src, src_spacing_km),
        np.full(n_src, src_depth_km),
    ])
    sources = (src_coords // spacing).astype(int) + [nbl, nbl]

    # Remove first and last source (paper convention)
    src_coords = src_coords[1:-1, :]
    sources    = sources[1:-1, :]

    # Receiver geometry
    rec_spacing_km = args.rec_spacing / 1000.0
    rec_depth_km   = args.rec_depth   / 1000.0

    n_rec = int(np.floor(domain_size[1] / rec_spacing_km)) + 1
    rec_coords = np.column_stack([
        np.arange(n_rec) * rec_spacing_km,
        np.full(n_rec, rec_depth_km),
    ])
    recs = (rec_coords // spacing).astype(int) + [nbl, nbl]
    recz = int(nbl + rec_depth_km / spacing[0])

    # Ricker source wavelet
    f0    = args.f0 * 1000.0          # kHz → Hz
    delay = 1.5 / f0                  # s
    t     = np.arange(0.0, tn_s, dt_s) - delay
    r     = (1.0 - 2.0 * (np.pi * f0 * t) ** 2) * np.exp(-(np.pi * f0 * t) ** 2)

    if args.plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(t + delay, r)
        plt.title("Ricker wavelet")
        plt.xlabel("Time [s]")

        f_axis = np.fft.rfftfreq(len(r), d=dt_s)
        plt.subplot(1, 2, 2)
        plt.plot(f_axis[f_axis <= f_max * 2], np.abs(np.fft.rfft(r))[f_axis <= f_max * 2])
        plt.title("Wavelet spectrum")
        plt.xlabel("Frequency [Hz]")
        plt.tight_layout()
        plt.show()

    # PML coefficients
    pmlc = generate_pml_coefficients_2d(vp_pad.shape, nbl)

    if args.plot:
        plt.figure()
        plt.imshow(pmlc.numpy(), cmap="jet")
        plt.title("PML damping coefficients")
        plt.colorbar()
        plt.show()

    # Move data to device
    wave_t = torch.from_numpy(r.astype(np.float32)).to(dev)
    vp_t   = torch.from_numpy(vp_pad.astype(np.float32)).to(dev)

    # Batched forward modelling
    batch_size    = args.batch_size
    src_batches   = [sources[i : i + batch_size] for i in range(0, len(sources), batch_size)]
    d_obs_batches = []

    for src_batch in src_batches:
        kwargs = dict(
            wave=wave_t,
            src_list=np.array(src_batch),
            domain=tuple(domain_pad),
            dt=dt_s,
            h=dh,
            dev=dev,
            recz=recz,
            b=pmlc,
            pmln=nbl,
        )
        with torch.no_grad():
            d_obs_batch = forward(c=vp_t, **kwargs)
        d_obs_batches.append(d_obs_batch.detach().cpu().numpy())

    d_obs = np.concatenate(d_obs_batches, axis=0)   # (nshots, nt, Nx_pad)

    # Select traces at receiver x-positions (strip PML offset)
    rec_x = np.array([recs[i][0] - nbl for i in range(len(recs))])
    d_obs = d_obs[:, :, rec_x]   # (nshots, nt, nrec)

    # Optional: diagnostic plots
    if args.plot:
        n_show = min(3, len(sources))
        fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 6))
        if n_show == 1:
            axes = [axes]
        for ax, shot_idx in zip(axes, range(1, n_show + 1)):
            ax.imshow(
                d_obs[shot_idx],
                cmap="RdBu_r",
                aspect=5,
                clim=[-0.2, 0.2],
                extent=[0, d_obs.shape[2] * dh, d_obs.shape[1] * dt_s, 0],
            )
            ax.set_title(f"Shot {shot_idx}")
            ax.set_xlabel("Distance [km]")
            ax.set_ylabel("Time [s]")
        plt.tight_layout()
        plt.show()

    # Save
    out_path = os.path.join("data", "shots", os.path.basename(args.vp_model_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        d_obs_list=d_obs,
        src_coordinates=src_coords,
        rec_coordinates=rec_coords,
        t0=0,
        tn=args.tn,
        dt=args.dt,
        nbl=nbl,
        spacing=spacing,
        wave=r,
        domain_pad=domain_pad,
        pmlc=pmlc.numpy(),
        domain=domain,
        allow_pickle=True,
    )
    print(f"Saved shot gathers to: {out_path}")
    print(f"  Shape: {d_obs.shape}  (nshots × nt × nrec)")


# Parameter Definition for user input

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-D acoustic forward modelling")

    parser.add_argument("--vp_model_path", type=str,
                        default="./data/v_models/marmousi_paper_sp15.npz")
    parser.add_argument("--src_spacing",   type=int,   default=300,
                        help="Source spacing [m]")
    parser.add_argument("--rec_spacing",   type=int,   default=15,
                        help="Receiver spacing [m]")
    parser.add_argument("--rec_depth",     type=int,   default=0,
                        help="Receiver depth [m]")
    parser.add_argument("--src_depth",     type=int,   default=30,
                        help="Source depth [m]")
    parser.add_argument("--f0",            type=float, default=0.008,
                        help="Dominant frequency of Ricker wavelet [kHz]")
    parser.add_argument("--tn",            type=float, default=1900,
                        help="Recording end time [ms]")
    parser.add_argument("--dt",            type=float, default=1.9,
                        help="Time step [ms]")
    parser.add_argument("--nbl",           type=int,   default=100,
                        help="Number of PML boundary layers")
    parser.add_argument("--batch_size",    type=int,   default=1000,
                        help="Sources per GPU batch")
    parser.add_argument("--plot",          action="store_true",
                        help="Show diagnostic plots")

    args = parser.parse_args()
    main(args)
