"""
Full Waveform Inversion (FWI) using a SIREN as implicit velocity parametrisation.

Image and Sound Processing Lab - Politecnico di Milano

Daniele Ugo Leonzio
Paolo Bestagini

IMPROVEMENTS v2:
- GradScaler REMOVED 
- Scheduler: CosineAnnealingWarmRestarts instead of CosineAnnealingLR
    → periodic LR resets prevent the optimiser from freezing
- Gradient clipping added (clip_grad_norm_ = 1.0) for training stability
- Normalised loss (divides by observed data amplitude) for balanced gradients
- Hard NaN check
- Multi-scale (frequency continuation) mode via --multiscale flag
- omega_0 reduced to 10 (consistent with pretrain_siren.py)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from pinn_utils import bandpass_shots, forward, set_gpu, Siren

os.nice(10)
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

# Helpers

def _select_shots(policy: str, epoch: int, n_total: int, n_per_epoch: int) -> np.ndarray:
    """
    Return an array of shot indices according to the selection policy.

    Policies
    --------
    random     : uniformly random subset of size n_per_epoch.
    sequential : rolling window that cycles through all shots in order.
    spaced     : equidistant shots across the full dataset.
    """
    if policy == "random":
        return np.random.choice(n_total, n_per_epoch, replace=False)
    elif policy == "sequential":
        return np.arange(epoch * n_per_epoch, (epoch + 1) * n_per_epoch) % n_total
    elif policy == "spaced":
        step = n_total // n_per_epoch
        return np.arange(epoch % step, n_total, step)
    else:
        raise ValueError(f"Unknown shot selection policy: '{policy}'. "
                         f"Choose from: random, sequential, spaced.")

# Main

def main(args: argparse.Namespace) -> None:
    """
    Run SIREN-parametrised FWI.

    Algorithm overview
    ------------------
    For each epoch:
      1. The SIREN generates a velocity model from fixed spatial coordinates.
      2. A mini-batch of shots is selected.
      3. Acoustic forward modelling produces synthetic seismograms.
      4. The normalised MSE between synthetics and observed data is computed.
      5. Gradients backpropagate through the entire wave simulation into the
         SIREN weights (backpropagation through time over the FD loop).
      6. Gradient clipping + AdamW step + scheduler step.
    """

    # Device
    set_gpu(-1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load observed data and metadata
    npzfile = np.load(args.obs_data_path, allow_pickle=True)
    d_obs_np = npzfile["d_obs_list"]                    # (nshots, nt, nrec)

    dt_ms       = float(npzfile["dt"])
    dt_s        = dt_ms / 1000.0
    spacing     = npzfile["spacing"]
    dh          = float(spacing[0])
    wave_np     = npzfile["wave"]
    domain_pad  = tuple(npzfile["domain_pad"].tolist())
    domain      = tuple(npzfile["domain"].tolist())
    nbl         = int(npzfile["nbl"])
    pmlc_np     = npzfile["pmlc"]

    fwi_iterations      = args.fwi_iterations
    shots_per_epoch     = args.shots_per_epoch
    shot_selection_policy = args.shot_selection_policy
    use_all_shots       = args.all_shots

    # Output directory
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    png_dir = os.path.join(out_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    # Geometry
    rec_coordinates = npzfile["rec_coordinates"]
    src_coordinates = npzfile["src_coordinates"]
    sources  = (src_coordinates // spacing).astype(int) + [nbl, nbl]
    recz     = int(nbl + float(rec_coordinates[0, 1]) / float(spacing[0]))

    n_total_shots  = len(sources)
    num_mini_batches = n_total_shots // shots_per_epoch if use_all_shots else 1
    epochs_per_plot  = 1 if use_all_shots else 20

    print(f"Total shots        : {n_total_shots}")
    print(f"Shots per epoch    : {shots_per_epoch}")
    print(f"Mini-batches/epoch : {num_mini_batches}")
    print(f"FWI iterations     : {fwi_iterations}")
    print(f"Shot policy        : {shot_selection_policy}")

    # Optional: multi-scale frequency schedule
    # Divide the inversion into frequency bands that widen over time.
    # Low frequencies first → avoids cycle-skipping.
    if args.multiscale:
        # Three stages: [0, T/3) → ≤5 Hz | [T/3, 2T/3) → ≤10 Hz | rest → full
        stage_len = fwi_iterations // 3
        freq_schedule = [
            (0,            stage_len,          1.0,  5.0),
            (stage_len,    2 * stage_len,       1.0, 10.0),
            (2 * stage_len, fwi_iterations,    1.0,  None),  # None → no filtering
        ]
        print("Multi-scale mode ON:")
        for s, e, fl, fh in freq_schedule:
            label = f"{fh} Hz" if fh else "full band"
            print(f"  epochs {s:4d}–{e:4d} → f ≤ {label}")
    else:
        freq_schedule = [(0, fwi_iterations, None, None)]

    # Optional diagnostic plot of observed data
    if args.plot:
        plt.figure()
        plt.imshow(d_obs_np[0], cmap="RdBu_r", aspect="auto", clim=[-0.5, 0.5])
        plt.title("Observed data — shot 0")
        plt.colorbar()
        plt.show()

    # SIREN model
    # IMPROVEMENT: omega_0 = 10 (was 30) — consistent with pretrain_siren
    imvel = Siren(
        in_features=2,
        out_features=1,
        hidden_features=128,
        hidden_layers=4,
        outermost_linear=True,
        domain_shape=domain,
        first_omega_0=10.0,   # ← was 30
        hidden_omega_0=10.0,  # ← was 30
        pretrained=os.path.abspath(args.siren_path),
    ).to(dev)

    # Loss
    l2loss = torch.nn.MSELoss()

    # Optimiser
    opt = torch.optim.AdamW(imvel.parameters(), lr=1e-4, weight_decay=1e-5)

    # IMPROVEMENT: CosineAnnealingWarmRestarts instead of CosineAnnealingLR
    # T_0=200 → first restart after 200 epochs
    # T_mult=2 → subsequent cycles double in length: 200, 400, 800, …
    # This prevents the LR from reaching zero and "freezing" the optimiser,
    # which was the main cause of the loss blow-up after ~1000 iterations.
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=200, T_mult=2, eta_min=1e-6
    )

    # IMPROVEMENT: GradScaler REMOVED.
    # GradScaler is designed for float16 mixed precision only.  With float32
    # it introduces a slowly-drifting loss scale that can cause gradient
    # underflow / overflow silently after many iterations.

    # Move static tensors to device
    obs_t    = torch.from_numpy(d_obs_np).float().to(dev)       # (nshots, nt, nrec)
    wave_t   = torch.from_numpy(wave_np.astype(np.float32)).to(dev)
    coords   = imvel.coords.to(dev)
    pmlc_t   = torch.from_numpy(pmlc_np.astype(np.float32)).to(dev)

    # FWI loop
    LOSS     = []
    LOSS_ALL = []
    best_loss  = float("inf")
    best_epoch = 0

    # Amplitude of observed data for loss normalisation (scalar, on GPU)
    obs_scale = obs_t.abs().max().clamp(min=1e-8)

    # Shared forward kwargs (src_list will be updated inside the loop)
    fwd_kwargs = dict(
        wave=wave_t,
        src_list=np.array(sources),
        domain=domain_pad,
        dt=dt_s,
        h=dh,
        dev=dev,
        recz=recz,
        b=pmlc_t,
        pmln=nbl,
    )

    for epoch in tqdm(range(fwi_iterations), desc="FWI"):

        opt.zero_grad()

        # Determine active frequency band for this epoch (multi-scale FWI)
        active_f_low, active_f_high = None, None
        if args.multiscale:
            for (s, e, fl, fh) in freq_schedule:
                if s <= epoch < e:
                    active_f_low, active_f_high = fl, fh
                    break

        INNER_LOSS = []

        for inner_epoch in range(num_mini_batches):
            src_idx = _select_shots(shot_selection_policy, epoch * num_mini_batches + inner_epoch,
                                    n_total_shots, shots_per_epoch)
            fwd_kwargs["src_list"] = np.array(sources)[src_idx]

            # Build velocity model from SIREN
            vp, _ = imvel(coords)                        # (Nz, Nx)

            std_vp  = 1.0
            mean_vp = 3.0
            vp = vp * std_vp + mean_vp
            vp = torch.clamp(vp, min=1.5, max=4.5)

            # Pad with PML border (replicate boundary values)
            vp_pad = torch.nn.functional.pad(
                vp.unsqueeze(0).unsqueeze(0),
                (nbl, nbl, nbl, nbl),
                mode="replicate",
            )[0, 0]                                      # (Nz_pad, Nx_pad)

            # Forward modelling
            syn = forward(c=vp_pad, **fwd_kwargs)        # (batch, nt, nrec)

            # NaN guard — catches numerical blow-ups immediately
            if torch.isnan(syn).any():
                raise RuntimeError(
                    f"NaN in synthetics at epoch {epoch}. "
                    "Check CFL condition and PML parameters."
                )

            # IMPROVEMENT: normalised loss
            # Dividing by obs_scale makes the loss dimensionless and keeps
            # the gradient magnitude consistent across shots with very
            # different amplitudes (near-source vs far-offset).
            obs_batch = obs_t[src_idx]
            loss = l2loss(syn / obs_scale, obs_batch / obs_scale)

            # Apply multi-scale frequency mask if active
            if active_f_high is not None:
                syn_np      = syn.detach().cpu().numpy()
                obs_np      = obs_batch.detach().cpu().numpy()
                syn_filt    = bandpass_shots(syn_np, active_f_low, active_f_high, dt_s)
                obs_filt    = bandpass_shots(obs_np, active_f_low, active_f_high, dt_s)
                syn_f  = torch.from_numpy(syn_filt).to(dev)
                obs_f  = torch.from_numpy(obs_filt).to(dev)
                scale_f = obs_f.abs().max().clamp(min=1e-8)
                loss = l2loss(syn_f / scale_f, obs_f / scale_f)

            loss = loss / num_mini_batches
            INNER_LOSS.append(loss.item())
            LOSS_ALL.append(loss.item())

            # Backward
            loss.backward()

        # IMPROVEMENT: gradient clipping
        # Caps the gradient norm before the optimiser step.
        # Prevents "exploding gradient" events that destroy previously
        # accumulated improvements — this was the most visible symptom
        # described (sudden loss blow-up after many iterations).
        torch.nn.utils.clip_grad_norm_(imvel.parameters(), max_norm=1.0)

        opt.step()
        sched.step(epoch)   # CosineAnnealingWarmRestarts wants the epoch index

        epoch_loss = float(np.mean(INNER_LOSS))
        LOSS.append(epoch_loss)

        # Best-model checkpoint
        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_epoch = epoch
            if not args.debug:
                best_path = os.path.join(out_dir, "fwi_best_model.pth")
                torch.save(imvel.state_dict(), best_path)

        # Periodic plot
        if epoch % epochs_per_plot == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"  Epoch {epoch:5d} | Loss: {epoch_loss:.6e} | LR: {lr_now:.2e} | Best @ {best_epoch}")

            inverted = vp_pad.detach().cpu().numpy()[nbl:-nbl, nbl:-nbl]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.imshow(inverted, cmap="RdBu_r", aspect="auto", vmin=1.0, vmax=4.5)
            ax1.set_title(f"Inverted velocity — epoch {epoch}")
            ax1.set_xlabel("x [samples]")
            ax1.set_ylabel("z [samples]")

            ax2.semilogy(LOSS, color="steelblue", lw=1.5, label="epoch loss")
            ax2.axvline(best_epoch, color="red", ls="--", lw=1, label=f"best ({best_epoch})")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Normalised MSE")
            ax2.legend()
            plt.tight_layout()

            if not args.debug:
                plt.savefig(f"{png_dir}/epoch{epoch:05d}.png", dpi=150, bbox_inches="tight")
            if args.plot:
                plt.show()
            plt.close(fig)

    # Save final results
    final_vp = vp_pad.detach().cpu().numpy()[nbl:-nbl, nbl:-nbl]

    if not args.debug:
        np.savez(
            os.path.join(out_dir, "fwi_results.npz"),
            vp=final_vp,
            LOSS=np.array(LOSS),
            LOSS_ALL=np.array(LOSS_ALL),
            allow_pickle=True,
        )
        torch.save(imvel.state_dict(), os.path.join(out_dir, "fwi_final_model.pth"))
        print(f"\nResults saved to: {out_dir}")
        print(f"Best model (epoch {best_epoch}, loss {best_loss:.6e}) → fwi_best_model.pth")
        print(f"Final model → fwi_final_model.pth")


# Parameter definition for user input

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIREN-parametrised Full Waveform Inversion")

    parser.add_argument("--obs_data_path",  type=str,
                        default="./data/shots/marmousi_paper_sp15.npz",
                        help="Path to the observed shot gathers (.npz)")
    parser.add_argument("--siren_path",     type=str,
                        default="./data/siren/marmousi_paper_sm10_sp15.pth",
                        help="Path to the pre-trained SIREN weights (.pth)")
    parser.add_argument("--out_dir",        type=str,
                        default="./data/output/marmousi_fwi_CLAUDE",
                        help="Output directory")
    parser.add_argument("--fwi_iterations", type=int,   default=10000,
                        help="Total number of FWI epochs")
    parser.add_argument("--shots_per_epoch",type=int,   default=13,
                        help="Number of shots per mini-batch")
    parser.add_argument("--shot_selection_policy", type=str, default="random",
                        choices=["random", "sequential", "spaced"],
                        help="Shot selection strategy per epoch")
    parser.add_argument("--all_shots",      action="store_true",
                        help="Use ALL shots every epoch (splits into mini-batches internally)")
    parser.add_argument("--multiscale",     action="store_true",
                        help="Enable multi-scale (frequency continuation) FWI")
    parser.add_argument("--plot",           action="store_true",
                        help="Show diagnostic plots")
    parser.add_argument("--debug",          action="store_true",
                        help="Debug mode: do not save results to disk")

    args = parser.parse_args()
    main(args)
