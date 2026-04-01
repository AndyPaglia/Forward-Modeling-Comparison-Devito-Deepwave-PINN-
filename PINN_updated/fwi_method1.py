"""
Full Waveform Inversion (FWI) – METHOD 1: Hard column override
==============================================================
Identico a fwi.py con una considerazione in più (vincolo maggiore): dopo che la SIREN produce il
modello di velocità, le colonne "note" (prese dal modello VERO) vengono
sovrascritte con i valori esatti prima del forward modelling.

Tre configurazioni via --known_col_spacing:
  25  → una colonna nota ogni 25
  50  → una colonna nota ogni 50
 100  → una colonna nota ogni 100

Image and Sound Processing Lab - Politecnico di Milano
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

    # Device
    # set_gpu(-1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load observed data and metadata
    npzfile = np.load(args.obs_data_path, allow_pickle=True)
    d_obs_np = npzfile["d_obs_list"]

    dt_ms       = float(npzfile["dt"])
    dt_s        = dt_ms / 1000.0
    spacing     = npzfile["spacing"]
    dh          = float(spacing[0])
    wave_np     = npzfile["wave"]
    domain_pad  = tuple(npzfile["domain_pad"].tolist())
    domain      = tuple(npzfile["domain"].tolist())
    nbl         = int(npzfile["nbl"])
    pmlc_np     = npzfile["pmlc"]

    fwi_iterations        = args.fwi_iterations
    shots_per_epoch       = args.shots_per_epoch
    shot_selection_policy = args.shot_selection_policy
    use_all_shots         = args.all_shots

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

    n_total_shots    = len(sources)
    num_mini_batches = n_total_shots // shots_per_epoch if use_all_shots else 1
    epochs_per_plot  = 1 if use_all_shots else 20

    print(f"[Method 1 – Hard column override]")
    print(f"Total shots        : {n_total_shots}")
    print(f"Shots per epoch    : {shots_per_epoch}")
    print(f"FWI iterations     : {fwi_iterations}")
    print(f"Shot policy        : {shot_selection_policy}")

    # Multi-scale frequency schedule
    if args.multiscale:
        stage_len = fwi_iterations // 3
        freq_schedule = [
            (0,             stage_len,         1.0,  5.0),
            (stage_len,     2 * stage_len,     1.0, 10.0),
            (2 * stage_len, fwi_iterations,    1.0,  None),
        ]
        print("Multi-scale mode ON:")
        for s, e, fl, fh in freq_schedule:
            print(f"  epochs {s:4d}-{e:4d} -> f <= {f'{fh} Hz' if fh else 'full band'}")
    else:
        freq_schedule = [(0, fwi_iterations, None, None)]

    # [M1] Carica colonne note dal modello VERO
    true_vp        = np.load(args.true_vp_path)["vp"].T # (Nz, Nx)
    col_spacing    = args.known_col_spacing
    known_col_idx  = np.arange(0, domain[1], col_spacing) # indici colonne note
    known_cols_t   = torch.from_numpy(
                         true_vp[:, known_col_idx].astype(np.float32)
                     ).to(dev) # (Nz, n_known)
    col_idx_t      = torch.tensor(known_col_idx, dtype=torch.long, device=dev)
    print(f"Known col spacing  : every {col_spacing} cols  ({len(known_col_idx)}/{domain[1]} known)")

    if args.plot:
        plt.figure()
        plt.imshow(d_obs_np[0], cmap="RdBu_r", aspect="auto", clim=[-0.5, 0.5])
        plt.title("Observed data - shot 0")
        plt.colorbar()
        plt.show()

    # SIREN model
    imvel = Siren(
        in_features=2,
        out_features=1,
        hidden_features=128,
        hidden_layers=4,
        outermost_linear=True,
        domain_shape=domain,
        first_omega_0=10.0,
        hidden_omega_0=10.0,
        pretrained=os.path.abspath(args.siren_path),
    ).to(dev)

    # Loss, optimiser, scheduler
    l2loss = torch.nn.MSELoss()
    opt    = torch.optim.AdamW(imvel.parameters(), lr=1e-4, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=200, T_mult=2, eta_min=1e-6
    )

    # Move static tensors to device
    obs_t    = torch.from_numpy(d_obs_np).float().to(dev)
    wave_t   = torch.from_numpy(wave_np.astype(np.float32)).to(dev)
    coords   = imvel.coords.to(dev)
    pmlc_t   = torch.from_numpy(pmlc_np.astype(np.float32)).to(dev)

    LOSS, LOSS_ALL = [], []
    best_loss, best_epoch = float("inf"), 0
    obs_scale = obs_t.abs().max().clamp(min=1e-8)

    fwd_kwargs = dict(
        wave=wave_t, src_list=np.array(sources), domain=domain_pad,
        dt=dt_s, h=dh, dev=dev, recz=recz, b=pmlc_t, pmln=nbl,
    )

    for epoch in tqdm(range(fwi_iterations), desc="FWI [M1]"):
        opt.zero_grad()

        active_f_low, active_f_high = None, None
        if args.multiscale:
            for (s, e, fl, fh) in freq_schedule:
                if s <= epoch < e:
                    active_f_low, active_f_high = fl, fh
                    break

        INNER_LOSS = []

        for inner_epoch in range(num_mini_batches):
            src_idx = _select_shots(shot_selection_policy,
                                    epoch * num_mini_batches + inner_epoch,
                                    n_total_shots, shots_per_epoch)
            fwd_kwargs["src_list"] = np.array(sources)[src_idx]

            # Build velocity model from SIREN
            vp, _ = imvel(coords)
            vp = vp * 1.0 + 3.0
            vp = torch.clamp(vp, min=1.5, max=4.5)

            # [M1] Sovrascrittura colonne note — unica modifica rispetto a fwi.py come ha detto prof
            vp = torch.where(
                torch.zeros_like(vp, dtype=torch.bool).index_fill_(1, col_idx_t, True),
                torch.zeros_like(vp).index_copy_(1, col_idx_t, known_cols_t),
                vp,
            )

            vp_pad = torch.nn.functional.pad(
                vp.unsqueeze(0).unsqueeze(0), (nbl, nbl, nbl, nbl), mode="replicate",
            )[0, 0]

            syn = forward(c=vp_pad, **fwd_kwargs)

            if torch.isnan(syn).any():
                raise RuntimeError(f"NaN in synthetics at epoch {epoch}.")

            obs_batch = obs_t[src_idx]
            loss = l2loss(syn / obs_scale, obs_batch / obs_scale)

            if active_f_high is not None:
                syn_filt = bandpass_shots(syn.detach().cpu().numpy(), active_f_low, active_f_high, dt_s)
                obs_filt = bandpass_shots(obs_batch.detach().cpu().numpy(), active_f_low, active_f_high, dt_s)
                syn_f, obs_f = torch.from_numpy(syn_filt).to(dev), torch.from_numpy(obs_filt).to(dev)
                loss = l2loss(syn_f / obs_f.abs().max().clamp(min=1e-8),
                              obs_f / obs_f.abs().max().clamp(min=1e-8))

            loss = loss / num_mini_batches
            INNER_LOSS.append(loss.item())
            LOSS_ALL.append(loss.item())
            loss.backward()

        torch.nn.utils.clip_grad_norm_(imvel.parameters(), max_norm=1.0)
        opt.step()
        sched.step(epoch)

        epoch_loss = float(np.mean(INNER_LOSS))
        LOSS.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch
            if not args.debug:
                torch.save(imvel.state_dict(), os.path.join(out_dir, "fwi_best_model.pth"))

        if epoch % epochs_per_plot == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"  Epoch {epoch:5d} | Loss: {epoch_loss:.6e} | LR: {lr_now:.2e} | Best @ {best_epoch}")

            inverted = vp_pad.detach().cpu().numpy()[nbl:-nbl, nbl:-nbl]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.imshow(inverted, cmap="RdBu_r", aspect="auto", vmin=1.0, vmax=4.5)
            ax1.set_title(f"Inverted Vp - epoch {epoch}  [M1, spacing={col_spacing}]")
            ax1.set_xlabel("x [samples]"); ax1.set_ylabel("z [samples]")
            for ci in known_col_idx:
                ax1.axvline(x=ci, color="k", linewidth=0.4, alpha=0.5)
            ax2.semilogy(LOSS, color="steelblue", lw=1.5, label="epoch loss")
            ax2.axvline(best_epoch, color="red", ls="--", lw=1, label=f"best ({best_epoch})")
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("Normalised MSE"); ax2.legend()
            plt.tight_layout()
            if not args.debug:
                plt.savefig(f"{png_dir}/epoch{epoch:05d}.png", dpi=150, bbox_inches="tight")
            if args.plot:
                plt.show()
            plt.close(fig)

    # Save
    final_vp = vp_pad.detach().cpu().numpy()[nbl:-nbl, nbl:-nbl]
    if not args.debug:
        np.savez(os.path.join(out_dir, "fwi_results.npz"),
                 vp=final_vp, LOSS=np.array(LOSS), LOSS_ALL=np.array(LOSS_ALL), allow_pickle=True)
        torch.save(imvel.state_dict(), os.path.join(out_dir, "fwi_final_model.pth"))
        print(f"\nResults saved to: {out_dir}")
        print(f"Best model (epoch {best_epoch}, loss {best_loss:.6e}) -> fwi_best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FWI Method 1 - hard column override")

    parser.add_argument("--obs_data_path",  type=str, default="./data/shots/marmousi_paper_sp15.npz")
    parser.add_argument("--siren_path",     type=str, default="./data/siren/marmousi_paper_sm10_sp15.pth")
    parser.add_argument("--out_dir",        type=str, default="./data/output/fwi_method1")
    parser.add_argument("--fwi_iterations", type=int, default=10000)
    parser.add_argument("--shots_per_epoch",type=int, default=5)
    parser.add_argument("--shot_selection_policy", type=str, default="random",
                        choices=["random", "sequential", "spaced"])
    parser.add_argument("--all_shots",   action="store_true")
    parser.add_argument("--multiscale",  action="store_true")
    parser.add_argument("--plot",        action="store_true")
    parser.add_argument("--debug",       action="store_true")
    # Argomenti specifici M1
    parser.add_argument("--true_vp_path",      type=str, default="./data/v_models/marmousi_paper_sp15.npz",
                        help="Modello vero da cui estrarre le colonne note")
    parser.add_argument("--known_col_spacing", type=int, default=50, choices=[25, 50, 100],
                        help="Una colonna nota ogni N colonne (25 | 50 | 100)")

    args = parser.parse_args()
    main(args)
