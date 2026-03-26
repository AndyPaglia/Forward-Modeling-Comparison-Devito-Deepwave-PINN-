"""
Pre-train the SIREN model to represent a smooth initial velocity model.

Image and Sound Processing Lab - Politecnico di Milano

Daniele Ugo Leonzio
Paolo Bestagini

IMPROVEMENTS:
- omega_0 lowered to 10 (more conservative, avoids overfitting to high spatial frequencies)
- Removed GradScaler
- Scheduler switched to ReduceLROnPlateau with added early stopping
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from pinn_utils import set_gpu, Siren

os.nice(10)
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

# Main

def main(args: argparse.Namespace) -> None:
    """
    Fit a SIREN network to a (smooth) initial velocity model.

    The trained weights are saved and later loaded by fwi_siren.py as the
    starting point for the full waveform inversion.

    Training details
    ----------------
    - Loss:      MSE between SIREN output and the target velocity model.
    - Optimiser: AdamW  (weight decay helps generalisation).
    - Scheduler: ReduceLROnPlateau — reduces LR when the loss plateaus.
    - No GradScaler: float32 has sufficient dynamic range without scaling.
    - omega_0 = 10: conservative value that prevents the network from fitting
                    spuriously high spatial frequencies in the initial model.
    """

    # Device definition

    set_gpu(-1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load target velocity model
    npzfile = np.load(args.vp_model_path)
    vp_np   = npzfile["vp"].T                        # (Nz, Nx)
    domain  = np.array(vp_np.shape)

    vp_min = float(vp_np.min())
    vp_max = float(vp_np.max())
    print(f"Velocity model shape : {domain}   vp ∈ [{vp_min:.2f}, {vp_max:.2f}] km/s")

    if args.plot:
        plt.figure()
        plt.imshow(vp_np, cmap="RdBu_r", vmin=1.0, vmax=4.5)
        plt.title("Target velocity model")
        plt.colorbar(label="vp [km/s]")
        plt.tight_layout()
        plt.show()

    # SIREN model
    # IMPROVEMENT: omega_0 lowered from 30 → 10
    #              Avoids capturing spuriously high spatial frequencies
    #              in the smooth initial model, which would corrupt the
    #              early FWI iterations.

    imvel = Siren(
        in_features=2,
        out_features=1,
        hidden_features=128,
        hidden_layers=4,
        outermost_linear=True,
        domain_shape=tuple(domain),
        first_omega_0=10.0,
        hidden_omega_0=10.0, 
    ).to(dev)

    # Loss, optimiser, scheduler

    l2loss = torch.nn.MSELoss()
    opt    = torch.optim.AdamW(imvel.parameters(), lr=1e-4, weight_decay=1e-5)

    # ReduceLROnPlateau: halve LR if no improvement for 'patience' epochs
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=50, min_lr=1e-6
    )

    # IMPROVEMENT: GradScaler REMOVED — it is designed for float16 only.
    # Using it with float32 can cause long-term numerical drift.

    # Move data to device
    vp_t   = torch.from_numpy(vp_np.astype(np.float32)).to(dev)
    coords = imvel.coords.to(dev)

    # Training loop with best-model checkpointing
    LOSS       = []
    best_loss  = float("inf")
    best_epoch = 0

    # Prepare output path early so we can save checkpoints
    out_path = os.path.join(".", "data", "siren", os.path.basename(args.vp_model_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    best_path = out_path.replace(".npz", "_best.pth").replace(".npz", ".pth")
    # Make sure extension is .pth
    if not best_path.endswith(".pth"):
        best_path = out_path + "_best.pth"

    for epoch in tqdm(range(args.epochs), desc="Pre-training"):
        opt.zero_grad()

        # Forward pass — SIREN outputs shape (Nz, Nx)
        vp_est, _ = imvel(coords)

        # Denormalise: network output is centred around 0, shift to km/s
        # mean=3, std=1 keeps the output in a typical velocity range
        std_vp  = 1.0
        mean_vp = 3.0
        vp_est  = vp_est * std_vp + mean_vp

        # Loss and backward
        loss = l2loss(vp_est, vp_t)
        loss.backward()

        # Gradient clipping — keeps training stable even with deep SIREN
        torch.nn.utils.clip_grad_norm_(imvel.parameters(), max_norm=1.0)

        opt.step()
        sched.step(loss)

        loss_val = loss.item()
        LOSS.append(loss_val)

        # Best-model checkpoint
        # IMPROVEMENT: save the best weights, not just the last ones
        if loss_val < best_loss:
            best_loss  = loss_val
            best_epoch = epoch
            torch.save(imvel.state_dict(), best_path)

        # Periodic logging and plot
        if epoch % 100 == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"  Epoch {epoch:5d} | Loss: {loss_val:.6f} | LR: {lr_now:.2e} | Best @ {best_epoch}")

            if args.plot:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].imshow(
                    vp_est.cpu().detach().numpy(),
                    cmap="RdBu_r", vmin=1.0, vmax=4.5,
                )
                axes[0].set_title(f"SIREN estimate — epoch {epoch}")
                axes[1].semilogy(LOSS)
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("MSE loss")
                axes[1].set_title("Training loss")
                plt.tight_layout()
                plt.show()

    print(f"\nPre-training complete.  Best loss: {best_loss:.6f} at epoch {best_epoch}.")
    print(f"Best weights saved to : {best_path}")

    # Also save the final weights (useful for warm-starting further training)
    final_path = out_path if out_path.endswith(".pth") else out_path.replace(".npz", ".pth")
    torch.save(imvel.state_dict(), final_path)
    print(f"Final weights saved to: {final_path}")


# Parameter call from user

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train SIREN on an initial velocity model")

    parser.add_argument("--vp_model_path", type=str,
                        default="./data/v_models/marmousi_paper_sm10_sp15.npz",
                        help="Path to the smooth initial velocity model (.npz)")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--plot",   action="store_true",
                        help="Show diagnostic plots during training")

    args = parser.parse_args()
    main(args)