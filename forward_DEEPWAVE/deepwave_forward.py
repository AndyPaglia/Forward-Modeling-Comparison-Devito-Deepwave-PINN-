import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar


def forward_modeling_deepwave(
    vp,
    spacing,
    src_coordinates,
    rec_coordinates,
    f0_hz,
    tn,
    dt,
    device,
    accuracy=8,
    pml_width=20,
    verbose=True,
):
    """
    Forward modeling 2D con Deepwave (coerente con Devito).

    Parametri:
    ----------
    vp : np.ndarray o torch.Tensor
        Modello di velocità (m/s), shape (nx, nz)
    spacing : tuple
        (dx, dz) in metri
    src_coordinates : np.ndarray
        Coordinate sorgenti (n_src, 2) in metri [x, z]
    rec_coordinates : np.ndarray
        Coordinate ricevitori (n_rec, 2) in metri [x, z]
    f0_hz : float
        Frequenza dominante della Ricker in Hz
    tn : float
        Tempo finale in ms
    dt : float
        Time step in ms
    """

    # ------------------------------------------------------------
    # 1. INPUT MODEL
    # ------------------------------------------------------------
    if torch.is_tensor(vp):
        v = vp.to(device)
        is_fwi = True
    else:
        v = torch.from_numpy(vp).float().to(device)
        is_fwi = False

    should_print = verbose and not is_fwi

    if should_print:
        print("\n" + "=" * 60)
        print("DEBUG - MODELLO")
        print("=" * 60)
        print(f"Shape vp           : {v.shape}")
        print(f"Velocità min / max : {v.min():.1f} / {v.max():.1f} m/s")
        print(f"Device             : {device}")

    # ------------------------------------------------------------
    # 2. PARAMETRI TEMPORALI
    # ------------------------------------------------------------
    dt_sec = dt / 1000.0
    tn_sec = tn / 1000.0
    nt = int(tn_sec / dt_sec) + 1

    peak_time = 1.5 / f0_hz

    dx = float(spacing[0])
    dz = float(spacing[1]) if len(spacing) > 1 else dx

    # ------------------------------------------------------------
    # 3. COORDINATE → INDICI (Deepwave gestisce il PML internamente)
    # ------------------------------------------------------------
    src_x_idx = torch.tensor(src_coordinates[:, 0] / dx, dtype=torch.long, device=device)
    src_z_idx = torch.tensor(src_coordinates[:, 1] / dz, dtype=torch.long, device=device)

    nx, nz = v.shape
    valid_mask = (
        (src_x_idx >= 0) & (src_x_idx < nx) &
        (src_z_idx >= 0) & (src_z_idx < nz)
    )

    src_x_idx = src_x_idx[valid_mask]
    src_z_idx = src_z_idx[valid_mask]
    n_shots = len(src_x_idx)

    if should_print:
        print(f"Numero di shot validi: {n_shots}")

    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[:, 0, 0] = src_x_idx
    source_locations[:, 0, 1] = src_z_idx

    rec_x_idx = torch.tensor(rec_coordinates[:, 0] / dx, dtype=torch.long, device=device)
    rec_z_idx = torch.tensor(rec_coordinates[:, 1] / dz, dtype=torch.long, device=device)

    n_rec = len(rec_x_idx)
    receiver_locations = torch.zeros(n_shots, n_rec, 2, dtype=torch.long, device=device)
    receiver_locations[:, :, 0] = rec_x_idx
    receiver_locations[:, :, 1] = rec_z_idx

    # ------------------------------------------------------------
    # 4. SORGENTE (RICKER)
    # ------------------------------------------------------------
    source_amplitudes = (
        deepwave.wavelets.ricker(f0_hz, nt, dt_sec, peak_time)
        .repeat(n_shots, 1, 1)
        .to(device)
    )

    # ------------------------------------------------------------
    # 5. FORWARD MODELLING
    # ------------------------------------------------------------
    if should_print:
        print("Esecuzione forward Deepwave...")

    out = scalar(
        v,
        dx,
        dt_sec,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=accuracy,
        pml_width=pml_width,
        pml_freq=f0_hz,
    )

    receiver_amplitudes = out[-1]  # (n_shots, n_rec, nt)

    if should_print:
        print(f"Output shape: {receiver_amplitudes.shape}")

    return receiver_amplitudes, source_amplitudes


# ======================================================================
# MAIN
# ======================================================================
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("CONFIGURAZIONE")
    print("=" * 60)
    print(f"Device : {device}")

    # ------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------
    npzfile = np.load(args.vp_model_path)
    vp = npzfile["vp"]
    spacing = npzfile["spacing"]

    if vp.max() < 10:  # km/s → m/s
        vp = vp * 1000.0

    domain_size_x = (vp.shape[0] - 1) * spacing[0]

    # ------------------------------------------------------------
    # GEOMETRIA (IN METRI)
    # ------------------------------------------------------------
    n_src = int(domain_size_x // args.src_spacing) + 1
    src_coordinates = np.zeros((n_src, 2))
    src_coordinates[:, 0] = np.linspace(0, domain_size_x, n_src)
    src_coordinates[:, 1] = args.src_depth

    n_rec = int(domain_size_x // args.rec_spacing) + 1
    rec_coordinates = np.zeros((n_rec, 2))
    rec_coordinates[:, 0] = np.linspace(0, domain_size_x, n_rec)
    rec_coordinates[:, 1] = args.rec_depth

    print("\nGEOMETRIA")
    print(f"Sorgenti  : {n_src} (spacing {args.src_spacing} m)")
    print(f"Ricevitori: {n_rec} (spacing {args.rec_spacing} m)")

    # ------------------------------------------------------------
    # FREQUENZA: kHz (Devito) → Hz (Deepwave)
    # ------------------------------------------------------------
    f0_hz = args.f0 * 1000.0
    print(f"f0 = {args.f0} kHz  →  {f0_hz} Hz")

    # ------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------
    receiver_amplitudes, _ = forward_modeling_deepwave(
        vp=vp,
        spacing=spacing,
        src_coordinates=src_coordinates,
        rec_coordinates=rec_coordinates,
        f0_hz=f0_hz,
        tn=args.tn,
        dt=args.dt,
        device=device,
        accuracy=args.accuracy,
        pml_width=args.pml_width,
    )

    # ------------------------------------------------------------
    # SAVE NPZ (FWI-READY)
    # ------------------------------------------------------------
    out_dir = "data/shots"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        os.path.splitext(os.path.basename(args.vp_model_path))[0] + "_deepwave.npz",
    )

    np.savez(
        out_path,
        d_obs_list=receiver_amplitudes.cpu().numpy(),
        src_coordinates=src_coordinates,
        rec_coordinates=rec_coordinates,
        spacing=spacing,
        f0_khz=args.f0,
        f0_hz=f0_hz,
        dt=args.dt,
        tn=args.tn,
        domain=vp.shape,
        nbl=args.pml_width,
    )

    print(f"\nDati salvati in: {out_path}")


    # ------------------------------------------------------------
    # PLOT E SALVATAGGIO SHOT CENTRALE
    # ------------------------------------------------------------
    
    shot_idx = receiver_amplitudes.shape[0] // 2
    shot = receiver_amplitudes[shot_idx].cpu().numpy()

    # Calcolo vlim per il contrasto (simile all'Immagine 2)
    vlim = np.percentile(np.abs(shot), 98)

    plt.figure(figsize=(10, 8))
    
    # Plotting con indici (no extent) per avere Channel e Time Sample
    im = plt.imshow(
        shot.T, 
        cmap="gray", 
        aspect="auto", 
        vmin=-vlim, 
        vmax=vlim
    )

    # Aggiunta Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Amplitude')

    # Titoli ed Etichette come richiesto
    plt.title(f"Common Shot Gather - Shot {shot_idx}\n(Position: {src_coordinates[shot_idx, 0]:.1f} m)")
    plt.xlabel("Channel (Receiver #)")
    plt.ylabel("Time Sample")

    plt.tight_layout()

    # --- PARTE AGGIUNTA PER IL SALVATAGGIO ---
    # Costruiamo il percorso salvando nella stessa cartella dell'NPZ
    png_path = out_path.replace(".npz", "_style2.png")
    plt.savefig(png_path, dpi=300)
    print(f"Immagine stile 2 salvata in: {png_path}")
    # ------------------------------------------

    plt.show()


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Deepwave Forward Modeling (Devito-compatible)")
    parser.add_argument("--vp_model_path", type=str, default="./data/v_models/marmousi_sp25.npz")
    parser.add_argument("--src_spacing", type=float, default=100.0)
    parser.add_argument("--rec_spacing", type=float, default=25.0)
    parser.add_argument("--src_depth", type=float, default=20.0)
    parser.add_argument("--rec_depth", type=float, default=20.0)
    parser.add_argument("--f0", type=float, default=0.005, help="kHz (Devito-style)")
    parser.add_argument("--tn", type=float, default=6000.0, help="ms")
    parser.add_argument("--dt", type=float, default=3.0, help="ms")
    parser.add_argument("--pml_width", type=int, default=20)
    parser.add_argument("--accuracy", type=int, default=8)

    main(parser.parse_args())
