# IMPORTING LIBRARIES
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
from tqdm import tqdm

# MAIN FUNCTION FOR DEEPWAVE FORWARD MODELING WITH APERTURE PARAMETER TO GENERALIZE
def run_deepwave_forward_with_aperture():
    # PARAMETERS
    vp_model_path = "./data/v_models/marmousi_sp25.npz"
    tn = 6000.0      
    dt = 3.0         
    f0 = 0.005       
    src_spacing = 100.0
    rec_spacing = 25.0
    src_depth = 25.0  # 25 m = 1 cell
    rec_depth = 25.0
    aperture = 12000.0 
    nbl = 200       
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # LOADING AND VELOCITY MODEL PREPARATION
    npzfile = np.load(vp_model_path)
    vp = npzfile["vp"]           
    spacing = npzfile["spacing"] 
    dx, dz = spacing
    
    # CHECK TO SEE IF IS COHERENT WITH DEEPWAVE EXPECTATIONS
    if vp.max() < 10:
        print("Scaling velocity from km/s to m/s...")
        vp = vp * 1000.0
        
    model_vp = torch.from_numpy(vp).float().to(device).T 
    nz, nx = model_vp.shape

    # COORDINATE SETUP AND WAVELET DEFINITION
    all_src_x = np.arange(0, nx * dx, src_spacing)
    all_rec_x = np.arange(0, nx * dx, rec_spacing)
    
    n_src = len(all_src_x)
    nt = int(tn / dt) + 1
    dt_sec = dt / 1000.0
    f0_hz = f0 * 1000.0

    wavelet = deepwave.wavelets.ricker(f0_hz, nt, dt_sec, 1/f0_hz).to(device)
    
    d_obs_list = []

    print(f"Esecuzione Forward Modeling con apertura {aperture}m...")
    for i in tqdm(range(n_src)):
        curr_src_x = all_src_x[i]
        
        min_dist = max(0, curr_src_x - aperture)
        max_dist = min((nx - 1) * dx, curr_src_x + aperture)
        mask = (all_rec_x >= min_dist) & (all_rec_x <= max_dist)
        curr_rec_x = all_rec_x[mask]
        
        src_loc = torch.tensor([[[src_depth / dz, curr_src_x / dx]]]).round().long().to(device)
        rec_locs = torch.zeros(1, len(curr_rec_x), 2)
        rec_locs[0, :, 0] = rec_depth / dz 
        rec_locs[0, :, 1] = torch.from_numpy(curr_rec_x / dx) 
        rec_locs = rec_locs.round().long().to(device)
        
        source_amplitude = wavelet.view(1, 1, nt)

        # CENTRAL PART FOR FORWARD MODELING
        out = scalar(
            model_vp, 
            grid_spacing=dx, 
            dt=dt_sec,
            source_amplitudes=source_amplitude,
            source_locations=src_loc,
            receiver_locations=rec_locs,
            pml_width=[nbl, nbl, nbl, nbl],
            pml_freq=f0_hz, # PML frequency set to source dominant frequency
            accuracy=4
        )
        
        shot_data = out[-1].cpu().numpy()[0].T 
        d_obs_list.append(shot_data)

    # PLOTTING THE CENTRAL SHOT 
    shot_idx = n_src // 2
    shot_data = -1 * d_obs_list[shot_idx]
    
    # AUTOMATIC SCALING USING PERCENTILE. 
    # IT USES THE 98° PERCENTILE OF THE ABSOLUTE AMPLITUDES TO SET DYNAMIC RANGE
    # THE PRESSURE VALUE ARE HIGHER THAN THE DEVITO ONE
    # PRESSURE VALUE BETWEEN -8.5 AND 8.5 FOR THE CENTRAL SHOT WITH APERTURE 6000m
    # TO TRY AND SEE UNCOMMENT THE NEXT LINE AND COMMENT THE FIXED VALUE

    # vmax = np.percentile(np.abs(shot_data), 98) 
    
    vmax = 5.0  # FIXED VALUE TO MATCH DEVITO PLOTTING

    plt.figure(figsize=(10, 8))
    plt.imshow(shot_data, cmap='gray', aspect='auto', 
               extent=[0, shot_data.shape[1], tn, 0], vmin=-vmax, vmax=vmax)
    
    plt.title(f'Deepwave Shot Centrale (Indice {shot_idx})')
    plt.xlabel('Indice Ricevitore')
    plt.ylabel('Tempo (ms)')
    plt.colorbar(label='Ampiezza Pressione')
    plt.savefig(os.path.join("data", "shots", 
                             f"{os.path.splitext(os.path.basename(vp_model_path))[0]}_aperture{int(aperture)}.png"),
                bbox_inches='tight')
    plt.show()

    # ========== PREPARAZIONE DATI PER FWI ==========
    print("\n Preparazione dati per FWI...")

    # Converti lista di shot in array numpy con padding
    max_n_rec = max(shot.shape[1] for shot in d_obs_list)
    nt_data = d_obs_list[0].shape[0]
    n_shots = len(d_obs_list)

    print(f"   Numero shots: {n_shots}")
    print(f"   Max ricevitori: {max_n_rec}")
    print(f"   Time samples: {nt_data}")

    # Crea array con padding (n_shots, max_n_rec, nt)
    receiver_data = np.zeros((n_shots, max_n_rec, nt_data), dtype=np.float32)
    for i, shot in enumerate(d_obs_list):
        # shot è (nt, n_rec) → trasponilo per avere (n_rec, nt)
        receiver_data[i, :shot.shape[1], :] = shot.T

    # Coordinate sorgenti (n_shots, 2) - formato [x, z]
    src_coordinates = np.zeros((n_shots, 2))
    src_coordinates[:, 0] = all_src_x  # x coordinates
    src_coordinates[:, 1] = src_depth  # z coordinates (costante)

    # Coordinate ricevitori (n_rec_total, 2) - tutti i ricevitori possibili
    rec_coordinates = np.zeros((len(all_rec_x), 2))
    rec_coordinates[:, 0] = all_rec_x  # x coordinates
    rec_coordinates[:, 1] = rec_depth  # z coordinates (costante)

    # Salva file NPZ per FWI
    fwi_filename = os.path.join("data", "shots", 
                               f"{os.path.splitext(os.path.basename(vp_model_path))[0]}_aperture{int(aperture)}_fwi.npz")

    np.savez(
        fwi_filename,
        receiver_data=receiver_data,        # (n_shots, n_rec, nt) - OBBLIGATORIO
        src_coordinates=src_coordinates,     # (n_shots, 2) - OBBLIGATORIO
        rec_coordinates=rec_coordinates,     # (n_rec, 2) - OBBLIGATORIO
        spacing=spacing,                     # (dx, dz) - OBBLIGATORIO
        f0=f0_hz,                           # Hz - OBBLIGATORIO
        dt=dt,                              # ms - OBBLIGATORIO
        tn=tn,                              # ms - OBBLIGATORIO
        aperture=aperture                   # m - OPZIONALE
    )

    print(f"\n File FWI salvato: {fwi_filename}")

    # STAMPA CONTENUTO DEL FILE NPZ
    print(f"\n Contenuto del file NPZ salvato:")
    with np.load(fwi_filename) as data:
        for key in data.files:
            val = data[key]
            if isinstance(val, np.ndarray):
                print(f"   {key:20s}: shape={str(val.shape):25s} dtype={val.dtype}")
            else:
                print(f"   {key:20s}: {val}")

    return d_obs_list

if __name__ == "__main__":
    shots = run_deepwave_forward_with_aperture()