# IMPORTING LIBRARIES
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
from tqdm import tqdm

# DEVICE CONFIGURATION
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" FWI in esecuzione su: {device}")

def run_deepwave_fwi():
    # DATA LOADING AND GEOMETRY SETUP
    data_path = "./data/shots/marmousi_sp25_aperture12000_fwi.npz" # controlla sempre questi path!!
    v0_path = "./data/v_models/marmousi_sm10_sp25.npz" # guarda commento sopra, attenzione!!
    
    fwi_data = np.load(data_path)
    d_obs = torch.from_numpy(fwi_data['receiver_data']).to(device) # (n_shots, n_rec, nt)
    src_coords = fwi_data['src_coordinates']
    rec_coords_all = fwi_data['rec_coordinates']
    dx, dz = fwi_data['spacing']
    dt_sec = fwi_data['dt'] / 1000.0
    f0_hz = fwi_data['f0']
    aperture = fwi_data['aperture']
    
    # MODEL INITIALIZATION --> STARTING POINT FOR INVERSION
    v0_np = np.load(v0_path)["vp"]
    if v0_np.max() < 10: v0_np *= 1000.0 # m/s
    
    # CHECK DIMENSIONS BECAUSE DEEPWAVE IS (nz, nx)
    model_vp = torch.tensor(v0_np.T, dtype=torch.float32, device=device).requires_grad_(True)
    nz, nx = model_vp.shape

    # FWI PARAMETERS AND OPTIMIZATION
    # WE ARE USING ADAM OPTIMIZER
    optimizer = torch.optim.Adam([model_vp], lr=10.0) 
    n_iterations = 1000 # Da mettere 10000 
    v_min, v_max = 1500.0, 4500.0
    nbl = 200 # Boundary layers
    
    # RICKER WAVELET 
    nt = d_obs.shape[2]
    wavelet = deepwave.wavelets.ricker(f0_hz, nt, dt_sec, 1/f0_hz).to(device)

    # OPTIMIZATION LOOP
    print(f"START INVERSION: {n_iterations} iterazioni...")
    

    for iter in range(n_iterations):
        optimizer.zero_grad()
        total_loss = 0
        
        # One shot at a time (or mini-batch) to manage GPU memory
        for i in tqdm(range(len(src_coords)), desc=f"Iter {iter+1}", leave=False):
            # Manage aperture and source/receiver locations
            curr_src_x = src_coords[i, 0]
            src_loc = torch.tensor([[[src_coords[i, 1]/dz, curr_src_x/dx]]]).round().long().to(device)
            
            # Check which receivers are within the aperture for the current source
            mask = (rec_coords_all[:, 0] >= curr_src_x - aperture) & \
                   (rec_coords_all[:, 0] <= curr_src_x + aperture)
            curr_rec_coords = rec_coords_all[mask]
            
            rec_locs = torch.zeros(1, len(curr_rec_coords), 2, device=device)
            rec_locs[0, :, 0] = torch.from_numpy(curr_rec_coords[:, 1] / dz)
            rec_locs[0, :, 1] = torch.from_numpy(curr_rec_coords[:, 0] / dx)
            rec_locs = rec_locs.round().long()

            # FORWARD MODELING - smoothed model
            out = scalar(
                model_vp, dx, dt_sec,
                source_amplitudes=wavelet.view(1, 1, nt),
                source_locations=src_loc,
                receiver_locations=rec_locs,
                pml_width=[nbl]*4, pml_freq=f0_hz
            )
            
            # LOSS CALCULATION
            d_syn = out[-1][0] # (n_rec, nt)
            d_real = d_obs[i, :len(curr_rec_coords), :]
            
            loss = torch.nn.functional.mse_loss(d_syn, d_real)
            
            # BACKPROPAGATION
            loss.backward() 
            total_loss += loss.item()

        # Velocity update
        optimizer.step()

        # Clamping the velocity model to physical bounds
        with torch.no_grad():
            model_vp.clamp_(min=v_min, max=v_max)

        print(f"Iter {iter+1:03d} | Loss: {total_loss:.6e}")

        # Save and visualize intermediate results every 10 iterations
        if (iter + 1) % 10 == 0:
            plt.imshow(model_vp.detach().cpu().numpy(), cmap='jet', aspect='auto')
            plt.title(f"Iterazione {iter+1}")
            plt.colorbar()
            plt.savefig(f"./data/fwi/step_{iter+1}.png")
            plt.close()

    # FINAL SAVING
    np.savez("./data/fwi/fwi_final_deepwave.npz", vp=model_vp.detach().cpu().numpy().T)

if __name__ == "__main__":
    run_deepwave_fwi()