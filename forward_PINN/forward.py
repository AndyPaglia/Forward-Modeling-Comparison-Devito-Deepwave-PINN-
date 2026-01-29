"""
Compute the forward modeling in a 2D domain using PyTorch

Image and Sound Processing Lab - Politecnico di Milano

Daniele Ugo Leonzio
Paolo Bestagini
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_utils import absorbing_boundaries, forward, set_gpu, generate_pml_coefficients_2d

os.nice(10)

torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    """
    Main function to compute the forward modeling

    :return: None
    """
    # Set device
    set_gpu(-1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup parameters
    vp_model_path = args.vp_model_path
    dt = args.dt
    tn = args.tn
    nbl = args.nbl

    # Geometry parameters
    rec_spacing = args.rec_spacing / 1000
    rec_depth = args.rec_depth / 1000
    src_spacing = args.src_spacing / 1000
    src_depth = args.src_depth / 1000

    # Load the velocity model
    npzfile = np.load(vp_model_path)
    vp = npzfile["vp"].T #* 1000
    spacing = npzfile["spacing"] / 1000  # Convert to km
    dh = spacing[0]
    domain = np.array(vp.shape)
    vp = np.pad(vp, ((nbl, nbl), (nbl, nbl)), mode='edge')
    domain_pad = np.array(vp.shape)

    # Check conditions related to the spacing and the involved frequencies
    f_max = 1 / (2 * (dt / 1000))  # Nyquist frequency
    delta_t = 0.5 * dh / np.max(vp)

    if dt/1000 > delta_t:
        print(f"Time step is too large for the CFL condition. The maximum time step is {delta_t}")
        #return None

    # Check for the maximum frequency in the model
    f_max_sp = (np.min(vp) / (2*dh))
    f_max = np.min([f_max, f_max_sp])
    print(f"Maximum frequency in the model: {f_max} Hz")

    # Plot the velocity model
    if args.plot:
        plt.figure()
        #plt.imshow(vp[nbl:-nbl, nbl:-nbl], cmap='RdBu_r', clim=[1000, 4500])
        plt.imshow(vp[nbl:-nbl, nbl:-nbl], cmap='RdBu_r', clim=[1, 4.5])
        plt.title('Velocity model')
        plt.show()

    # Source coordinates
    domain_size = (domain - 1) * dh
    n_src = int(np.ceil(domain_size[1]/src_spacing))  # Fit as many shots as possible
    src_coordinates = np.empty((n_src, 2))
    src_coordinates[:, 0] = np.arange(0, src_spacing * n_src, src_spacing)
    src_coordinates[:, 1] = src_depth
    sources = np.asarray(src_coordinates // spacing, dtype=int) + [nbl, nbl]
    if True:  # vp_model_path == "./data/v_models/marmousi_paper_sp15.npz":
        src_coordinates = src_coordinates[1:-1, :]  # Remove first and last source as per paper conditions
        sources = sources[1:-1, :]  # Remove first and last source as per paper conditions

    # Receiver coordinates
    recz = int(nbl + rec_depth/spacing[0])
    n_rec = int(np.floor(domain_size[1]/rec_spacing)) + 1  # Fit as many receivers as possible
    rec_coordinates = np.empty((n_rec, 2))
    rec_coordinates[:, 0] = np.arange(0, n_rec) * rec_spacing
    rec_coordinates[:, 1] = rec_depth
    recs = np.asarray(rec_coordinates // spacing, dtype=int) + [nbl, nbl]

    # Define wavelet
    f0 = args.f0 * 1000  # source peak frequency [Hz]
    delay = 1.5 / f0  # delay [s]
    t = np.arange(0, tn/1000, dt/1000) - delay
    r = (1 - 2 * (np.pi * f0 * t) ** 2) * np.exp(-(np.pi * f0 * t) ** 2)

    # Plot the resampled wavelet
    if args.plot:
        plt.figure()
        plt.plot(r)
        plt.title('Resampled wavelet')
        plt.xlabel('Time [samples]')
        plt.show()

        f_axis = np.arange(0, len(r) + 1, 1) / (len(r) * (dt / 1000))
        f_axis = f_axis[:len(r)]
        r_fft = np.fft.fft(r)
        r_fft = np.abs(r_fft)
        plt.figure()
        plt.plot(f_axis[:len(f_axis)//8], r_fft[:len(f_axis)//8])
        plt.title('FFT of the wavelet')
        plt.xlabel('Frequency [Hz]')
        plt.show()

    # Define boundary mask
    #pmlc = absorbing_boundaries(vp.shape[1], vp.shape[0], nbl, 0.00005)
    pmlc = generate_pml_coefficients_2d(vp.shape, nbl)

    # Plot the pml coefficients
    if args.plot:
        plt.figure()
        plt.imshow(pmlc, cmap='jet')
        plt.title('Boundary mask')
        plt.colorbar()
        plt.show()

    # Move variables to the device
    wave = torch.from_numpy(r).to(dev)
    vp = torch.from_numpy(vp).to(dev)
    #pmlc = torch.from_numpy(pmlc).to(dev, dtype=torch.float32)

    # Split sources into batches of args.batch_size sources per batch (needed to fit them in the VRAM)
    batch_size = args.batch_size
    src_batch_list = [sources[i:i + batch_size] for i in range(0, len(sources), batch_size)]

    # Generate the observed data for all the sources
    d_obs_list = []
    for src_batch in src_batch_list:

        # Prepare arguments for the forward modeling
        kwargs = dict(wave=wave, src_list=np.array(src_batch), domain=domain_pad, dt=dt/1000, h=dh, dev=dev, recz=recz,
                      b=pmlc, pmln=nbl)

        # Run the forward modeling
        with torch.no_grad():
            d_obs_batch = forward(c=vp, **kwargs)

        # Detach the data and move it to the CPU
        d_obs_batch = d_obs_batch.detach().cpu().numpy()
        d_obs_list.append(d_obs_batch)

    # Concatenate the observed data into a single array
    d_obs_list = np.concatenate(d_obs_list, axis=0)

    # Select traces at the receiver locations
    rec_x = np.array([recs[i][0] - nbl for i in range(len(recs))])
    d_obs_list = d_obs_list[:, :, rec_x]

    # Plot the observed data
    if args.plot:
        plt.figure(figsize=(12, 7))
        for s, shot_idx in enumerate([1, 3, 5]):
            plt.subplot(1, 3, s+1)
            plt.imshow(d_obs_list[shot_idx, :, :], cmap='RdBu_r', aspect=5, clim=[-0.2, 0.2],
                                  extent=[0, d_obs_list.shape[2] * spacing[0], d_obs_list.shape[1] * dt/1000, 0])
            plt.title('Shot')
            plt.xlabel('Distance [km]')
            plt.ylabel('Time [s]')
        plt.show()

    # Plot the f-k spectrum of one shot
    if args.plot:
        shot = d_obs_list[0, :, :]
        spec = (np.fft.fft2(shot))
        plt.imshow(np.fft.fftshift((np.abs(spec)), axes=1), aspect='auto',
                   extent=[-1 / (2 * 12.5), 1 / (2 * 12.5), 1 / dt * 1000, 0], cmap='viridis')
        plt.yticks()
        plt.ylim(60, 0)
        plt.xlim(-1 / (4 * 12.5), 1 / (4 * 12.5))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Wavenumber [1/m]')
        plt.locator_params(axis='x', nbins=5)
        plt.grid(False)
        plt.show()

    # Plot the spectrum of one trace
    if args.plot:
        f_axis = np.arange(0, d_obs_list.shape[1]+1, 1) / (d_obs_list.shape[1] * (dt/1000))
        f_axis = f_axis[:d_obs_list.shape[1]]
        d_obs_fft = np.fft.fft(d_obs_list[0, :, 100])
        d_obs_fft = np.abs(d_obs_fft)
        plt.plot(f_axis[:len(f_axis)//16], d_obs_fft[:len(f_axis)//16])
        plt.show()

    # Save shots
    out_path = os.path.join("../../data", "shots", os.path.basename(vp_model_path))
    if not os.path.isdir(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    np.savez(out_path, d_obs_list=d_obs_list, src_coordinates=src_coordinates,
             rec_coordinates=rec_coordinates, t0=0, tn=tn, dt=dt, nbl=nbl, spacing=spacing,
             wave=wave.detach().cpu().numpy(), domain_pad=domain_pad, pmlc=pmlc.detach().cpu().numpy(), domain=domain,
             allow_pickle=True)

    return None


if __name__ == "__main__":
    """
    Run the main function
    """

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Forward with RNN')
    parser.add_argument('--vp_model_path', type=str, default="./data/v_models/overthrust_paper_sp15.npz",
                        help='Path to the true velocity model')
    parser.add_argument('--src_spacing', type=int, default=300, help='Spacing between shots')
    parser.add_argument('--rec_spacing', type=int, default=15, help='Spacing between receivers')
    parser.add_argument('--rec_depth', type=int, default=0, help='Depth of the receivers')
    parser.add_argument('--src_depth', type=int, default=30, help='Depth of the shots')
    parser.add_argument('--f0', type=float, default=0.008, help='Dominant frequency of the source wavelet')
    parser.add_argument('--tn', type=float, default=1900, help='Recording end time in ms')
    parser.add_argument('--dt', type=float, default=1.9, help='Time step in ms')
    parser.add_argument('--nbl', type=int, default=100, help='Number of boundary layers')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of sources per batch')
    parser.add_argument('--plot', type=bool, default=True, help='Show additional plots')
    args = parser.parse_args()

    # Run the main function
    main(args)

'''

"""
Compute the forward modeling in a 2D domain using PyTorch (RNN-based)
Aligned with Devito parameters and NORMALIZED for amplitude comparison

Image and Sound Processing Lab - Politecnico di Milano
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from pinn_utils import set_gpu, generate_pml_coefficients_2d, forward

os.nice(10)

# Ottimizzazioni per GPU
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True

def main(args):
    # 1. Setup Device
    set_gpu(-1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Caricamento Modello e Parametri
    vp_model_path = args.vp_model_path
    dt = args.dt         
    tn = args.tn         
    nbl = args.nbl       

    npzfile = np.load(vp_model_path)
    vp = npzfile["vp"].T  
    spacing = npzfile["spacing"] / 1000  
    dh = spacing[0]
    domain = np.array(vp.shape)
    
    vp_padded = np.pad(vp, ((nbl, nbl), (nbl, nbl)), mode='edge')
    domain_pad = np.array(vp_padded.shape)

    # 3. Geometria di Acquisizione
    rec_spacing = args.rec_spacing / 1000  
    rec_depth = args.rec_depth / 1000      
    src_spacing = args.src_spacing / 1000  
    src_depth = args.src_depth / 1000      

    domain_size_km = (domain - 1) * dh
    n_src = int(np.ceil(domain_size_km[1]/src_spacing))
    src_coordinates = np.empty((n_src, 2))
    src_coordinates[:, 0] = np.arange(0, src_spacing * n_src, src_spacing)
    src_coordinates[:, 1] = src_depth
    sources = np.asarray(src_coordinates // spacing, dtype=int) + [nbl, nbl]

    recz_idx = int(nbl + rec_depth/spacing[0])
    n_rec = int(np.floor(domain_size_km[1]/rec_spacing)) + 1
    rec_coordinates = np.empty((n_rec, 2))
    rec_coordinates[:, 0] = np.arange(0, n_rec) * rec_spacing
    rec_coordinates[:, 1] = rec_depth
    recs = np.asarray(rec_coordinates // spacing, dtype=int) + [nbl, nbl]

    # 4. Definizione Wavelet e SCALING
    f0_hz = args.f0 * 1000  
    delay = 1.5 / f0_hz     
    t = np.arange(0, tn/1000, dt/1000) - delay
    
    # Calcolo Ricker base
    r = (1 - 2 * (np.pi * f0_hz * t) ** 2) * np.exp(-(np.pi * f0_hz * t) ** 2)
    
    # --- SCALATURA PER NORMALIZZAZIONE ---
    # Moltiplichiamo per un fattore (es. 50) per pareggiare la scala clim [-5, 5]
    scaling_factor = 50.0 
    r = r * scaling_factor
    # -------------------------------------
    
    wave = torch.from_numpy(r).to(dev).float()
    vp_tensor = torch.from_numpy(vp_padded).to(dev).float()

    # 5. Boundary Mask (PML)
    pmlc = generate_pml_coefficients_2d(vp_padded.shape, nbl)

    # 6. Simulazione
    batch_size = args.batch_size
    src_batch_list = [sources[i:i + batch_size] for i in range(0, len(sources), batch_size)]

    d_obs_list = []
    print(f"Inizio modeling su {len(sources)} sorgenti (Scalate)...")
    for src_batch in src_batch_list:
        kwargs = dict(wave=wave, src_list=np.array(src_batch), domain=domain_pad, 
                      dt=dt/1000, h=dh, dev=dev, recz=recz_idx, b=pmlc, pmln=nbl)

        with torch.no_grad():
            d_obs_batch = forward(c=vp_tensor, **kwargs)
        
        d_obs_list.append(d_obs_batch.detach().cpu().numpy())

    d_obs_all = np.concatenate(d_obs_list, axis=0)
    rec_x_indices = np.array([recs[i][0] - nbl for i in range(len(recs))])
    d_obs_final = d_obs_all[:, :, rec_x_indices]

    # 7. Plot e Salvataggio (Visualizzazione identica a Devito)
    if args.plot:
        shot_idx = len(d_obs_final) // 2
        plt.figure(figsize=(10, 8))
        
        # clim=[-5, 5] ora mostrerà il segnale grazie allo scaling_factor
        plt.imshow(d_obs_final[shot_idx, :, :], cmap='gray', aspect='auto', clim=[-5, 5],
                   extent=[0, d_obs_final.shape[2] * args.rec_spacing, tn, 0])
        
        plt.title(f'Shot Centrale (PINN/RNN) - Sorgente a {src_coordinates[shot_idx, 0]:.2f} km (Scalato)')
        plt.xlabel('Ricevitori [m]')
        plt.ylabel('Tempo [ms]')
        
        plot_path = os.path.join("data", "shots", "marmousi_sp25_pinn_scaled.png")
        if not os.path.exists(os.path.dirname(plot_path)):
            os.makedirs(os.path.dirname(plot_path))
            
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Grafico salvato in: {plot_path}")
        plt.show()

    # 8. Salvataggio Dati
    out_data_path = os.path.join("data", "shots", "pinn_scaled_" + os.path.basename(vp_model_path))
    np.savez(out_data_path, d_obs_list=d_obs_final, src_coordinates=src_coordinates,
             rec_coordinates=rec_coordinates, dt=dt, tn=tn, f0=args.f0, allow_pickle=True)
    
    print(f"Modeling completato con successo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Forward Modeling PyTorch Aligned & Scaled')
    parser.add_argument('--vp_model_path', type=str, default="./data/v_models/marmousi_sp25.npz")
    parser.add_argument('--src_spacing', type=int, default=100)
    parser.add_argument('--rec_spacing', type=int, default=25)
    parser.add_argument('--rec_depth', type=int, default=20)
    parser.add_argument('--src_depth', type=int, default=20)
    parser.add_argument('--f0', type=float, default=0.005) 
    parser.add_argument('--tn', type=float, default=6000)
    parser.add_argument('--dt', type=float, default=3.0)
    parser.add_argument('--nbl', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--plot', type=bool, default=True)
    
    args = parser.parse_args()
    main(args)

    '''