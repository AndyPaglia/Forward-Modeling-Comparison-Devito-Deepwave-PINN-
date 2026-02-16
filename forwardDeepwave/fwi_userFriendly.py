"""
Full Waveform Inversion (FWI) con Deepwave
"""

# IMPORTING LIBRARIES
import os
import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import deepwave
from deepwave import scalar
from tqdm import tqdm

# UTILITY FUNCTIONS FOR LOADING DATA, FORWARD MODELING, LOSS CALCULATION, SAVING MODELS, AND PLOTTING

# LOAD OBSERVED DATA FROM FORWARD MODELING
def load_observed_data(data_path):
    """CaricO dati osservati dal forward modeling"""
    print(f" Loading observed data from: {data_path}")
    
    data = np.load(data_path)
    d_obs = data['receiver_data']  # (n_shots, n_rec_max, nt)
    src_coords = data['src_coordinates']  # (n_shots, 2)
    rec_coords = data['rec_coordinates']  # (n_rec, 2)
    dx, dz = data['spacing']
    dt_sec = float(data['dt']) / 1000.0
    f0_hz = float(data['f0'])
    aperture = float(data['aperture'])
    
    print(f"   Shots: {len(src_coords)}")
    print(f"   Max receivers: {len(rec_coords)}")
    print(f"   Time samples: {d_obs.shape[2]}")
    print(f"   Aperture: {aperture} m\n")
    
    return d_obs, src_coords, rec_coords, dx, dz, dt_sec, f0_hz, aperture

# LOAD INITIAL VELOCITY MODEL (WE LOAD THE TRUE VELOCITY MODEL AND APPLY SMOOTHING TO CREATE THE INITIAL GUESS)
def load_initial_model(model_path, smoothing_sigma=0):
    """CaricO e preparO modello iniziale"""
    print(f" Loading initial model from: {model_path}")
    
    v_init = np.load(model_path)["vp"]
    
    # Converti km/s → m/s (se serve)
    if v_init.max() < 10:
        print("   Converting velocities from km/s to m/s")
        v_init *= 1000.0
    
    # Applica smoothing (SIGMA = 10 NEI READ ME DEI PROF)
    if smoothing_sigma > 0:
        print(f"   Applying Gaussian smoothing (σ={smoothing_sigma})")
        v_init = gaussian_filter(v_init, sigma=smoothing_sigma)
    
    print(f"   Model shape: {v_init.shape}")
    print(f"   Velocity range: [{v_init.min():.0f}, {v_init.max():.0f}] m/s\n")
    
    return v_init

# COMPUTE SHOT GEOMETRY (SOURCE AND RECEIVER LOCATIONS)
def compute_shot_geometry(shot_idx, src_coords, rec_coords, aperture, dx, dz, device):
    """Calcola posizioni sorgente e ricevitori per uno shot"""
    curr_src_x = src_coords[shot_idx, 0]
    curr_src_z = src_coords[shot_idx, 1]
    
    # Posizione sorgente
    src_loc = torch.tensor(
        [[[curr_src_z / dz, curr_src_x / dx]]]
    ).round().long().to(device)
    
    # Ricevitori considerando l'apertura
    mask = (rec_coords[:, 0] >= curr_src_x - aperture) & \
           (rec_coords[:, 0] <= curr_src_x + aperture)
    curr_rec_coords = rec_coords[mask]
    
    rec_locs = torch.zeros(1, len(curr_rec_coords), 2, device=device)
    rec_locs[0, :, 0] = torch.from_numpy(curr_rec_coords[:, 1] / dz)
    rec_locs[0, :, 1] = torch.from_numpy(curr_rec_coords[:, 0] / dx)
    rec_locs = rec_locs.round().long()
    
    return src_loc, rec_locs, len(curr_rec_coords)

# FORWARD MODELING + LOSS CALCULATION PER BATCH OF SHOTS
def forward_and_loss(model_vp, shot_indices, d_obs, src_coords, rec_coords, 
                     aperture, dx, dz, dt_sec, wavelet, f0_hz, nbl, accuracy, device):
    """Forward modeling + loss per un batch di shots"""
    batch_loss = 0.0
    nt = wavelet.shape[0]
    
    for shot_idx in shot_indices:
        # Geometria
        src_loc, rec_locs, n_rec = compute_shot_geometry(
            shot_idx, src_coords, rec_coords, aperture, dx, dz, device
        )
        
        # Forward modeling
        out = scalar(
            model_vp, dx, dt_sec,
            source_amplitudes=wavelet.view(1, 1, nt),
            source_locations=src_loc,
            receiver_locations=rec_locs,
            pml_width=[nbl] * 4,
            pml_freq=f0_hz,
            accuracy=accuracy
        )
        
        # Dati sintetici vs reali
        d_syn = out[-1][0]  # (n_rec, nt)
        d_real = d_obs[shot_idx, :n_rec, :]
        
        # Loss MSE + backward
        loss = torch.nn.functional.mse_loss(d_syn, d_real)
        loss.backward()
        batch_loss += loss.item()
    
    return batch_loss / len(shot_indices)

# SAVE CURRENT MODEL
def save_model(model_vp, iteration, output_dir, dx, dz):
    """Salva modello corrente"""
    v_current = model_vp.detach().cpu().numpy().T  # Torna a (nx, nz)
    
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    np.savez(
        os.path.join(output_dir, 'models', f'model_iter_{iteration:05d}.npz'),
        vp=v_current,
        spacing=np.array([dx, dz])
    )

# PLOT CURRENT MODEL AND METRICS
def plot_results(model_vp, iteration, loss_history, grad_norm_history, 
                 v_range_history, output_dir, v_min, v_max):
    """Plotta modello corrente e metriche"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    v_current = model_vp.detach().cpu().numpy()
    
    # Modello corrente
    im0 = axes[0, 0].imshow(v_current, cmap='jet', aspect='auto',
                            vmin=v_min, vmax=v_max)
    axes[0, 0].set_title(f'Velocity Model - Iteration {iteration}', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('X (grid points)')
    axes[0, 0].set_ylabel('Z (grid points)')
    plt.colorbar(im0, ax=axes[0, 0], label='Velocity (m/s)')
    
    # Loss history
    axes[0, 1].plot(loss_history, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss (MSE)')
    axes[0, 1].set_title('Loss History', fontsize=14, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gradient norm
    axes[1, 0].plot(grad_norm_history, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_title('Gradient Norm History', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Velocity range
    v_ranges = np.array(v_range_history)
    axes[1, 1].plot(v_ranges[:, 0], 'b-', label='V_min', linewidth=2)
    axes[1, 1].plot(v_ranges[:, 1], 'r-', label='V_max', linewidth=2)
    axes[1, 1].axhline(v_min, color='b', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(v_max, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Velocity (m/s)')
    axes[1, 1].set_title('Velocity Range Evolution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, 'plots', f'fwi_iter_{iteration:05d}.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()

# MAIN FWI FUNCTION
def run_fwi(observed_data_path, initial_model_path, output_dir,
            n_iterations=1000, learning_rate=10.0, batch_size=5,
            v_min=1500.0, v_max=4500.0, smoothing_sigma=10.0,
            nbl=200, accuracy=4, save_every=10, plot_every=10):
    """
    Esegue Full Waveform Inversion
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f" FWI START")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Carica dati
    d_obs_np, src_coords, rec_coords, dx, dz, dt_sec, f0_hz, aperture = \
        load_observed_data(observed_data_path)
    
    v_init_np = load_initial_model(initial_model_path, smoothing_sigma)
    
    # Converti a torch
    model_vp = torch.tensor(
        v_init_np.T,  # Deepwave usa (nz, nx)
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )
    
    d_obs = torch.from_numpy(d_obs_np).float().to(device)
    nt = d_obs.shape[2]
    
    # Genera wavelet
    wavelet = deepwave.wavelets.ricker(f0_hz, nt, dt_sec, 1/f0_hz).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam([model_vp], lr=learning_rate)
    
    # Metriche
    loss_history = []
    grad_norm_history = []
    v_range_history = []
    
    print(f"  Configuration:")
    print(f"   Iterations: {n_iterations}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Batch size: {batch_size} shots")
    print(f"   Velocity bounds: [{v_min}, {v_max}] m/s")
    print(f"   Output: {output_dir}\n")
    print(f"{'='*70}\n")
    
    # OPTIMIZATION LOOP
    n_shots = len(src_coords)
    start_time = time.time()
    
    for iteration in range(n_iterations):
        iter_start = time.time()
        optimizer.zero_grad()
        
        # Shuffle shots
        shot_indices = list(range(n_shots))
        np.random.shuffle(shot_indices)
        
        # Mini-batching
        total_loss = 0.0
        batch_iterator = range(0, n_shots, batch_size)
        pbar = tqdm(batch_iterator, desc=f"Iter {iteration+1:04d}", leave=False)
        
        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, n_shots)
            batch_shots = shot_indices[batch_start:batch_end]
            
            # Forward + loss + backward
            batch_loss = forward_and_loss(
                model_vp, batch_shots, d_obs, src_coords, rec_coords,
                aperture, dx, dz, dt_sec, wavelet, f0_hz, nbl, accuracy, device
            )
            
            total_loss += batch_loss * len(batch_shots)
            pbar.set_postfix({'loss': f'{batch_loss:.4e}'})
        
        avg_loss = total_loss / n_shots
        
        # Salva gradient norm PRIMA dell'update
        grad_norm = model_vp.grad.norm().item()
        
        # Update modello
        optimizer.step()
        
        # Applica constraints
        with torch.no_grad():
            model_vp.clamp_(min=v_min, max=v_max)
        
        # Salva metriche
        loss_history.append(avg_loss)
        grad_norm_history.append(grad_norm)
        v_current = model_vp.detach().cpu().numpy()
        v_range_history.append([v_current.min(), v_current.max()])
        
        iter_time = time.time() - iter_start
        v_curr_min, v_curr_max = v_range_history[-1]
        
        print(f"[{iteration+1:4d}/{n_iterations}] "
              f"Loss: {avg_loss:.6e} | "
              f"GradNorm: {grad_norm:.4e} | "
              f"V: [{v_curr_min:.0f}, {v_curr_max:.0f}] m/s | "
              f"Time: {iter_time:.2f}s")
        
        # Salva modello
        if (iteration + 1) % save_every == 0:
            save_model(model_vp, iteration + 1, output_dir, dx, dz)
        
        # Plotta
        if (iteration + 1) % plot_every == 0:
            plot_results(
                model_vp, iteration + 1, loss_history, grad_norm_history,
                v_range_history, output_dir, v_min, v_max
            )
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f" FWI COMPLETED")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Final loss: {loss_history[-1]:.6e}")
    print(f"{'='*70}\n")
    
    # Salva risultati finali
    os.makedirs(output_dir, exist_ok=True)
    v_final = model_vp.detach().cpu().numpy().T
    
    np.savez(
        os.path.join(output_dir, 'fwi_final_results.npz'),
        vp=v_final,
        spacing=np.array([dx, dz]),
        loss_history=loss_history,
        grad_norm_history=grad_norm_history,
        v_range_history=v_range_history
    )
    
    print(f" Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='FWI with Deepwave - Simple Version')
    
    # Paths
    parser.add_argument('--observed_data_path', type=str, required=True,
                       help='Path to observed data (.npz)')
    parser.add_argument('--initial_model_path', type=str, required=True,
                       help='Path to initial velocity model (.npz)')
    parser.add_argument('--output_dir', type=str, default='./data/fwi_results',
                       help='Output directory')
    
    # FWI parameters
    parser.add_argument('--n_iterations', type=int, default=1000,
                       help='Number of iterations')
    parser.add_argument('--learning_rate', type=float, default=10.0,
                       help='Learning rate (Adam)')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Shots per mini-batch')
    
    # Model constraints
    parser.add_argument('--v_min', type=float, default=1500.0,
                       help='Minimum velocity (m/s)')
    parser.add_argument('--v_max', type=float, default=4500.0,
                       help='Maximum velocity (m/s)')
    parser.add_argument('--smoothing_sigma', type=float, default=10.0,
                       help='Gaussian smoothing sigma (0 = no smoothing)')
    
    # Numerical parameters
    parser.add_argument('--nbl', type=int, default=200,
                       help='PML boundary width')
    parser.add_argument('--accuracy', type=int, default=4, choices=[2,4,6,8],
                       help='Spatial accuracy')
    
    # Output
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save model every N iterations')
    parser.add_argument('--plot_every', type=int, default=10,
                       help='Plot every N iterations')
    
    args = parser.parse_args()
    
    run_fwi(
        observed_data_path=args.observed_data_path,
        initial_model_path=args.initial_model_path,
        output_dir=args.output_dir,
        n_iterations=args.n_iterations,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        v_min=args.v_min,
        v_max=args.v_max,
        smoothing_sigma=args.smoothing_sigma,
        nbl=args.nbl,
        accuracy=args.accuracy,
        save_every=args.save_every,
        plot_every=args.plot_every
    )
    
    print(" Done!\n")


if __name__ == "__main__":
    main()