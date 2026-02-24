"""
Full Waveform Inversion con Deepwave (forward) + SIREN PINN 

Architettura:
  - La SIREN mappa coordinate (x,z) normalizzate in [-1,1] -> velocità
  - Deepwave esegue il forward modeling e calcola il gradiente rispetto a model_vp
  - Il gradiente fluisce da model_vp -> SIREN weights

"""

import os
import shutil
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import deepwave
from deepwave import scalar
from tqdm import tqdm


# DEFINIZIONE DELLA SIREN - Physics-Informed Neural Network


class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Inizializzazione uniforme per il primo layer
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # Inizializzazione per layer nascosti
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    """
    Mappa coordinate 2D normalizzate in [-1,1] -> valore di velocità scalare.
    Architettura: [in=2] -> [SineLayer x hidden_layers] -> [Linear out=1]
    """
    def __init__(self, in_features=2, hidden_features=128, hidden_layers=4,
                 out_features=1, first_omega_0=30, hidden_omega_0=30.,
                 domain_shape=None, pretrained=None):
        super().__init__()
        self.domain_shape = domain_shape  # (nz, nx) - shape del modello di velocità

        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                                    is_first=False, omega_0=hidden_omega_0))
        # Layer finale lineare 
        final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / hidden_omega_0
            final.weight.uniform_(-bound, bound)
        layers.append(final)

        self.net = nn.Sequential(*layers)

        # shape: (nz*nx, 2)
        # Questa griglia è fissa durante tutto il training
        self.coords = self._generate_mesh(domain_shape)

        # Carica pesi pre-addestrati se disponibili
        if pretrained and os.path.exists(pretrained):
            self.load_state_dict(torch.load(pretrained, weights_only=True))
            print(f"   Loaded pretrained SIREN from: {pretrained}")
        else:
            if pretrained:
                print(f"   Pretrained model not found: '{pretrained}'. Using random init.")
            else:
                print("   No pretrained model specified. Using random initialization.")

    def _generate_mesh(self, shape):
        """
        Genera una griglia di coordinate 2D normalizzate in [-1, 1].
        Args:
            shape: (nz, nx) - dimensioni del dominio
        Returns:
            coords: tensor di shape (nz*nx, 2) con coordinate normalizzate
        """
        nz, nx = shape
        z_coords = torch.linspace(-1, 1, steps=nz)
        x_coords = torch.linspace(-1, 1, steps=nx)
        # meshgrid indicizzato 'ij': prima dimensione = z, seconda = x
        grid_z, grid_x = torch.meshgrid(z_coords, x_coords, indexing='ij')
        # Appiattisce e concatena: ogni riga è una coppia (z_norm, x_norm)
        coords = torch.stack([grid_z.flatten(), grid_x.flatten()], dim=-1)
        return coords

    def forward(self, coords):
        """
        Genera il modello di velocità dato le coordinate.
        Args:
            coords: tensor (N, 2) di coordinate normalizzate
        Returns:
            output: tensor (nz, nx) - mappa di velocità normalizzata in output SIREN
            coords: le coordinate (needed per gradienti rispetto agli input se usato come PINN)
        """
        # Detach + requires_grad=True permette di calcolare derivate rispetto alle coordinate
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords).view(self.domain_shape)
        return output, coords


# FUNZIONI DI SUPPORTO

def load_observed_data(data_path):
    """Carica i dati osservati dal forward modeling."""
    print(f"   Loading observed data from: {data_path}")
    data = np.load(data_path)
    
    d_obs      = data['receiver_data']     # (n_shots, n_rec_max, nt)
    src_coords = data['src_coordinates']   # (n_shots, 2) in metri
    rec_coords = data['rec_coordinates']   # (n_rec, 2) in metri
    dx, dz     = data['spacing']           # grid spacing in metri
    dt_sec     = float(data['dt']) / 1000.0
    f0_hz      = float(data['f0'])
    aperture   = float(data['aperture'])

    print(f"   Shots: {len(src_coords)} | Receivers: {len(rec_coords)} | "
          f"Time samples: {d_obs.shape[2]} | f0: {f0_hz} Hz")
    return d_obs, src_coords, rec_coords, dx, dz, dt_sec, f0_hz, aperture


def load_initial_model(model_path, smoothing_sigma=0):
    """
    Carica il modello di velocità iniziale.
    Lo smoothing crea il modello iniziale "smussato" usato dalla FWI.
    """
    print(f"   Loading initial model from: {model_path}")
    v_init = np.load(model_path)["vp"]
    if v_init.max() < 10:
        print("   Converting km/s -> m/s")
        v_init *= 1000.0
    if smoothing_sigma > 0:
        v_init = gaussian_filter(v_init, sigma=smoothing_sigma)
    print(f"   Shape: {v_init.shape} | Range: [{v_init.min():.0f}, {v_init.max():.0f}] m/s")
    return v_init


def pretrain_siren(siren, v_true_np, domain_shape, device,
                   n_epochs=1000, lr=1e-4, v_min=1500., v_max=4500.,
                   output_dir=None):
    """
    Pre-addestra la SIREN sul modello di velocità iniziale (smussato).
    
    Obiettivo: la SIREN apprende a rappresentare il modello iniziale prima
    di iniziare la FWI. Questo è equivalente al pre-training della PINN e di Devito+PINN (pretrain_siren.py)
    
    Normalizzazione:
      - Input SIREN: coordinate in [-1, 1]
      - Output SIREN: velocità normalizzata (vp_norm = (vp - mean) / std)
      - mean=3000, std=1000 -> range approssimativo [−1.5, 1.5] per Marmousi
    
    Args:
        siren: modello SIREN da addestrare
        v_true_np: array numpy (nx, nz) del modello di velocità target
        domain_shape: (nz, nx)
        device: cuda/cpu
        n_epochs: epoche di pre-training
        lr: learning rate
        v_min, v_max: bounds per il clipping
        output_dir: se specificato, salva i plot di pre-training
    """
    print(f"\n{'='*60}")
    print(f" PRE-TRAINING SIREN ({n_epochs} epochs)")
    print(f"{'='*60}")

    # Normalizzazione del target: porta il modello in un range ~[-1.5, 1.5]
    # che è compatibile con l'output della SIREN (senza attivazione finale)
    MEAN_VP = 3000.0   # m/s - media tipica per modelli crostali
    STD_VP  = 1000.0   # m/s - deviazione standard tipica

    # v_true ha shape (nx, nz) -> serve (nz, nx) per deepwave (nz, nx)
    # Ricordarsi che il modello è salvato come (nx, nz), deepwave vuole (nz, nx)
    v_nz_nx = v_true_np.T  # (nz, nx)
    v_normalized = (v_nz_nx - MEAN_VP) / STD_VP

    # Converte in tensore e sposta su device
    v_target = torch.from_numpy(v_normalized).float().to(device)
    coords   = siren.coords.to(device)

    # Optimizer per il pre-training
    opt   = torch.optim.AdamW(siren.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=50, factor=0.5)
    loss_fn = nn.MSELoss()

    pretrain_losses = []
    for epoch in tqdm(range(n_epochs), desc="Pre-training"):
        opt.zero_grad()

        # Forward pass SIREN: genera il modello normalizzato
        vp_pred, _ = siren(coords)

        # Loss: differenza tra predizione SIREN e target normalizzato
        loss = loss_fn(vp_pred, v_target)
        loss.backward()
        opt.step()
        sched.step(loss)

        pretrain_losses.append(loss.item())

        if epoch % 200 == 0:
            tqdm.write(f"  Epoch {epoch:4d} | Loss: {loss.item():.6e} | "
                       f"LR: {opt.param_groups[0]['lr']:.2e}")

    # Salva il modello pre-addestrato
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pretrain_path = os.path.join(output_dir, 'siren_pretrained.pth')
        torch.save(siren.state_dict(), pretrain_path)
        print(f"   Pretrained SIREN saved to: {pretrain_path}")

        # Plot pre-training
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        with torch.no_grad():
            vp_pred_np, _ = siren(coords)
            vp_pred_denorm = (vp_pred_np * STD_VP + MEAN_VP).cpu().numpy()

        axes[0].imshow(v_nz_nx, cmap='jet', aspect='auto', vmin=v_min, vmax=v_max)
        axes[0].set_title('Target (initial model)')
        axes[1].imshow(vp_pred_denorm, cmap='jet', aspect='auto', vmin=v_min, vmax=v_max)
        axes[1].set_title('SIREN output after pre-training')
        axes[2].plot(pretrain_losses)
        axes[2].set_yscale('log')
        axes[2].set_title('Pre-training loss')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pretrain_result.png'), dpi=150)
        plt.close()

    print(f"   Pre-training complete. Final loss: {pretrain_losses[-1]:.6e}")
    return MEAN_VP, STD_VP


def compute_shot_geometry(shot_idx, src_coords, rec_coords, aperture, dx, dz, device):
    """
    Calcola le posizioni di sorgente e ricevitori per uno shot.
    Applica la maschera di apertura: include solo i ricevitori entro 
    +- apertura metri dalla sorgente.
    
    Returns:
        src_loc: tensor (1, 1, 2) con indici griglia [iz, ix] della sorgente
        rec_locs: tensor (1, n_rec_in_aperture, 2) con indici griglia dei ricevitori
        n_rec: numero di ricevitori attivi
        mask: array booleano per selezionare le tracce corrispondenti in d_obs
    """
    curr_src_x = src_coords[shot_idx, 0]

    # Sorgente: converti coordinate fisiche -> indici griglia
    src_loc = torch.tensor(
        [[[src_coords[shot_idx, 1] / dz, curr_src_x / dx]]]
    ).round().long().to(device)

    # Ricevitori: applica apertura
    mask = ((rec_coords[:, 0] >= curr_src_x - aperture) &
            (rec_coords[:, 0] <= curr_src_x + aperture))
    curr_rec = rec_coords[mask]

    rec_locs = torch.zeros(1, len(curr_rec), 2, device=device)
    rec_locs[0, :, 0] = torch.from_numpy(curr_rec[:, 1] / dz)
    rec_locs[0, :, 1] = torch.from_numpy(curr_rec[:, 0] / dx)
    rec_locs = rec_locs.round().long()

    return src_loc, rec_locs, len(curr_rec), mask


# FWI LOOP PRINCIPALE

def run_fwi_deepwave_siren(
        observed_data_path, initial_model_path, output_dir,
        overwrite=False,
        siren_pretrained_path=None,
        n_iterations=500,
        lr_siren=1e-4,
        batch_size=5,
        v_min=1500., v_max=4500.,
        smoothing_sigma=10.,
        nbl=200, accuracy=4,
        pretrain_epochs=1000,
        pretrain_lr=1e-4,
        save_every=10, plot_every=10,
        mean_vp=3000., std_vp=1000.
):
    """
    FWI con Deepwave come forward engine e SIREN come parametrizzazione del modello.

    FLUSSO COMPUTAZIONALE: coords (fisse) --> SIREN (coords) --> vp_norm (nz, nx) --> denormalizza (vp = vp_norm * std + mean) + 
    clamp (vp, v_min, v_max) --> deepwave.scalar(vp) --> d_syn --> loss = MSE(d_syn, d_obs) --> loss.backward() --> 
    gradiente fluisce da d_syn a vp a SIREN weights --> AdamW.step() aggiorna i pesi della SIRENù

      - In FWI diretta: optimizer aggiorna i pixel di velocità
      - Con SIREN: optimizer aggiorna i pesi della rete che genera i pixel
      - Effetto: la SIREN impone una regolarizzazione implicita (smoothness)
    """
    # Cancella output_dir se --overwrite è specificato
    if overwrite and os.path.exists(output_dir):
        print(f" Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f" FWI DEEPWAVE + SIREN PINN")
    print(f"{'='*70}")
    print(f" Device: {device}")

    # 1. CARICA DATI E MODELLO
    d_obs_np, src_coords, rec_coords, dx, dz, dt_sec, f0_hz, aperture = \
        load_observed_data(observed_data_path)

    v_init_np = load_initial_model(initial_model_path, smoothing_sigma)

    # Shape del dominio: deepwave usa (nz, nx), il modello è salvato come (nx, nz)
    nz, nx = v_init_np.T.shape  # v_init_np è (nx, nz), trasposto diventa (nz, nx)
    domain_shape = (nz, nx)
    print(f"   Domain shape (nz, nx): {domain_shape}")

    nt = d_obs_np.shape[2]
    n_shots = len(src_coords)

    # Wavelet di Ricker
    wavelet = deepwave.wavelets.ricker(f0_hz, nt, dt_sec, 1.0 / f0_hz).to(device)

    # Sposta i dati osservati su device
    # NOTA MEMORIA: se il dataset è molto grande, teniamolo in CPU
    # e fare .to(device) solo per il batch corrente
    d_obs = torch.from_numpy(d_obs_np).float()  # CPU - evita di saturare la VRAM (miglioramento per problema OUT OF MEMORY)


    # 2. INIZIALIZZA SIREN
    print(f"\n Initializing SIREN (hidden={128}, layers={4})...")
    siren = Siren(
        in_features=2,
        hidden_features=128,
        hidden_layers=4,
        out_features=1,
        first_omega_0=30,
        hidden_omega_0=30.,
        domain_shape=domain_shape,
        pretrained=siren_pretrained_path
    ).to(device)

    coords = siren.coords.to(device)


    # 3. PRE-TRAINING (se non ci sono pesi pre-addestrati)
    if siren_pretrained_path is None or not os.path.exists(siren_pretrained_path):
        mean_vp, std_vp = pretrain_siren(
            siren, v_init_np, domain_shape, device,
            n_epochs=pretrain_epochs,
            lr=pretrain_lr,
            v_min=v_min, v_max=v_max,
            output_dir=output_dir
        )
    else:
        # Se si caricano pesi pre-addestrati, usa i parametri di normalizzazione forniti come argomenti
        print(f"   Using pretrained SIREN. Normalization: mean={mean_vp}, std={std_vp}")


    # 4. SETUP OPTIMIZER FWI
    # AdamW ottimizza i PESI della SIREN (non i valori di velocità direttamente)
    # Il weight decay è giusto se usato qui perché agisce sui pesi della rete,
    # non sui valori fisici di velocità
    optimizer = torch.optim.AdamW(siren.parameters(), lr=lr_siren, weight_decay=1e-4)

    # Cosine Annealing: riduce gradualmente il lr durante la FWI
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iterations, eta_min=lr_siren * 0.01
    )

    # Deepwave alloca buffer float32 interni per i wavefields.
    # float16 causa instabilità numerica nella propagazione d'onda.
    scaler = None

    loss_fn = nn.MSELoss()


    # 5. FWI LOOP
    print(f"\n{'='*70}")
    print(f" FWI LOOP")
    print(f"   Iterations: {n_iterations} | LR: {lr_siren} | Batch: {batch_size} shots")
    print(f"{'='*70}\n")

    loss_history      = []
    grad_norm_history = []

    start_time = time.time()

    for iteration in range(n_iterations):
        optimizer.zero_grad()
        torch.cuda.empty_cache()  # Libera cache prima di ogni iterazione

        # Seleziona shots casuali per questo step
        shot_indices = np.random.choice(n_shots, min(batch_size, n_shots), replace=False)

        iter_loss = 0.0

        # DIAGNOSTICA iter 1: confronto syn vs obs per verificare compatibilità dati
        if iteration == 0:
            with torch.no_grad():
                vp_diag, _ = siren(coords)
                vp_diag = (vp_diag * std_vp + mean_vp).clamp(v_min, v_max)
                # Usa shot centrale per diagnostica
                diag_idx = n_shots // 2
                src_diag, rec_diag, nrec_diag, _ = compute_shot_geometry(
                    diag_idx, src_coords, rec_coords, aperture, dx, dz, device)
                out_diag = scalar(vp_diag, dx, dt_sec,
                    source_amplitudes=wavelet.view(1, 1, nt),
                    source_locations=src_diag,
                    receiver_locations=rec_diag,
                    pml_width=[nbl]*4, pml_freq=f0_hz, accuracy=accuracy)
                d_syn_diag = out_diag[-1][0].cpu().numpy()
                d_obs_diag = d_obs[diag_idx, :nrec_diag, :].numpy()

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            vmax = np.percentile(np.abs(d_obs_diag), 98)
            axes[0].imshow(d_obs_diag, aspect='auto', cmap='seismic',
                          vmin=-vmax, vmax=vmax)
            axes[0].set_title('Observed (target)')
            vmax_syn = np.percentile(np.abs(d_syn_diag), 98)
            axes[1].imshow(d_syn_diag, aspect='auto', cmap='seismic',
                          vmin=-vmax_syn, vmax=vmax_syn)
            axes[1].set_title('Synthetic (from initial model)')
            # Differenza normalizzata
            diff = d_syn_diag / (np.abs(d_syn_diag).max() + 1e-8) -                    d_obs_diag  / (np.abs(d_obs_diag).max()  + 1e-8)
            axes[2].imshow(diff, aspect='auto', cmap='seismic',
                          vmin=-1, vmax=1)
            axes[2].set_title(f'Residual (normalized) | max_diff={np.abs(diff).max():.3f}')
            plt.tight_layout()
            diag_dir = os.path.join(output_dir, 'diagnostics')
            os.makedirs(diag_dir, exist_ok=True)
            plt.savefig(os.path.join(diag_dir, 'shot_comparison_iter0.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Diagnostic plot saved. Syn range: [{d_syn_diag.min():.3e}, {d_syn_diag.max():.3e}]")
            print(f"   Obs range:  [{d_obs_diag.min():.3e}, {d_obs_diag.max():.3e}]")
            del vp_diag, out_diag, d_syn_diag
            torch.cuda.empty_cache()

        # STEP A+B: per ogni shot, rigenera vp dalla SIREN e fai forward
        # IMPORTANTE: vp viene rigenerato ad ogni shot perché .backward()
        # libera il grafo computazionale. 

        for shot_idx in shot_indices:

            # Genera vp dalla SIREN - nuovo grafo per ogni shot
            vp_norm, _ = siren(coords)                          # (nz, nx)
            vp = vp_norm * std_vp + mean_vp                     # denormalizza
            vp = torch.clamp(vp, min=v_min, max=v_max)          # vincola

            src_loc, rec_locs, n_rec, mask = compute_shot_geometry(
                shot_idx, src_coords, rec_coords, aperture, dx, dz, device
            )

            # Forward modeling Deepwave
            # scalar() è differenziabile rispetto a vp grazie all'autograd PyTorch
            # Il gradiente d_loss/d_vp viene calcolato automaticamente
            out = scalar(
                vp,                                      # (nz, nx) - dalla SIREN
                dx,                                      # grid spacing
                dt_sec,                                  # time step
                source_amplitudes=wavelet.view(1, 1, nt),
                source_locations=src_loc,
                receiver_locations=rec_locs,
                pml_width=[nbl] * 4,
                pml_freq=f0_hz,
                accuracy=accuracy
            )

            # d_syn: dati sintetici per questo shot
            d_syn = out[-1][0]          # (n_rec, nt)

            # Dati osservati corrispondenti (stesso shot, stessi ricevitori)
            d_real = d_obs[shot_idx, :n_rec, :].to(device)  # CPU -> GPU solo per questo shot

            # Normalizza i dati per la loss: porta entrambi a scala unitaria
            # Questo evita che differenze di ampiezza assoluta dominino i gradienti
            d_syn_norm  = d_syn  / (d_syn.abs().max()  + 1e-8)
            d_real_norm = d_real / (d_real.abs().max() + 1e-8)

            # Loss MSE sui dati normalizzati
            shot_loss = loss_fn(d_syn_norm, d_real_norm)
            
            # Divide per batch_size per avere una loss media sul batch
            shot_loss = shot_loss / len(shot_indices)

            # STEP C: backward pass
            # Gradiente: d_loss/d_d_syn -> d_loss/d_vp (Deepwave autograd)
            #            -> d_loss/d_vp_norm (clamp + denorm, chain rule)
            #            -> d_loss/d_siren_weights (SIREN backprop)
            # Backward: libera il grafo e aggiorna i gradienti SIREN
            shot_loss.backward()
            iter_loss += shot_loss.item()

            # Libera esplicitamente i tensori del grafo per questo shot
            # prima che Deepwave allochi i buffer per il prossimo
            del out, d_syn, d_syn_norm, d_real_norm, vp, vp_norm, shot_loss
            torch.cuda.empty_cache()


        # STEP D: aggiorna i pesi della SIREN
        # Gradient clipping: previene esplosione dei gradienti
        grad_norm = torch.nn.utils.clip_grad_norm_(siren.parameters(), max_norm=0.1)
        grad_norm_history.append(grad_norm.item())
        optimizer.step()

        scheduler.step()

        loss_history.append(iter_loss)


        # Logging e salvataggio
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"[{iteration+1:4d}/{n_iterations}] "
              f"Loss: {iter_loss:.6e} | "
              f"GradNorm: {grad_norm:.4e} | "
              f"LR: {curr_lr:.2e}")

        if (iteration + 1) % save_every == 0:
            # Recupera il modello di velocità corrente per salvarlo
            with torch.no_grad():
                vp_save, _ = siren(coords)
                vp_save = (vp_save * std_vp + mean_vp).clamp(v_min, v_max)
                vp_np = vp_save.cpu().numpy().T  # Torna a (nx, nz)

            models_dir = os.path.join(output_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            np.savez(
                os.path.join(models_dir, f'model_iter_{iteration+1:05d}.npz'),
                vp=vp_np, spacing=np.array([dx, dz])
            )
            torch.save(siren.state_dict(),
                       os.path.join(models_dir, f'siren_iter_{iteration+1:05d}.pth'))

        if (iteration + 1) % plot_every == 0:
            with torch.no_grad():
                vp_plot, _ = siren(coords)
                vp_plot = (vp_plot * std_vp + mean_vp).clamp(v_min, v_max)
                vp_plot_np = vp_plot.cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            im = axes[0].imshow(vp_plot_np, cmap='jet', aspect='auto',
                                vmin=v_min, vmax=v_max)
            axes[0].set_title(f'Inverted Model - Iter {iteration+1}')
            axes[0].set_xlabel('X (grid points)')
            axes[0].set_ylabel('Z (grid points)')
            plt.colorbar(im, ax=axes[0], label='Velocity (m/s)')

            axes[1].plot(loss_history, 'b-', linewidth=2)
            axes[1].set_yscale('log')
            axes[1].set_title('FWI Loss')
            axes[1].set_xlabel('Iteration')
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(grad_norm_history, 'r-', linewidth=2)
            axes[2].set_yscale('log')
            axes[2].set_title('Gradient Norm (SIREN weights)')
            axes[2].set_xlabel('Iteration')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, f'fwi_iter_{iteration+1:05d}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    # 6. SALVO RISULTATI FINALI
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f" FWI COMPLETED in {total_time/60:.2f} minutes")
    print(f" Final loss: {loss_history[-1]:.6e}")
    print(f"{'='*70}")

    with torch.no_grad():
        vp_final, _ = siren(coords)
        vp_final = (vp_final * std_vp + mean_vp).clamp(v_min, v_max)
        vp_final_np = vp_final.cpu().numpy().T  # (nx, nz)

    np.savez(
        os.path.join(output_dir, 'fwi_final_results.npz'),
        vp=vp_final_np,
        spacing=np.array([dx, dz]),
        loss_history=loss_history,
        grad_norm_history=grad_norm_history
    )
    torch.save(siren.state_dict(), os.path.join(output_dir, 'siren_final.pth'))
    print(f" Results saved to: {output_dir}")


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description='FWI con Deepwave forward + SIREN PINN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument('--observed_data_path', type=str, required=True,
                        help='Path ai dati osservati (.npz)')
    parser.add_argument('--initial_model_path', type=str, required=True,
                        help='Path al modello di velocità iniziale (.npz)')
    parser.add_argument('--output_dir', type=str, default='./data/fwi_deepwave_siren',
                        help='Directory di output')
    parser.add_argument('--siren_pretrained_path', type=str, default=None,
                        help='Path a pesi SIREN pre-addestrati (.pth). '
                             'Se None, esegue il pre-training automaticamente.')

    # FWI parameters
    parser.add_argument('--n_iterations', type=int, default=500,
                        help='Numero di iterazioni FWI')
    parser.add_argument('--lr_siren', type=float, default=1e-5,
                        help='Learning rate AdamW per pesi SIREN')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Shots per mini-batch')
    parser.add_argument('--v_min', type=float, default=1500.0,
                        help='Velocità minima (m/s)')
    parser.add_argument('--v_max', type=float, default=4500.0,
                        help='Velocità massima (m/s)')
    parser.add_argument('--smoothing_sigma', type=float, default=10.0,
                        help='Smoothing Gaussiano per modello iniziale (0=no smoothing)')

    # Pre-training parameters
    parser.add_argument('--pretrain_epochs', type=int, default=1000,
                        help='Epoche di pre-training SIREN')
    parser.add_argument('--pretrain_lr', type=float, default=1e-4,
                        help='Learning rate pre-training')

    # Normalizzazione (coerente con il pre-training (se si caricano pesi))
    parser.add_argument('--mean_vp', type=float, default=3000.0,
                        help='Media per normalizzazione velocità (m/s)')
    parser.add_argument('--std_vp', type=float, default=1000.0,
                        help='Std per normalizzazione velocità (m/s)')

    # Numerical parameters
    parser.add_argument('--nbl', type=int, default=50,
                        help='Larghezza PML boundary')
    parser.add_argument('--accuracy', type=int, default=2, choices=[2, 4, 6, 8],
                        help='Ordine accuratezza spaziale')

    # Output
    parser.add_argument('--save_every', type=int, default=10,
                        help='Salva modello ogni N iterazioni')
    parser.add_argument('--overwrite', action='store_true',
                        help='Cancella output_dir esistente prima di iniziare')
    parser.add_argument('--plot_every', type=int, default=10,
                        help='Plotta ogni N iterazioni')

    args = parser.parse_args()

    run_fwi_deepwave_siren(
        observed_data_path=args.observed_data_path,
        initial_model_path=args.initial_model_path,
        output_dir=args.output_dir,
        siren_pretrained_path=args.siren_pretrained_path,
        n_iterations=args.n_iterations,
        lr_siren=args.lr_siren,
        batch_size=args.batch_size,
        v_min=args.v_min,
        v_max=args.v_max,
        smoothing_sigma=args.smoothing_sigma,
        nbl=args.nbl,
        accuracy=args.accuracy,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        save_every=args.save_every,
        plot_every=args.plot_every,
        mean_vp=args.mean_vp,
        std_vp=args.std_vp,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()