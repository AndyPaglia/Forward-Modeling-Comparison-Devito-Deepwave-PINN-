# IMPORTING LIBRARIES AND SETTING UP DEVICE
import matplotlib.pyplot as plt
import torch
import numpy as np
import deepwave
from deepwave import scalar

# DEVICE CONFIGURATION (IF AVAILABLE, USE GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOADING MODEL (.NPZ FILE)
data = np.load("./data/v_models/marmousi_sp25.npz")
v = torch.from_numpy(data['vp'] * 1000).float().to(device)
# Convertion to m/s (*1000) for Deepwave compatibility)
# Deepwave expects velocity in m/s and the provided model is in km/s.
# Deepwave uses (Z, X) indexing for 2D models. 
# The model has: length (X) equal to 481 while depth (Z) equal to 121. 

# Definition of the parameters to match Devito's setup.
dx = 25.0 # Spacing in meters
dt = 0.003 # Time step in seconds (3ms)
tn = 6.0 # Total time in seconds (6000ms)
nt = int(tn / dt) # Number of time steps
freq = 5.0 # Frequency of the Ricker wavelet. Deepwave uses Hz directly.
peak_time = 1.0 / freq # Peak time for Ricker wavelet

# ACQUISITION GEOMETRY
# X = 12000m (480 indices * 25m)
# Source spacing = 100m (every 4 indices).
# Receiver spacing = 25m.
n_shots = 120 # 120000m / 100m = 120 shots
source_depth = 1 # --src_depth 20m / 25m = 0.8 index - rounded to 1
n_receivers_per_shot = 481 # Receivers over the full model length
receiver_depth = 1 # --rec_depth 20m / 25m = 0.8 index - rounded to 1

# SOURCES COORDINATES SETUP
source_locations = torch.zeros(n_shots, 1, 2, device=device)
source_locations[..., 1] = source_depth
# One source every 4 indices (100m spacing)
source_locations[:, 0, 0] = torch.arange(n_shots) * 4

# RECEIVERS COORDINATES SETUP
receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2, device=device)
receiver_locations[..., 1] = receiver_depth
# One receiver every index (25m spacing)
receiver_locations[:, :, 0] = torch.arange(n_receivers_per_shot).repeat(n_shots, 1)

# RICKER WAVELET SOURCE AMPLITUDES
source_amplitudes = (
    deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    .repeat(n_shots, 1, 1)
    .to(device)
)

# WAVE PROPAGATION USING DEEPWAVE - FORWARD MODELING
out = scalar(
    v, dx, dt,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    accuracy=4,  # From 8 to 4 to match Devito's accuracy
    pml_freq=freq,
    pml_width=[100, 100, 100, 100]  # Increased PML width to 100 to match Devito and avoid reflections
)
# Results extraction
receiver_amplitudes = out[-1]

# CENTRAL SHOT PLOTTING
plt.figure(figsize=(10, 8))
shot_centrale = -1 * receiver_amplitudes[60].cpu().T # Sign inverted to match physical convention
n_rec = shot_centrale.shape[1]

plt.imshow(
    shot_centrale, 
    aspect="auto", 
    cmap="gray", 
    vmin=-5, 
    vmax=5,
    extent=[0, n_rec, tn * 1000, 0]
)

plt.title("Shot Centrale - Deepwave (Ottimizzato per Devito)")
plt.xlabel("Indice Ricevitore")
plt.ylabel("Tempo (ms)")
plt.colorbar(label="Ampiezza Pressione")
plt.savefig("deepwave_optimized.png", bbox_inches='tight')
plt.show()