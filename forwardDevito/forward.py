"""
Compute the forward modeling in a 2D domain using Devito - CORRECTED VERSION

Image and Sound Processing Lab - Politecnico di Milano

Daniele Ugo Leonzio
Paolo Bestagini
"""

import argparse
import multiprocessing as mp
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from devito import configuration
from examples.seismic import AcquisitionGeometry, SeismicModel
from tqdm import tqdm

from devito_utils import forward_single_source

configuration['log-level'] = 'WARNING'
os.nice(10)


def main(args):
    """
    Main function to compute the forward modeling

    Returns:
        None
    """

    # Create a pool of processes (needed for Multiprocessing)
    num_cpu = mp.cpu_count()  # Number of processes to create
    pool = mp.Pool(num_cpu)

    # Setup parameters
    vp_model_path = args.vp_model_path
    dt = args.dt
    t0 = 0
    tn = args.tn
    f0 = args.f0
    src_type = args.src_type
    nbl = args.nbl
    space_order = args.space_order

    # Geometry parameters
    rec_spacing = args.rec_spacing
    rec_depth = args.rec_depth
    src_spacing = args.src_spacing
    src_depth = args.src_depth
    aperture = args.aperture

    # Load the velocity model
    npzfile = np.load(vp_model_path)
    vp = npzfile["vp"]
    spacing = npzfile["spacing"]
    model = SeismicModel(vp=vp, origin=(0, 0), shape=vp.shape, spacing=spacing, nbl=nbl,
                         bcs="damp", space_order=space_order)
    model.dt_scale = dt / model.critical_dt
    assert model.dt_scale <= 1, "Time step too large for model CFL conditions on the true model"

    # Plot the velocity model
    plt.figure()
    plt.imshow(model.vp.data[model.nbl:-model.nbl, model.nbl:-model.nbl].T, cmap='jet')
    plt.title('Velocity model')
    plt.savefig(os.path.join("data", "v_models", os.path.splitext(os.path.basename(vp_model_path))[0]),
                bbox_inches='tight')
    plt.show()

    # Source coordinates
    n_src = int(np.ceil(model.domain_size[0]/src_spacing))  # Fit as many shots as possible
    src_coordinates = np.empty((n_src, 2))
    src_coordinates[:, 0] = np.arange(0, src_spacing * n_src, src_spacing)
    src_coordinates[:, 1] = src_depth

    # Receiver coordinates
    n_rec = int(np.ceil(model.domain_size[0]/rec_spacing))  # Fit as many receivers as possible
    rec_coordinates = np.empty((n_rec, 2))
    rec_coordinates[:, 0] = np.arange(0, n_rec * rec_spacing, rec_spacing)
    rec_coordinates[:, 1] = rec_depth

    # Create the geometry
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0=t0, tn=tn, f0=f0, src_type=src_type)

    # Generate the observed data for all the sources considering infinite aperture (the entire model)
    fun_part = partial(forward_single_source, model, geometry, dt, aperture, False, False)
    result_list = list(tqdm(pool.imap(fun_part, range(n_src)), total=n_src, desc='Forward modeling'))
    d_obs_list = [result[0] for result in result_list]

    # HO MODIFICATO LA GESTIONE DEL PLOT. SOTTO COMMENTO QUELLO ORIGINALE.
    # Plot the observed data with proper time axis in milliseconds
    # Using all samples.
    plt.figure(figsize=(10, 8))
    shot_idx = len(d_obs_list) // 2
    shot_data = d_obs_list[shot_idx].data  # Use ALL samples
    n_rec_plot = shot_data.shape[1]
    
    print(f"Shot data shape: {shot_data.shape}")
    print(f"Expected time samples: {int(tn/dt)} for tn={tn}ms, dt={dt}ms")
    
    # X-axis: receiver indices from 0 to n_rec
    # Y-axis: time from tn (bottom) to 0 (top) in milliseconds
    plt.imshow(shot_data, cmap='gray', aspect='auto', vmin=-5, vmax=5, 
               extent=[0, n_rec_plot, tn, 0])
    
    plt.title(f'Shot Centrale (Indice {shot_idx}) - Devito')
    plt.xlabel('Indice Ricevitore')
    plt.ylabel('Tempo (ms)')
    plt.colorbar(label='Ampiezza Pressione')
    plt.savefig(os.path.join("data", "shots", 
                             f"{os.path.splitext(os.path.basename(vp_model_path))[0]}_full.png"),
                bbox_inches='tight')
    plt.show()

    '''
    # Plot the observed data
    plt.figure()
    plt.imshow(d_obs_list[len(d_obs_list)//2].data, cmap='gray', aspect='auto', clim=[-5, 5])
    plt.title('Shot')
    plt.savefig(os.path.join("data", "shots", os.path.splitext(os.path.basename(vp_model_path))[0]),
                bbox_inches='tight')
    plt.show()
    '''

    # Save shots
    out_path = os.path.join("data", "shots", os.path.basename(vp_model_path))
    np.savez(out_path, d_obs_list=d_obs_list, src_coordinates=src_coordinates,
             rec_coordinates=rec_coordinates, t0=t0, tn=tn, dt=dt, f0=f0, src_type=src_type, nbl=nbl,
             space_order=args.space_order, aperture=aperture, allow_pickle=True)

    # Close the pool of processes
    pool.close()
    pool.join()

    return None


if __name__ == "__main__":
    """
    Run the main function
    """

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Full Waveform Inversion')
    parser.add_argument('--vp_model_path', type=str, default="./data/v_models/marmousi_sp25.npz", 
                        help='Path to the true velocity model')
    parser.add_argument('--space_order', type=int, default=4, help='Space order of the simulation')
    parser.add_argument('--src_spacing', type=int, default=100, help='Spacing between shots')
    parser.add_argument('--rec_spacing', type=int, default=25, help='Spacing between receivers')
    parser.add_argument('--rec_depth', type=int, default=20, help='Depth of the receivers')
    parser.add_argument('--src_depth', type=int, default=20, help='Depth of the shots')
    parser.add_argument('--f0', type=float, default=0.005, help='Dominant frequency of the source wavelet')
    parser.add_argument('--src_type', type=str, default='Ricker', help='Wavelet source type')
    parser.add_argument('--tn', type=float, default=6000, help='Recording end time in ms')
    parser.add_argument('--aperture', type=float, default=5000, 
                        help='Aperture in meters to the left and the right of the source in meters')
    parser.add_argument('--dt', type=float, default=3, help='Time step in ms')
    parser.add_argument('--nbl', type=int, default=200, help='Number of boundary layers')
    args = parser.parse_args()

    # Run the main function
    main(args)