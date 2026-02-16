"""
FWI using SciPy optimization with parallel computation

Image and Sound Processing Lab - Politecnico di Milano

Daniele Ugo Leonzio
Paolo Bestagini
"""

import argparse
import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from devito import configuration
from examples.seismic import AcquisitionGeometry, SeismicModel
from scipy import optimize
from tqdm import tqdm

from devito_utils import fwi_callback, loss_function

configuration['log-level'] = 'WARNING'
os.nice(10)


def main(args):
    """
    Main function to compare FWI performance between serial and parallel implementations

    Returns:
        None
    """

    # Create a pool of processes (needed for Multiprocessing)
    num_cpu = mp.cpu_count()  # Number of processes to create
    pool = mp.Pool(num_cpu)

    # Load the observed data
    obs_data_path = args.obs_data_path
    npzfile = np.load(obs_data_path, allow_pickle=True)
    d_obs_list = npzfile["d_obs_list"]

    # Geometry parameters
    rec_coordinates = npzfile["rec_coordinates"]
    src_coordinates = npzfile["src_coordinates"]
    aperture = npzfile["aperture"]

    # Setup parameters
    t0 = npzfile["t0"]
    tn = npzfile["tn"]
    f0 = npzfile["f0"]
    src_type = npzfile["src_type"]
    dt = npzfile["dt"]
    nbl = 100  # npzfile["nbl"]
    space_order = npzfile["space_order"]
    v_min = args.velocity_bounds[0]
    v_max = args.velocity_bounds[1]

    # FWI parameters
    fwi_iterations = args.fwi_iterations

    # Load the initial velocity model
    vp_model_path = args.vp_model_path
    npzfile = np.load(vp_model_path)
    vp = npzfile["vp"]
    spacing = npzfile["spacing"]
    model0 = SeismicModel(vp=vp, origin=(0, 0), shape=vp.shape, spacing=spacing, nbl=nbl,
                          bcs="damp", space_order=int(space_order))
    model0.dt_scale = dt / model0.critical_dt
    assert model0.dt_scale <= 1, "Time step too large for model CFL conditions on the true model"

    # Plot the initial velocity model
    '''
    plt.figure()
    plt.imshow(model0.vp.data[model0.nbl:-model0.nbl, model0.nbl:-model0.nbl].T, cmap='jet')
    plt.title('Initial velocity model')
    plt.show()
    '''

    # Create the geometry
    geometry = AcquisitionGeometry(model0, rec_coordinates, src_coordinates, t0=t0, tn=tn, f0=f0, src_type=str(src_type))

    # Init FWI main iterations
    ftol = 1e-20  # Tolerance for the optimization
    v0 = model0.vp.data[model0.nbl:-model0.nbl, model0.nbl:-model0.nbl]
    m0 = 1.0 / (v0.reshape(-1).astype(np.float64))**2  # in [s^2/km^2]

    # FWI loop (no control over inner iterations)
    '''
    result = optimize.minimize(loss_function, m0, args=(model0, pool, geometry, d_obs_list, aperture, v_min, v_max, dt),
                               method='L-BFGS-B', jac=True, callback=fwi_callback,
                               options={'ftol': ftol, 'maxiter': fwi_iterations, 'maxls': 20, 'disp': True})
    '''

    print("Starting FWI iterations...")

    # FWI loop (full control over inner iterations)
    cost_fun = []
    time_list = []
    exp_name = os.path.basename(args.obs_data_path).replace('.npz', '')
    for n_iter in tqdm(range(fwi_iterations), desc='FWI iterations', total=fwi_iterations, leave=False):

        # Start the timer
        start = time.time()

        # Run one iteration of the optimizer
        result = optimize.minimize(loss_function, m0, args=(model0, pool, geometry, d_obs_list, aperture, v_min, v_max,
                                                            dt),
                                   method='L-BFGS-B', jac=True, callback=None,
                                   options={'ftol': ftol, 'maxiter': 0, 'maxls': 20, 'disp': False})

        # Update the cost function values
        cost_fun.append(result.fun)

        # Clip the squared slowness to the desired bounds
        xk = np.clip(result.x, 1.0 / v_max ** 2, 1.0 / v_min ** 2)

        # Convert squared slowness to velocity
        vp = 1.0 / np.sqrt(xk).reshape(v0.shape)

        # Update the model
        m0 = 1.0 / (vp.reshape(-1).astype(np.float64))**2  # in [s^2/km^2]
        model0 = SeismicModel(vp=vp, origin=(0, 0), shape=vp.shape, spacing=spacing, nbl=nbl, bcs="damp",
                              space_order=int(space_order))
        model0.dt_scale = dt / model0.critical_dt
        assert model0.dt_scale <= 1, "Time step too large for model CFL conditions on the true model"

        # Stop the timer
        time_list.append(time.time() - start)

        # Save the intermediate velocity model
        out_path = os.path.join("./data", "fwi", f"{exp_name}_iter{n_iter:03d}.npz")
        np.savez(out_path, vp=vp, message=result.message, success=result.success, fun=result.fun, nit=result.nit,
                 nfev=result.nfev, njev=result.njev, cost_fun=cost_fun, time_list=time_list, args=args,
                 allow_pickle=True)

        # Plot the intermediate velocity model (save every 10 iterations)
        if n_iter % 10 == 0:
            plt.figure()
            plt.imshow(vp.T, cmap='jet', clim=(v_min, v_max))
            plt.title(f"It: {n_iter:03d} | Cost: {result.fun:.2e}")
            plt.savefig(out_path.replace('.npz', '.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Print the intermediate results
        tqdm.write(f"  It: {n_iter:03d} | Cost: {result.fun:.2e} | Time: {time_list[-1]:.2f} s")

    # Save the final result
    out_path = os.path.join("./data", "fwi", f"{exp_name}_final.npz")
    np.savez(out_path, vp=vp, message=result.message, success=result.success, fun=result.fun, nit=result.nit,
             nfev=result.nfev, njev=result.njev, cost_fun=cost_fun, time_list=time_list, args=args, allow_pickle=True)

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
    parser.add_argument('--vp_model_path', type=str, default="./data/v_models/marmousi_sm10_sp25.npz", help='Path to the initial velocity model')
    parser.add_argument('--obs_data_path', type=str, default="./data/shots/marmousi_sp25_aperture6000_2.npz", help='Path to the observed data')
    parser.add_argument('--fwi_iterations', type=int, default=1000, help='Number of FWI iterations')
    parser.add_argument('--velocity_bounds', type=float, nargs=2, default=[1.5, 4.5], help='Minimum and maximum velocity values')

    args = parser.parse_args()

    # Run the main function
    main(args)