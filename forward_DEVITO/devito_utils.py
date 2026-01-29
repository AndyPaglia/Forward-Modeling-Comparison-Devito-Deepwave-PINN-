"""
Utility functions for the Devito

Image and Sound Processing Lab - Politecnico di Milano

Daniele Ugo Leonzio
Paolo Bestagini
"""

import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from devito import norm
from examples.seismic import AcquisitionGeometry, Receiver, SeismicModel
from examples.seismic.acoustic import AcousticWaveSolver


def crop_model_geometry(model, geometry, aperture, src_idx, dt, crop_model=True):
    """
    Crop the model and geometry to the correct aperture

    Args:
        model: SeismicModel object
        geometry: Geometry object containing all sources and all receivers positions
        aperture: Aperture in meters to the left and the right of the source
        src_idx: Index of the source within the geometry object
        dt: Time step
        crop_model: Flag to crop the model as well

    Returns:
        model_crop: Cropped SeismicModel object
        geometry_crop: Cropped Geometry object
    """

    # Compute the minimum and maximum distance for the source (not exceeding the model boundaries)
    min_dist = max(0, geometry.src_positions[src_idx, 0] - aperture)
    max_dist = min(model.domain_size[0], geometry.src_positions[src_idx, 0] + aperture)
    min_idx = int(min_dist // model.spacing[0])
    max_idx = int(max_dist // model.spacing[0])

    # Check if the model should be cropped or not
    if crop_model:
        # Crop the model boundaries
        vp_crop = model.vp.data[model.nbl:-model.nbl, model.nbl:-model.nbl]

        # Generate the cropped model
        # NOTE: set SeismicModel.origin[0]=min_dist/1000 to correctly plot the model with `plot_velocity`, but the
        # solver won't work correctly. The solver wants the origin in meters, while the plot wants the origin in
        # kilometers...
        model_crop = SeismicModel(vp=vp_crop[min_idx:max_idx+1, :],
                                  origin=[min_dist, model.origin[1]], spacing=model.spacing,
                                  shape=(max_idx-min_idx+1, model.shape[1]), nbl=model.nbl,
                                  space_order=model.space_order, bcs="damp")
        model_crop.dt_scale = dt / model_crop.critical_dt
        # TODO: Uncomment to go back to the original model
        # assert model_crop.dt_scale <= 1, "Time step too large for model CFL conditions on the cropped model"
    else:
        model_crop = model

    # Generate the cropped geometry
    src_position = geometry.src_positions[src_idx, :]
    rec_pos_idx = np.logical_and(geometry.rec_positions[:, 0] >= min_dist, geometry.rec_positions[:, 0] <= max_dist)
    rec_positions = geometry.rec_positions[rec_pos_idx, :]
    geometry_crop = AcquisitionGeometry(model_crop, rec_positions, src_position,
                                        geometry.t0, geometry.tn, f0=geometry.f0, src_type=geometry.src_type)

    return model_crop, geometry_crop


def forward_single_source(model, geometry, dt, aperture=np.inf, save=False, crop_model=True, src_idx=0):
    """
    Forward modeling for a single source

    Args:
        model: SeismicModel object
        geometry: Geometry object containing all sources and all receivers positions
        dt: Time step
        aperture: Aperture in meters to the left and the right of the source
        save: Save the entire wavefield u0
        crop_model: Flag to crop the model as well
        src_idx: Index of the source within the geometry object

    Returns:
        data: Estimated data for the given source
        model_crop: Cropped SeismicModel object
        u0: Wavefield
        geometry_crop: Cropped Geometry object
    """

    # Crop model and geometry to the correct aperture
    model_crop, geometry_crop = crop_model_geometry(model, geometry, aperture, src_idx, dt, crop_model)

    # Create the solver object
    solver = AcousticWaveSolver(model_crop, geometry_crop, space_order=model_crop.space_order)

    # Generate synthetic data from true model
    if save:
        data, u0, _ = solver.forward(vp=model_crop.vp, save=True)
    else:
        data, _, _ = solver.forward(vp=model_crop.vp, save=False)
        u0 = None

    return data, model_crop, u0, geometry_crop


def forward_and_grad_single_source(model, geometry, dt, aperture=np.inf, save=True, crop_model=True, args=None):
    """
    Forward modeling and gradient computation for a single source

    Args:
        model: SeismicModel object
        geometry: Geometry object containing all sources and all receivers positions
        dt: Time step
        aperture: Aperture in meters to the left and the right of the source
        save: Save the entire wavefield u0
        crop_model: Flag to crop the model as well
        args: Tuple containing the following elements
            d_obs: Observed data for the given source
            src_idx: Index of the source within the geometry object

    Returns:
        model_crop: Cropped SeismicModel object
        objective: Objective function value
        grad: Gradient
    """

    # Run the forward modeling
    d_obs, src_idx = args
    data, model_crop, u0, geometry_crop = forward_single_source(model, geometry, dt, aperture, save, crop_model,
                                                                src_idx)

    # Wave solver
    solver = AcousticWaveSolver(model_crop, geometry_crop, space_order=model.space_order)

    # Create placeholders for the data residual and data
    residual = Receiver(name='residual', grid=model_crop.grid, time_range=geometry_crop.time_axis,
                        coordinates=geometry_crop.rec_positions)

    # Compute gradient from data residual and update objective function
    residual.data[:] = data.data[:] - d_obs.data[:]

    objective = .5 * norm(residual) ** 2
    grad = solver.gradient(rec=residual, u=u0, vp=model_crop.vp)[0]

    return model_crop, objective, grad


def loss_function(x, model0, pool, geometry, d_obs_list, aperture, v_min, v_max, dt):
    """
    Loss function for the FWI optimization using SciPy

    Args:
        x: current squared slowness model
        model0: current SeismicModel object
        pool: multiprocessing pool for parallel computation
        geometry: Geometry object containing all sources and all receivers positions
        d_obs_list: List of observations
        aperture: Aperture in meters to the left and the right of the source
        v_min: Minimum velocity value
        v_max: Maximum velocity value
        dt: Time step

    Returns:
        objective: Objective function value
        grad_data: Gradient
    """
    # Compute the number of sources
    n_src = len(d_obs_list)

    # Clip the squared slowness to the desired bounds
    x_clip = np.clip(x, 1.0 / v_max ** 2, 1.0 / v_min ** 2)

    # Convert squared slowness to velocity
    v_curr = 1.0/np.sqrt(x_clip.reshape(model0.shape))

    # Save the current velocity model
    # np.save(f'v_curr_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.npy', v_curr)

    # Update the estimated model
    model0_upd = SeismicModel(vp=v_curr,
                              origin=model0.origin, spacing=model0.spacing,
                              shape=model0.shape, nbl=model0.nbl, space_order=model0.space_order,
                              bcs="damp")
    model0_upd.dt_scale = dt / model0_upd.critical_dt
    # TODO: Uncomment to go back to the original model
    # assert model0_upd.dt_scale <= 1, "Time step too large for model CFL conditions on the estimated model"

    # Forward and gradient computation on estimated model
    fun_part = partial(forward_and_grad_single_source, model0_upd, geometry, dt, aperture, True, True)
    result_list = list(pool.imap(fun_part, zip(d_obs_list, range(n_src))))
    model0_crop_list = [result[0] for result in result_list]
    objective_list = [result[1] for result in result_list]
    grad_list = [result[2] for result in result_list]

    # Combine gradients
    objective = np.sum(objective_list)
    grad_data = np.zeros_like(model0.vp.data)
    for grad, model0_crop in zip(grad_list, model0_crop_list):
        min_idx = int(model0_crop.origin[0] // model0.spacing[0])
        max_idx = int(min_idx + grad.data.shape[0])
        grad_data[min_idx:max_idx, :] += grad.data
    grad_data = grad_data[model0.nbl:-model0.nbl, model0.nbl:-model0.nbl]

    return objective, grad_data.flatten().astype(np.float64)


def fwi_callback(xk):
    """
    Callback function to plot the current velocity model during the FWI optimization

    Args:
        xk: current squared slowness model

    Returns:
        None
    """

    # Clip the squared slowness to the desired bounds
    xk = np.clip(xk, 1.0 / 4.5 ** 2, 1.0 / 1.5 ** 2)  # TODO: avoid hardcoding the bounds

    # Convert squared slowness to velocity
    v_curr = 1.0/np.sqrt(xk.reshape([481, 121]))  # TODO: avoid hardcoding the shape (even if it's just for plotting)

    # Plot the current velocity model
    plt.imshow(v_curr.T, cmap='jet', clim=[1.5, 4.5])
    #plt.savefig(f'v_curr_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png', bbox_inches='tight')
    plt.show()

    return None
