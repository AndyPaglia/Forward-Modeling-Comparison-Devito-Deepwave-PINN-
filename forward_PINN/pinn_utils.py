"""
Utility functions for the PINN implementation

Image and Sound Processing Lab - Politecnico di Milano

Daniele Ugo Leonzio
Paolo Bestagini
"""

import os

import GPUtil
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def set_gpu(id=-1):
    """
    Set tensor computation device

    :param id: CPU or GPU device id (None for CPU, -1 for the device with the lowest memory usage, or the ID)

    hint: use gpustat (pip install gpustat) in a bash CLI, or gputil (pip install gputil) in python.
    """
    if id is None:
        # CPU only
        print('GPU not selected')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    else:
        # -1 for automatic choice
        device = id if id != -1 else GPUtil.getFirstAvailable(order='memory')[0]
        try:
            name = GPUtil.getGPUs()[device].name
        except IndexError:
            print('The selected GPU does not exist. Switching to the most available one.')
            device = GPUtil.getFirstAvailable(order='memory')[0]
            name = GPUtil.getGPUs()[device].name
        print('GPU selected: %d - %s' % (device, name))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


def generate_pml_coefficients_2d(domain_shape, N=50, B=100., multiple=False):
    Nx, Ny = domain_shape

    R = 10 ** (-((np.log10(N) - 1) / np.log10(2)) - 3)
    # d0 = -(order+1)*cp/(2*abs_N)*np.log(R) # Origin
    R = 1e-6;
    order = 2;
    cp = 1000.
    d0 = (1.5 * cp / N) * np.log10(R ** -1)
    d_vals = d0 * torch.linspace(0.0, 1.0, N + 1) ** order
    d_vals = torch.flip(d_vals, [0])

    d_x = torch.zeros(Ny, Nx)
    d_y = torch.zeros(Ny, Nx)

    if N > 0:
        d_x[0:N + 1, :] = d_vals.repeat(Nx, 1).transpose(0, 1)
        d_x[(Ny - N - 1):Ny, :] = torch.flip(d_vals, [0]).repeat(Nx, 1).transpose(0, 1)
        if not multiple:
            d_y[:, 0:N + 1] = d_vals.repeat(Ny, 1)
        d_y[:, (Nx - N - 1):Nx] = torch.flip(d_vals, [0]).repeat(Ny, 1)

    _d = torch.sqrt(d_x ** 2 + d_y ** 2).transpose(0, 1)
    _d = _corners(domain_shape, N, _d, d_x.T, d_y.T, multiple)

    return _d


def _corners(domain_shape, abs_N, d, dx, dy, multiple=False):
    Nx, Ny = domain_shape
    for j in range(Ny):
        for i in range(Nx):
            # Left-Top
            if not multiple:
                if i < abs_N + 1 and j < abs_N + 1:
                    if i < j:
                        d[i, j] = dy[i, j]
                    else:
                        d[i, j] = dx[i, j]
            # Left-Bottom
            if i > (Nx - abs_N - 2) and j < abs_N + 1:
                if i + j < Nx:
                    d[i, j] = dx[i, j]
                else:
                    d[i, j] = dy[i, j]
            # Right-Bottom
            if i > (Nx - abs_N - 2) and j > (Ny - abs_N - 2):
                if i - j > Nx - Ny:
                    d[i, j] = dy[i, j]
                else:
                    d[i, j] = dx[i, j]
            # Right-Top
            if not multiple:
                if i < abs_N + 1 and j > (Ny - abs_N - 2):
                    if i + j < Ny:
                        d[i, j] = dy[i, j]
                    else:
                        d[i, j] = dx[i, j]

    return d


def absorbing_boundaries(nx, ny, nb, u):
    """
    Generate the absorbing boundary coefficients for a 2D domain

    :param nx: size of the domain in the x direction
    :param ny: size of the domain in the y direction
    :param nb: number of points in the absorbing boundary
    :param u: scaling factor

    :return: matrix of the absorbing boundary coefficients
    """
    # Generate the absorbing boundary coefficients for the left, right, top and bottom boundaries
    bound_coeffs = np.exp(-u * (nb - np.arange(nb)) ** 2)

    # Concatenate the coefficients for the left, center, and right parts of the domain
    central_coeffs_x = np.ones(nx - (2 * nb))
    coeffs_mask_x = np.concatenate([bound_coeffs, central_coeffs_x, bound_coeffs[::-1]])
    coeffs_mask_x = np.tile(coeffs_mask_x, (ny, 1))

    # Concatenate the coefficients for the bottom, center, and top parts of the domain
    central_coeffs_y = np.ones(ny - (2 * nb))
    coeffs_mask_y = np.concatenate([bound_coeffs, central_coeffs_y, bound_coeffs[::-1]])[:, np.newaxis]
    coeffs_mask_y = np.tile(coeffs_mask_y, (1, nx))

    # Combine the x and y coefficients to get the final absorbing boundary coefficients
    coeffs_mask = coeffs_mask_x * coeffs_mask_y

    return coeffs_mask


def laplace(u, h, dev):
    """
    Compute the laplacian of a field u using a 4th order finite difference scheme

    :param u: field to compute the laplacian
    :param h: grid spacing
    :param dev: device

    :return: laplacian of the field u
    """
    # Chose the kernel order
    kernel_order = 2  # supported 4, 6, 8

    # Define the 2nd order laplacian kernel coefficients
    if kernel_order == 2:
        kernel_size = 3
        coeffs = torch.tensor([1, -2, 1])

    # Define the 4th order laplacian kernel coefficients
    if kernel_order == 4:
        kernel_size = 5
        coeffs = torch.tensor([-1/12, 4/3, -5/2, 4/3, -1/12])

    # Define the 6th order laplacian kernel coefficients
    elif kernel_order == 6:
        kernel_size = 7
        coeffs = torch.tensor([-1/90, 3/20, -3/2, 49/18, -3/2, 3/20, -1/90])

    # Define the 8th order laplacian kernel coefficients
    elif kernel_order == 8:
        kernel_size = 9
        coeffs = torch.tensor([1/560, -8/315, 1/5, -8/5, 205/72, -8/5, 1/5, -8/315, 1/560])

    # Define the kernel
    kernel = torch.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] += coeffs
    kernel[:, kernel_size // 2] += coeffs
    kernel = kernel.view(1, 1, kernel_size, kernel_size).to(dev)

    # Compute the laplacian
    laplacian = torch.nn.functional.conv2d(u, kernel, padding=kernel_size//2) / (h ** 2)

    return laplacian


def step(u_pre, u_now, dev, c, dt, h, b=None):
    """
    Compute the next step of the wavefield using the 2nd order wave equation

    :param u_pre: field at the previous time step
    :param u_now: field at the current time step
    :param dev: device
    :param c: velocity
    :param dt: time step
    :param h: grid spacing

    :return: field at the next time step
    """
    # Compute the next step of the wavefield
    #u_next = 2 * u_now - u_pre + (c * dt) ** 2 * laplace(u_now, h, dev)
    u_next = torch.mul((dt**-2 + b * dt**-1).pow(-1),
                       (2 / dt**2 * u_now - torch.mul((dt**-2 - b * dt**-1), u_pre)
                        + torch.mul(c.pow(2), laplace(u_now, h, dev))))

    if torch.isnan(u_next).any():
        print('Nan values in the wavefield')
        print(u_next)

    return u_next


def forward(wave, c, b, src_list, domain, dt, h, dev, recz, pmln):
    """
    Compute the forward modeling of the wavefield

    :param wave: wavelet
    :param c: velocity model
    :param b: boundary mask
    :param src_list: list of source coordinates
    :param domain: domain size
    :param dt: time step
    :param h: grid spacing
    :param dev: device
    :param recz: receiver depth
    :param pmln: number of points in the absorbing boundary

    :return: recorded data
    """
    # Number of time steps
    nt = wave.shape[0]

    # Domain size
    nz, nx = domain

    # Number of shots
    nshots = len(src_list)

    # Initialize wavefields
    u_pre = torch.zeros(nshots, 1, *domain).to(dev)
    u_now = torch.zeros(nshots, 1, *domain).to(dev)

    # Initialize receiver data
    rec = torch.zeros(nshots, nt, nx - 2 * pmln).to(dev)

    # Unsqueeze boundary mask and velocity model
    b = b.unsqueeze(0).to(dev)
    c = c.unsqueeze(0)

    # Create tensor for shot indices
    shots = torch.arange(nshots).to(dev)

    # Extract source coordinates
    srcx, srcz = zip(*src_list)

    # Convert grid spacing and time step to tensors
    h = torch.Tensor([h]).to(dev)
    dt = torch.Tensor([dt]).to(dev)

    # Initialize source mask
    source_mask = torch.zeros_like(u_now)
    source_mask[shots, :, srcz, srcx] = 1

    # Apply boundary mask
    #b_mask = b
    #b_mask = b_mask.unsqueeze(0).to(dev)

    # Forward modeling loop
    # for it in tqdm(range(nt), desc='Forward modeling', leave=False):
    for it in range(nt):

        # Add source term
        u_now += source_mask * wave[it]

        # Compute next wavefield
        u_next = step(u_pre, u_now, dev, c, dt, h, b)

        # Apply boundary mask and update wavefields
        #u_pre, u_now = u_now * b_mask, u_next * b_mask
        u_pre, u_now = u_now, u_next

        # Record data at receiver locations
        rec[:, it, :] = u_now[:, 0, recz, pmln:-pmln]

    return rec


class SineLayer(nn.Module):
    """
    Sine layer for the physics-informed neural network
    """
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    """
    Physics-informed neural network with sine activation functions
    """
    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 outermost_linear=False,
                 pretrained=None,
                 first_omega_0=30,
                 hidden_omega_0=30.,
                 domain_shape=None,
                 dh=None,
                 **kwargs):
        super().__init__()
        self.domain_shape = domain_shape
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
            # self.net.append(nn.Tanh())
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)
        self.coords = self.generate_mesh(domain_shape, dh)
        self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained):
        pretrained = '' if pretrained is None else pretrained
        if os.path.exists(pretrained):
            self.load_state_dict(torch.load(pretrained, weights_only=True))
        else:
            print(f"Cannot find the pretrained model '{pretrained}'. Using random initialization.")

    def generate_mesh(self, mshape, dh):
        tensors_for_meshgrid = []
        for size in mshape:
            tensors_for_meshgrid.append(torch.linspace(-1, 1, steps=size))
        mgrid = torch.stack(torch.meshgrid(*tensors_for_meshgrid, indexing='ij'), dim=-1)
        mgrid = mgrid.reshape(-1, len(mshape))
        return mgrid

    def step(self, ):
        pass

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords).view(self.domain_shape)
        return output, coords


class SirenElastic(nn.Module):
    """
    Physics-informed neural network with sine activation functions (elastic)
    """
    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 outermost_linear=False,
                 pretrained=None,
                 first_omega_0=30,
                 hidden_omega_0=30.,
                 domain_shape=None,
                 dh=None,
                 **kwargs):
        super().__init__()
        self.domain_shape = domain_shape
        self.out_features = out_features
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)
        self.coords = self.generate_mesh(domain_shape, dh)
        self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained):
        pretrained = '' if pretrained is None else pretrained
        if os.path.exists(pretrained):
            self.load_state_dict(torch.load(pretrained))
        else:
            print(f"Cannot find the pretrained model '{pretrained}'. Using random initialization.")

    def generate_mesh(self, mshape, dh):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int'''
        tensors_for_meshgrid = []
        for size in mshape:
            tensors_for_meshgrid.append(torch.linspace(-1, 1, steps=size))
            # tensors_for_meshgrid.append(torch.linspace(0, size*dh/1000, steps=size))
        mgrid = torch.stack(torch.meshgrid(*tensors_for_meshgrid, indexing='ij'), dim=-1)
        mgrid = mgrid.reshape(-1, len(mshape))
        return mgrid

    def step(self, ):
        pass

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords).view(*self.domain_shape, self.out_features)
        return output, coords


def gradient(input, dim=-1, forward=True, padding_value=0):
    def forward_diff(x, dim=-1, padding_value=0):
        """
        Compute the forward difference of an input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int, optional): The dimension along which to compute the difference.
            padding_value (float, optional): The value to use for padding.

        Returns:
            torch.Tensor: The forward difference of the input tensor.
        """
        # x[:,0] = padding_value
        diff = x - torch.roll(x, shifts=1, dims=dim)
        if dim == 1:
            diff[:, 0] = padding_value
        elif dim == 2:
            diff[..., 0] = padding_value  # pad with specified value
        return diff

    def backward_diff(x, dim=-1, padding_value=0):
        """
        Compute the backward difference of an input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int, optional): The dimension along which to compute the difference.
            padding_value (float, optional): The value to use for padding.

        Returns:
            torch.Tensor: The backward difference of the input tensor.
        """
        # x[...,-1] = padding_value
        diff = torch.roll(x, shifts=-1, dims=dim) - x
        if dim == 1:
            diff[:, -1] = padding_value
        elif dim == 2:
            diff[..., -1] = padding_value  # pad with specified value
        return diff

    if forward:
        return forward_diff(input, dim=dim)
    else:
        return backward_diff(input, dim=dim)


def step_elastic(parameters, wavefields, geometry):
    vp, vs, rho = parameters
    vx, vz, txx, tzz, txz = wavefields
    dt, h, d = geometry

    lame_lambda = rho * (vp.pow(2) - 2 * vs.pow(2))
    lame_mu = rho * (vs.pow(2))
    c = 0.5 * dt * d

    vx_x = gradient(vx, 2)
    vz_z = gradient(vz, 1, False)
    vx_z = gradient(vx, 1)
    vz_x = gradient(vz, 2, False)

    # Equation A-8
    y_txx = (1 + c) ** -1 * (dt * h.pow(-1) * ((lame_lambda + 2 * lame_mu) * vx_x + lame_lambda * vz_z) + (1 - c) * txx)
    # Equation A-9
    y_tzz = (1 + c) ** -1 * (dt * h.pow(-1) * ((lame_lambda + 2 * lame_mu) * vz_z + lame_lambda * vx_x) + (1 - c) * tzz)
    # Equation A-10
    y_txz = (1 + c) ** -1 * (dt * lame_mu * h.pow(-1) * (vz_x + vx_z) + (1 - c) * txz)

    txx_x = gradient(y_txx, 2, False)
    txz_z = gradient(y_txz, 1, False)
    tzz_z = gradient(y_tzz, 1)
    txz_x = gradient(y_txz, 2)

    # Update y_vx
    y_vx = (1 + c) ** -1 * (dt * rho.pow(-1) * h.pow(-1) * (txx_x + txz_z) + (1 - c) * vx)
    # Update y_vz
    y_vz = (1 + c) ** -1 * (dt * rho.pow(-1) * h.pow(-1) * (txz_x + tzz_z) + (1 - c) * vz)

    return y_vx, y_vz, y_txx, y_tzz, y_txz


def forward_elastic(wave, parameters, pmlc, src_list, domain, dt, h, dev, npml=50, recz=0):
    nt = wave.shape[0]
    nz, nx = domain
    nshots = len(src_list)

    dt = torch.tensor(dt, dtype=torch.float32, device=dev)
    h = torch.tensor(h, dtype=torch.float32, device=dev)

    vx = torch.zeros(nshots, *domain, device=dev)
    vz = torch.zeros(nshots, *domain, device=dev)
    txx = torch.zeros(nshots, *domain, device=dev)
    tzz = torch.zeros(nshots, *domain, device=dev)
    txz = torch.zeros(nshots, *domain, device=dev)

    wavefields = [vx, vz, txx, tzz, txz]
    geoms = [dt, h, pmlc]

    rec = torch.zeros(nshots, nt, nx - 2 * npml).to(dev)

    shots = torch.arange(nshots).to(dev)
    srcx, srcz = zip(*src_list)
    src_mask = torch.zeros_like(vx)
    src_mask[shots, srcz, srcx] = 1

    for it in range(nt):
        # GPU ALIGNED
        wavefields[1] += src_mask * wave[it]
        wavefields = step_elastic(parameters, wavefields, geoms)
        wavefields = list(wavefields)
        rec[:, it, :] = wavefields[1][:, recz, npml:-npml]
    return rec
