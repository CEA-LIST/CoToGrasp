import torch
import numpy as np

def compute_aligned_dist_v2(X, Y, gamma=2.0, delta=0.1, use_sqrt=True, return_distances=False):
    """
    Computes the aligned distance between two point sets X and Y.

    Args:
        X (torch.Tensor): Point set X, with normals, of shape (B, N, 6).
        Y (torch.Tensor): Point set Y of shape (B, M, 3).
        gamma (float, optional): Exponential weighting factor for alignment. Defaults to 2.0.
        delta (float, optional): Scaling factor for the final aligned distance. Defaults to 0.1.
        use_sqrt (bool, optional): Whether to apply square root to the final aligned distance. Defaults to True.
        return_distances (bool, optional): If True, returns the full aligned distances tensor. Defaults to False.

    Returns:
        torch.Tensor: Aligned distances of shape (B, M).
        torch.Tensor: Indices of the closest points in X for each point in Y of shape (B, M).
        torch.Tensor (optional): Aligned distances of shape (B, N, M) if return_distances is True.
    """
    # print(f"X shape: {X.shape}, Y shape: {Y.shape}")  # Debugging shape

    if X.dim() < 3:
        X = X.unsqueeze(0)  # Ensure X is at least 3D (B, N, 6)

    B, N, _ = X.shape
    if Y.dim() < 3:
        Y = Y.unsqueeze(0).repeat(B, 1, 1)  # Ensure Y is at least 3D (B, M, 3)
    _, M, _ = Y.shape
    
    # Extract points and normals from X
    X_points = X[:, :, :3]  # (B, N, 3)
    X_normals = X[:, :, 3:]  # (B, N, 3)
    
    # Expand dimensions for pairwise computation
    X_points_expanded = X_points.unsqueeze(2)  # (B, N, 1, 3)
    X_normals_expanded = X_normals.unsqueeze(2)  # (B, N, 1, 3)
    Y_expanded = Y.unsqueeze(1)  # (B, 1, M, 3)
    
    # Repeat to create pairwise combinations
    X_points_expanded = X_points_expanded.repeat(1, 1, M, 1)  # (B, N, M, 3)
    X_normals_expanded = X_normals_expanded.repeat(1, 1, M, 1)  # (B, N, M, 3)
    Y_expanded = Y_expanded.repeat(1, N, 1, 1)  # (B, N, M, 3)
    
    # print(f"X_points_expanded: {X_points_expanded.shape}, Y_expanded: {Y_expanded.shape}")  # Debugging shape

    # Compute distances
    deltas = Y_expanded - X_points_expanded  # (B, N, M, 3)
    dists = deltas.norm(dim=3)  # (B, N, M)
    
    # Compute alignment
    alignment = (deltas * X_normals_expanded).sum(dim=3)  # (B, N, M)
    alignment = alignment / (dists + 1e-5)  # Normalize by distance
    
    # Compute aligned distance
    aligned_dist = dists * torch.exp(gamma * (1.0 - alignment))  # (B, N, M)

    if return_distances:
        return aligned_dist

    # Take minimum over N dimension to get final result
    result, indices = aligned_dist.min(dim=1)  # result: (B, M), indices: (B, M)
    result = result / delta

    if use_sqrt:
        result = torch.sqrt(result)

    return result, indices

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array

def sample_sphere_uniform(n_points=1000, n_dims=3, radius=1.0, random_seed=13):
    """Sample uniformly from d-dimensional unit ball

    The code is inspired by this small note:
    https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html

    Parameters
    ----------
    n_points : int
        number of samples
    n_dims : int
        number of dimensions
    radius: float
        ball radius
    random_seed: int
        random seed for basis point selection
    Returns
    -------
    x : numpy array
        points sampled from d-ball
    """
    np.random.seed(random_seed)
    # sample point from d-sphere
    x = np.random.normal(size=[n_points, n_dims])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms
    # now sample radiuses uniformly
    r = np.random.uniform(size=[n_points, 1])
    u = np.power(r, 1.0 / n_dims)
    x = radius * x_unit * u
    np.random.seed(None)
    return to_tensor(x)

def sample_sphere_nonuniform(n_points=1000, n_dims=3, radius=1.0, random_seed=13):
    """Sample nonuniformly from d-dimensional unit ball

    The code is inspired by this small note:
    https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html

    Parameters
    ----------
    n_points : int
        number of samples
    n_dims : int
        number of dimensions
    radius: float
        ball radius
    random_seed: int
        random seed for basis point selection
    Returns
    -------
    x : numpy array
        points sampled from d-ball
    """
    np.random.seed(random_seed)
    # sample point from d-sphere
    x = np.random.normal(size=[n_points, n_dims])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms
    # now sample radiuses uniformly
    r = np.random.uniform(size=[n_points, 1])
    u = np.power(r, 1.0 / 1.5) # set the 1.5 to change the distribution
    x = radius * x_unit * u
    np.random.seed(None)
    return to_tensor(x)

def sample_grid_cube(grid_size=32, n_dims=3, minv=-1.0, maxv=1.0):
    """ Generate d-dimensional grid BPS basis
    Parameters
    ----------
    grid_size: int
        number of elements in each grid axe
    minv: float
        minimum element of the grid
    maxv
        maximum element of the grid
    Returns
    -------
    basis: numpy array [grid_size**n_dims, n_dims]
        n-d grid points
    """

    linspaces = [np.linspace(minv, maxv, num=grid_size) for d in range(0, n_dims)]
    coords = np.meshgrid(*linspaces)
    basis = np.concatenate([coords[i].reshape([-1, 1]) for i in range(0, n_dims)], axis=1)

    return to_tensor(basis)

def sample_grid_sphere(n_points=1000, n_dims=3, radius=1.0):
    grid_points = int(6 * n_points / np.pi)
    grid_size = int(np.power(grid_points, 1 / n_dims))
    in_sphere_points = 0
    while in_sphere_points < n_points:
        c_grid = to_np(sample_grid_cube(grid_size=grid_size)) * radius
        in_sphere_points = np.where(np.linalg.norm(c_grid, axis=1) < radius)[0].shape[0]
        grid_size += 1

    c_grid = to_np(sample_grid_cube(grid_size=grid_size - 2)) * radius
    in_sphere = np.where(np.linalg.norm(c_grid, axis=1) < radius)[0]
    on_sphere_size = n_points - in_sphere.shape[0]
    in_sp = c_grid[in_sphere]
    on_sp = fibonacci_sphere(on_sphere_size) * radius
    sphere = np.concatenate([in_sp, on_sp], 0)
    return to_tensor(sphere)

def fibonacci_sphere(samples=1, randomize=True):
    import math
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2. / samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return np.array(points)

class bps_torch():
    def __init__(self, custom_basis, n_dims=3):

        basis_set = to_tensor(custom_basis)

        self.bps = basis_set.reshape(1,-1,n_dims)

        if self.bps.ndim > 2:
            self.bps = self.bps.squeeze(0)

    def encode(self, x):
        """Compute BPS

        Args:
            x (_type_): (B, N, 6)

        Returns:
            BPS: dict with 'dists' (B, M), 'ids' (B, M) and 'deltas' (B, M, 3)
        """
        x = to_tensor(x)
        is_batch = True if x.ndim > 2 else False

        if not is_batch:
            x = x.unsqueeze(0)

        aligned_dist, indices = compute_aligned_dist_v2(X=x, Y=self.bps, gamma=2.0, delta=0.1, use_sqrt=True)
        matched_points = torch.gather(x[:, :, :3], 1, indices.unsqueeze(-1).expand(-1, -1, 3))  # (B, M, 3)
        deltas = self.bps.unsqueeze(0).expand(x.shape[0], -1, -1) - matched_points  # (B, M, 3)

        x_bps = {}
        x_bps['dists'] = aligned_dist
        x_bps['ids'] = indices
        x_bps['deltas'] = deltas
        return x_bps