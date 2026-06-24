import torch
from scipy.spatial.transform import Rotation

def matrix_to_euler(matrix):
    device = matrix.device
    # forward_kinematics() requires intrinsic euler ('XYZ')
    euler = Rotation.from_matrix(matrix.cpu().numpy()).as_euler('XYZ')
    return torch.tensor(euler, dtype=torch.float32, device=device)

def euler_to_matrix(euler):
    """euler should be in radians"""
    device = euler.device
    matrix = Rotation.from_euler('XYZ', euler.cpu().numpy(), degrees=False).as_matrix()
    return torch.tensor(matrix, dtype=torch.float32, device=device)

# def matrix_to_rot6d(matrix):
#     return matrix.T.reshape(9)[:6]

def matrix_to_rot6d(matrix):
    # Ensure the matrix is in the correct shape for batched operations
    if matrix.ndimension() == 3:
        # Batched case: transpose the last two dimensions
        matrix = matrix.transpose(-1, -2)
    else:
        # Single matrix case: transpose the matrix
        matrix = matrix.T
    # Reshape to 6D representation
    return matrix.reshape(matrix.shape[:-2] + (-1,))[..., :6]

def rot6d_to_matrix(rot6d):
    x = normalize(rot6d[..., 0:3])
    y = normalize(rot6d[..., 3:6])
    a = normalize(x + y)
    b = normalize(x - y)
    x = normalize(a + b)
    y = normalize(a - b)
    z = normalize(torch.cross(x, y, dim=-1))
    matrix = torch.stack([x, y, z], dim=-2).mT
    return matrix

def euler_to_rot6d(euler):
    matrix = euler_to_matrix(euler)
    return matrix_to_rot6d(matrix)

def quaternion_to_rot6d(quaternion):
    """ quaternion should be in (x, y, z, w) format """
    matrix = quaternion_to_matrix(quaternion)
    return matrix_to_rot6d(matrix)

def rot6d_to_euler(rot6d):
    matrix = rot6d_to_matrix(rot6d)
    return matrix_to_euler(matrix)

def axisangle_to_matrix(axis, angle):
    (x, y, z), c, s = axis, torch.cos(angle), torch.sin(angle)
    return torch.tensor([
        [(1 - c) * x * x + c, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
        [(1 - c) * x * y + s * z, (1 - c) * y * y + c, (1 - c) * y * z - s * x],
        [(1 - c) * x * z - s * y, (1 - c) * y * z + s * x, (1 - c) * z * z + c]
    ])

def euler_to_quaternion(euler):
    """ euler should be in (x, y, z) format and quaternion is in (x, y, z, w) format """
    device = euler.device
    quaternion = Rotation.from_euler('XYZ', euler.cpu().numpy()).as_quat()
    return torch.tensor(quaternion, dtype=torch.float32, device=device)

def quaternion_to_euler(quaternion):
    """ quaternion should be in (x, y, z, w) format """
    device = quaternion.device
    euler = Rotation.from_quat(quaternion.cpu().numpy()).as_euler('XYZ')
    return torch.tensor(euler, dtype=torch.float32, device=device)

def quaternion_to_matrix(quaternion):
    """ quaternion should be in (x, y, z, w) format """
    device = quaternion.device
    matrix = Rotation.from_quat(quaternion.cpu().numpy()).as_matrix()
    return torch.tensor(matrix, dtype=torch.float32, device=device)

def normalize(v):
    return v / torch.norm(v, dim=-1, keepdim=True)

def q_euler_to_q_rot6d(q_euler):
    """ euler should be in radians 
    
    Args:
        q_euler: (B, 6+) tensor where the first 6 values are (x, y, z, roll, pitch, yaw) and the rest are other parameters (e.g. joint angles)
    Returns:
        q_rot6d: (B, 9+) tensor where the first 6 values are (x, y, z, rot6d) and the rest are other parameters (e.g. joint angles)

    """
    return torch.cat([q_euler[..., :3], euler_to_rot6d(q_euler[..., 3:6]), q_euler[..., 6:]], dim=-1)

def q_rot6d_to_q_euler(q_rot6d):
    """ Convert 6D representation back to euler angles in radians """
    return torch.cat([q_rot6d[..., :3], rot6d_to_euler(q_rot6d[..., 3:9]), q_rot6d[..., 9:]], dim=-1)

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    # Create orthonormal vectors
    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    z = normalize_vector(torch.cross(x, y, dim=1))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix

def normalize_vector(v):
    v_mag = torch.norm(v, dim=1, keepdim=True)
    v_mag = torch.clamp(v_mag, min=1e-8) # Avoid division by zeros
    v = v / v_mag
    return v

def get_rotation_matrix_from_vectors(v_from, v_to):
    """
    Calculates the rotation matrix that rotates v_from to v_to.
    Formula: R = I + [v]_x + [v]_x^2 * (1 / (1 + c))
    where c = dot(a, b) and [v]_x is the skew-symmetric matrix of cross(a, b).
    
    :param v_from: (B, 3) source vectors
    :param v_to:   (B, 3) target vectors
    :return:       (B, 3, 3) rotation matrices
    """
    # 1. Normalize input vectors
    v_from = v_from / torch.norm(v_from, dim=1, keepdim=True)
    v_to   = v_to   / torch.norm(v_to, dim=1, keepdim=True)

    # 2. Compute Cross Product (Rotation Axis) and Dot Product (Cosine)
    v = torch.cross(v_from, v_to, dim=1)  # (B, 3)
    c = torch.sum(v_from * v_to, dim=1)   # (B,)

    # 3. Create Skew-Symmetric Matrix [v]_x
    # Tensor: [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    zero = torch.zeros_like(c)
    vx = torch.stack([
        torch.stack([zero, -v[:, 2], v[:, 1]], dim=1),
        torch.stack([v[:, 2], zero, -v[:, 0]], dim=1),
        torch.stack([-v[:, 1], v[:, 0], zero], dim=1)
    ], dim=1)  # (B, 3, 3)

    I = torch.eye(3, device=v_from.device).unsqueeze(0)
    R = I + vx + torch.bmm(vx, vx) * (1.0 / (1.0 + c + 1e-6)).view(-1, 1, 1)
    
    return R