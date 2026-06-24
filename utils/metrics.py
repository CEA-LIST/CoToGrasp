import torch
from scipy.spatial import ConvexHull
import numpy as np
from time import time
####################################################################################################
## CLASS
####################################################################################################

class GraspMetrics:
    
    m_G_min, m_G_max = 2.0, 15.0
    m_H_min, m_H_max = 1e-5, 1e-1

    def __init__(self, hand_model):
        self.hand_model = hand_model
        self.G = None
        self.G_FC = None
        self.J = None
        self.H = None
        self.can_be_fc = False
        self.rank_H = 0
        self.is_manipulable = False

    def compute_matrices(self, q, contact_points):
        """
        Compute G, J, G_FC and H matrices.
        
        :param q: joint values (B, n_joints)
        :param contact_points_normals_labels: (B, N, 7) tensor of contact points, normals and labels
        """
        self.G = grasp_matrix(contact_points)                           # (B, 6, 3*N)
        self.J = jacobian_matrix(self.hand_model, q, contact_points)    # (B, 3*N, num_joints)

        self.G_FC = wrench_space(self.G)                                # (B, 6, num_friction_vectors*N)
        self.H = hand_object_jacobian(self.G, self.J)                   # (B, 6, num_joints)

        self.can_be_fc = can_be_force_closure(self.G_FC)
        self.rank_H, self.is_manipulable = how_manipulable(self.H)

    def get_volume_wrench_space(self, normalize=False):
        if self.can_be_fc.sum() == 0:
            return torch.zeros(self.G.shape[0], device=self.G.device)
        res, _ = volume_wrench_space(self.G_FC)
        if normalize:
            res = self.log_normalize(res, self.m_G_min, self.m_G_max)
        return res
    
    def get_volume_ellipsoid_wrench_space(self, normalize=False):
        if self.can_be_fc.sum() == 0:
            return torch.zeros(self.G.shape[0], device=self.G.device)
        res = volume_ellipsoid_wrench_space(self.G_FC)
        if normalize:
            res = self.log_normalize(res, self.m_G_min, self.m_G_max)
        return res

    def get_manipulability_ellipsoid(self, normalize=False):
        if self.can_be_fc.sum() == 0:
            return torch.zeros(self.G.shape[0], device=self.G.device)
        res = manipulability_ellipsoid(self.H)
        if normalize:
            res = self.log_normalize(res, self.m_H_min, self.m_H_max)
        return res

    def get_singular_values(self, normalize=False):
        """
        :returns: sv_min, ratio
        """
        if self.can_be_fc.sum() == 0:
            return torch.zeros(self.G.shape[0], device=self.G.device, dtype=torch.float32)
        sv_min = distance_singular_configuration(self.H)
        ratio = uniformity_transformation(self.H)
        if normalize:
            sv_min = self.log_normalize(sv_min, self.m_H_min, self.m_H_max)
            ratio = self.log_normalize(ratio, self.m_H_min, self.m_H_max)
        return sv_min, ratio

    def normalise(self, res, m_min, m_max):
        res_norm = (res - m_min) / (m_max - m_min)
        res_norm = torch.clamp(res_norm, 0, 1)
        return res_norm

    def log_normalize(self, res, m_min, m_max):
        res_clamped = torch.clamp(res, min=m_min, max=m_max)
        log_res = torch.log10(res_clamped)
        log_m_min = torch.log10(torch.tensor([m_min], device=res.device))
        log_m_max = torch.log10(torch.tensor([m_max], device=res.device))
        res_norm = (log_res - log_m_min) / (log_m_max - log_m_min)
        res_norm = torch.clamp(res_norm, 0, 1)
        return res_norm
    
####################################################################################################
## MATHEMATICAL TOOLS
####################################################################################################

def SkewMatrix(r):
    S = torch.zeros((*r.shape[:-1], 3, 3), device=r.device)
    S[..., 0, 1] = -r[..., 2]
    S[..., 0, 2] = r[..., 1]
    S[..., 1, 0] = r[..., 2]
    S[..., 1, 2] = -r[..., 0]
    S[..., 2, 0] = -r[..., 1]
    S[..., 2, 1] = r[..., 0]
    return S

def B_i(B, N, device):
    """ Wrench Basis B_i: Point with Friction (Coulomb Friction) (B, N, 6, 3)"""
    B_i = torch.zeros((B, N, 6, 3), device=device)      # (B, N, 6, 3)
    B_i[:, :, :3, :3] = torch.eye(3, device=device)
    return B_i

def R_i(n_i):
    """Contact Frame rotation matrix with z-axis as n_i.
    Args:
        n_i: (B, N, 3) tensor of normal vectors
    Returns:
        R_i: (B, N, 3, 3) rotation matrices, where R_i[..., :, 2] == n_i
    """
    B, N, _ = n_i.shape
    # Create arbitrary vector not parallel to normal
    arbitrary = torch.tensor([1., 0., 0.], device=n_i.device).expand_as(n_i).clone()
    parallel_mask = torch.abs(torch.sum(n_i * arbitrary, dim=2)) > 0.9
    arbitrary[parallel_mask] = torch.tensor([0., 1., 0.], device=n_i.device)
    # First tangent vector (cross product)
    t_i = torch.cross(arbitrary, n_i, dim=2)
    t_i = t_i / (torch.norm(t_i, dim=2, keepdim=True) + 1e-8)
    # Second tangent vector (cross product of normal and first tangent)
    o_i = torch.cross(n_i, t_i, dim=2)
    o_i = o_i / (torch.norm(o_i, dim=2, keepdim=True) + 1e-8)
    # Rotation matrices R_i of the i-th contact frame c_i => (B, N, 3, 3)
    R_i = torch.stack([t_i, o_i, n_i], dim=2).transpose(-2, -1)
    return R_i

def A_d_T(n_i, p_i):
    """ Adjoint transformation matrix A_d_T (B, N, 6, 6) """
    Ad = BlockDiag_R_i(n_i)                                        # (B, N, 6, 6)
    Ad[:, :, 3:, :3] = SkewMatrix(p_i) @ Ad[:, :, 0:3, 0:3]
    return Ad

def BlockDiag_R_i(n_i):
    """ Object Twist relative to contact frame """
    B, N, _ = n_i.shape
    Ri = R_i(n_i)  # (B, N, 3, 3)
    # Block diagonal rotation matrices BlockDiag_R_i => (B, N, 6, 6)
    BlockDiag_R_i = torch.zeros((B, N, 6, 6), device=n_i.device)
    BlockDiag_R_i[:, :, :3, :3] = Ri
    BlockDiag_R_i[:, :, 3:, 3:] = Ri
    return BlockDiag_R_i

#####################################################################################################
## GRASP MATRIX
#####################################################################################################

def grasp_matrix(contact_points_normals_labels):
    """ Compute the object wrench for a given contact points.
    Args:
        contact_points_normals_labels: (B, N, 7) tensor of contact points, normals and labels
    Returns:
        G: (B, 6, 3*N) Grasp matrix
    """
    assert contact_points_normals_labels.shape[2] == 7, "contact_points_normals_labels should have shape (B, N, 7)"
    B, N, _ = contact_points_normals_labels.shape

    # Split contact points and normals (normalize normals)
    c_i, n_i, _ = torch.split(contact_points_normals_labels, [3, 3, 1], dim=2)  # (B, N, 3), (B, N, 3), (B, N, 1)
    n_i = n_i / (torch.norm(n_i, dim=2, keepdim=True) + 1e-8)

    # Contact Points relative to center of mass of contact points
    o = (c_i.min(dim=1, keepdim=True).values + c_i.max(dim=1, keepdim=True).values) / 2.0
    p_i = c_i - o
    
    # Normalize torques by a characteristic length: maximum distance from contact point to object center
    r = torch.norm(p_i, dim=2, keepdim=True).max(dim=1, keepdim=True).values
    r = torch.clamp(r, min=1e-8)
    p_i = p_i / r

    G_i = A_d_T(n_i, p_i) @ B_i(B, N, c_i.device)                                                 # (B, N, 6, 3)
    G = G_i.transpose(1, 2).reshape(B, 6, 3*N)                      # (B, 6, 3N)

    return G

def grasp_matrix_with_object_center(contact_points_normals_labels, object_center):
    """ Compute the object wrench for a given contact points and object center.
    Args:
        contact_points_normals_labels: (B, N, 6+) tensor of contact points and normals
        object_center: (B, 3) tensor of object center points
    Returns:
        G: (B, 6, 3*N) Grasp matrix
    """
    assert contact_points_normals_labels.shape[2] >= 6, "contact_points_normals_labels should have shape (B, N, 6)"
    B, N, _ = contact_points_normals_labels.shape

    # Mask of valid contact points, i.e. not all zeros
    is_valid_contact = (torch.abs(contact_points_normals_labels[:, :, :6]).sum(dim=2) > 1e-6)   # (B, N)

    # Split contact points and normals (normalize normals)
    c_i, n_i = contact_points_normals_labels[:, :, :3], contact_points_normals_labels[:, :, 3:6]
    n_i = n_i / (torch.norm(n_i, dim=2, keepdim=True) + 1e-8)

    # Contact Points relative to center of mass of contact points
    # o = (c_i.min(dim=1, keepdim=True).values + c_i.max(dim=1, keepdim=True).values) / 2.0
    o = object_center.unsqueeze(1)   # (B, 1, 3)
    p_i = c_i - o
    
    # Normalize torques by a characteristic length: maximum distance from contact point to object center
    p_i_dist = torch.norm(p_i, dim=2, keepdim=True)
    p_i_dist[~is_valid_contact] = 0.0
    r = p_i_dist.max(dim=1, keepdim=True).values
    r = torch.clamp(r, min=1e-8)
    p_i = p_i / r

    G_i = A_d_T(n_i, p_i) @ B_i(B, N, c_i.device)                                                 # (B, N, 6, 3)

    # Mask out invalid contact points
    mask_expanded = is_valid_contact.view(B, N, 1, 1).float()
    G_i = G_i * mask_expanded

    G = G_i.transpose(1, 2).reshape(B, 6, 3*N)                      # (B, 6, 3N)

    return G

def friction_cone(B, N, mu=0.3, num_discretizations=8, device='cuda'):
    """ Discretize the friction cone into num_discretizations vectors.
    Args:
        mu: friction coefficient
        num_discretizations: number of vectors to discretize the friction cone
        device: device to create the tensors on
    Returns:
        F_fc: (B, 3N, num_discretizations*N) Friction cone vectors
    """ 
    theta = torch.linspace(0, 2 * torch.pi, steps=num_discretizations, device=device, dtype=torch.float32)
    friction_cone = torch.stack([
        mu * torch.cos(theta),
        mu * torch.sin(theta),
        torch.ones_like(theta)
    ], dim=0)
    friction_cone = friction_cone / (torch.norm(friction_cone, dim=0, keepdim=True) + 1e-8)         # (3, num_discretizations)
    # Repeat friction cone for each contact point
    F_fc = torch.block_diag(*[friction_cone for _ in range(N)]).unsqueeze(0).repeat(B, 1, 1)        # (B, 3N, num_discretizations*N)
    return F_fc

def wrench_space(G, mu=0.3):
    """ Compute the Wrench Space given the Grasp Matrix G and friction coefficient mu.
    Args:
        G: (B, 6, 3*N) Grasp matrix
    Returns:
        W: (B, 6, num_discretizations*N) Wrench Space
    """ 
    F_fc = friction_cone(B=G.shape[0], N=G.shape[2]//3, mu=mu, device=G.device)
    # Compute Wrench Space
    W = G @ F_fc    # (B, 6, num_discretizations*N)
    return W

####################################################################################################
## JACOBIAN MATRIX
####################################################################################################

def jacobian(hand_model, q, frame_X_dict):
    """
    Calculate Jacobian (dX/dq) of all frames

    Notation: (similar as https://manipulation.csail.mit.edu/pick.html#monogram)
        J: jacobian, X: transform, R: rotation, p: position, v: velocity, w: angular velocity
        <>_BA_C: <X, R, p, w> of frame A measured from frame B expressed in frame C
        W: world frame, J: joint frame, F: link frame

    :param hand_model: HandModel object
    :param q: (6 + DOF,) or (B, 6 + DOF), joint values (euler representation)
    :param frame_X_dict: Dictionary of frame transforms {frame_name: Transform3d()}
    :return: Jacobian: {frame_name: (B, 6, num_joints)}
    """
    jacobian_dict = {}

    q = torch.atleast_2d(q)
    batch_size = q.shape[0]
    joint_names = hand_model.robot.get_joint_parameter_names()
    num_joints = len(joint_names)
    joint_name2idx = {name: idx for idx, name in enumerate(joint_names)}

    frames = [hand_model.robot.find_frame(name) for name in hand_model.robot.get_joint_parent_frame_names()]
    idx = lambda frame: joint_name2idx[frame.joint.name]

    transfer_X = {}
    for frame in frames:
        q_frame = q[:, idx(frame)]
        # if frame.joint.joint_type == 'prismatic':
        q_frame = q_frame.unsqueeze(-1)
        transfer_X[idx(frame)] = frame.get_transform(q_frame).get_matrix()

    frame_X_dict = {f: frame_X_dict[f] for f in frame_X_dict if f in hand_model.link_names}

    for frame_name, frame_X in frame_X_dict.items():
        jacobian = torch.zeros((batch_size, 6, num_joints), dtype=hand_model.robot.dtype, device=hand_model.robot.device)

        R_WF = frame_X.get_matrix()[:, :3, :3]
        X_JF = torch.eye(4, dtype=hand_model.robot.dtype, device=hand_model.robot.device).repeat(batch_size, 1, 1)
        for frame_idx in reversed(hand_model.robot.parents_indices[hand_model.robot.frame_to_idx[frame_name + '_frame']].tolist()):
            frame = hand_model.robot.find_frame(hand_model.robot.idx_to_frame[frame_idx])
            joint = frame.joint
            if joint.joint_type == 'fixed':
                if joint.offset is not None:
                    X_JF = joint.offset.get_matrix() @ X_JF
                continue

            R_FJ = X_JF[:, :3, :3].mT
            R_WJ = R_WF @ R_FJ
            p_JF_J = X_JF[:, :3, 3][:, :, None]
            w_WJ_J = joint.axis[None, :, None].repeat(batch_size, 1, 1)
            if joint.joint_type == 'revolute':
                jacobian_v = R_WJ @ torch.cross(w_WJ_J, p_JF_J, dim=1)
                jacobian_w = R_WJ @ w_WJ_J
            elif joint.joint_type == 'prismatic':
                jacobian_v = R_WJ @ w_WJ_J
                jacobian_w = torch.zeros([batch_size, 3, 1], dtype=jacobian_v.dtype, device=jacobian_v.device)
            else:
                raise NotImplementedError(f"Unknown joint_type: {joint.joint_type}")

            joint_idx = joint_name2idx[joint.name]
            X_JF = transfer_X[joint_idx] @ X_JF
            jacobian[:, :, joint_idx] = torch.cat([jacobian_v[..., 0], jacobian_w[..., 0]], dim=1)

        jacobian_dict[frame_name] = jacobian
    return jacobian_dict

def jacobian_matrix(hand_model, q, contact_points_normals_labels):
    """ Compute the Jacobian matrix for a given robotic hand configuration and contact points.
    Args:
        hand_model: HandModel object
        q: (B, n_joints) tensor of joint angles
        contact_points_normals: (B, N, 7) tensor of contact points, normals and labels
    Returns:
        J: (B, 3*N, num_links) Jacobian matrix. num_links if the number of links used for the grasp.
    """

    assert q.shape[0] == contact_points_normals_labels.shape[0], "Batch size of q and contact_points_normals_labels should be the same"
    assert contact_points_normals_labels.shape[2] == 7, "contact_points_normals_labels should have shape (B, N, 7)"
    B, N, _ = contact_points_normals_labels.shape

    # Split contact points and normals (normalize normals)
    c_i, n_i, l_i = torch.split(contact_points_normals_labels, [3, 3, 1], dim=2)  # (B, N, 3), (B, N, 3), (B, N, 1)
    n_i = n_i / (torch.norm(n_i, dim=2, keepdim=True) + 1e-8)

    # Associate each contact point to a link name
    l_i_long = l_i.squeeze(-1).long()
    target_link_indices = hand_model.get_link_indices_from_labels(l_i_long) # (B, N)

    if hand_model.link_idx_to_root_idx.device != c_i.device:
        hand_model.link_idx_to_root_idx = hand_model.link_idx_to_root_idx.to(c_i.device)

    # Get corresponding Root indices (B, N)
    target_root_indices = hand_model.link_idx_to_root_idx[target_link_indices] # (B, N)

    # Compute forward kinematics and Jacobian for all links
    forward_kinematics = hand_model.robot.forward_kinematics(q)
    J_dict = jacobian(hand_model, q, forward_kinematics)

    all_J_list = [J_dict[name] for name in hand_model.all_link_names]
    J_stack = torch.stack(all_J_list, dim=1) 
    
    # Stack Forward Kinematics Matrices: (B, Num_Links, 4, 4)
    all_fk_list = []
    for name in hand_model.all_link_names:
        mat = forward_kinematics[name].get_matrix()
        # Handle static palm case (if shape is (1, 4, 4) or (4, 4), expand to B)
        if mat.shape[0] == 1: 
             mat = mat.expand(B, 4, 4)
        elif mat.ndim == 2:
             mat = mat.unsqueeze(0).expand(B, 4, 4)
        all_fk_list.append(mat)
    s_i_stack = torch.stack(all_fk_list, dim=1)

    # Gather the specific matrices we need using indices
    batch_idx = torch.arange(B, device=c_i.device).unsqueeze(1).expand(-1, N) # (B, N)
    J_i = J_stack[batch_idx, target_link_indices]                                               # (B, N, 6, total_joints)
    s_i_mat = s_i_stack[batch_idx, target_root_indices]                                         # (B, N, 4, 4)

    s_i = s_i_mat[:, :, :3, 3]                                                                  # (B, N, 3)
    p_i = c_i - s_i                                                                             # (B, N, 3)

    J = B_i(B, N, c_i.device).transpose(-2, -1) @ A_d_T(n_i, p_i).transpose(-2, -1) @ J_i       # (B, N, 3, 6) x (B, N, 6, total_joints) => (B, N, 3, total_joints)
    J = J.reshape(B, N*3, -1)                                                                   # (B, 3*N, total_joints)
    return J

def hand_object_jacobian(G,J):
    """ Compute the hand-object Jacobian H = G^+ * J => dx = H * dq
    Args:
        G: (B, 6, 3*N) Grasp matrix
        J: (B, 3*N, num_joints) Jacobian matrix
    Returns:
        H: (B, 6, num_joints) Hand-object Jacobian
    """
    pinv_G_T = torch.linalg.pinv(G.transpose(-2, -1))   # (B, 3N, 6)
    H = pinv_G_T @ J                                     # (B, 6, num_joints)
    return H

#####################################################################################################
## METRICS: 
## - Force Closure Existence -> bool
## - Object Manipulability -> bool
#####################################################################################################

def can_be_force_closure(G_FC):
    """ Check if the grasp is force closure: rank(G(FC)) == 6
    Args:
        G_FC: (B, 6, 3*N) Grasp matrix for force closure
    Returns:
        is_force_closure: (B,) boolean tensor indicating if the grasp is force closure
    """
    rank_G_FC = torch.linalg.matrix_rank(G_FC)
    return torch.where(rank_G_FC == 6, True, False)

def how_manipulable(H):
    """ Compute the rank of the hand-object Jacobian H, indicating how manipulable the object is.
    Args:
        H: (B, 6, num_joints) Hand-object Jacobian
    Returns:
        rank_H: (B,) tensor of ranks
    """
    rank_H = torch.linalg.matrix_rank(H)

    return rank_H, torch.where(rank_H > 0, True, False)

#####################################################################################################
## METRICS: Related to Contact Points (G Matrix)
#####################################################################################################

def minimum_singular_value(G):
    """ Compute the minimum singular value of the grasp matrix G.
    Args:
        G: (B, 6, 3*N) Grasp matrix for force closure
    Returns:
        min_singular_value: (B,) tensor of minimum singular values
    """
    sv = torch.linalg.svdvals(G)   # (B, min(m, n))
    min_singular_value = sv.min(dim=1).values  # (B,)
    return min_singular_value

def volume_ellipsoid_grasp_space(G):
    """ Compute the volume of the ellipsoid in the Wrench Space defined by the grasp matrix G.
    Args:
        G: (B, 6, 3*N) Grasp matrix
    Returns:
        volumes: (B,) tensor of ellipsoid volumes
    """
    return torch.sqrt(torch.linalg.det(G @ G.transpose(-2, -1)))

def grasp_isotropy_index(G):
    """ Compute the grasp isotropy index: ratio of minimum to maximum singular value of the grasp matrix G.
    Args:
        G: (B, 6, 3*N) Grasp matrix
    Returns:
        isotropy_index: (B,) tensor of isotropy indices
    """
    sv = torch.linalg.svdvals(G)   # (B, min(m, n))
    isotropy_index = sv.min(dim=1).values / (sv.max(dim=1).values + 1e-12)  # (B,)
    return isotropy_index

def volume_ellipsoid_wrench_space(W):
    """ Compute the volume of the ellipsoid in the Wrench Space defined by the Wrench Space W.
    Args:
        W: (B, 6, num_friction_vectors*N) Wrench Space
    Returns:
        volumes: (B,) tensor of ellipsoid volumes
    """
    covariance_matrix = W @ W.transpose(-2, -1)  # (B, 6, 6)
    # return torch.prod(torch.sqrt(torch.linalg.svdvals(covariance_matrix) + 1e-18), dim=1)
    # (prod(sqrt(S)))^(1/6) = exp(mean(log(sqrt(S)))) = exp(0.5 * mean(log(S)))
    log_S = torch.log(torch.linalg.svdvals(covariance_matrix) + 1e-18)
    return torch.exp(0.5 * torch.mean(log_S, dim=1))

def volume_wrench_space(W):
    """ Compute the volume of the convex hull of the Wrench Space W.
    Args:
        W: (B, 6, num_friction_vectors*N) Wrench Space
    Returns:
        volumes: (B,) tensor of convex hull volumes
    """
    W = W.permute(0, 2, 1)
    hull = None
    volumes = torch.zeros(W.shape[0], dtype=torch.float32, device=W.device)
    for b in range(W.shape[0]):
        W_batch = W[b]  # (num_friction_vectors*N, 6)
        try:
            hull = ConvexHull(W_batch.cpu().numpy())
            volumes[b] = hull.volume
        except:
            continue
    return volumes, hull

def epsilon_quality_measure(W):
    """ Compute the maximum radius of a sphere (centered at the origin) within the convex hull of the Wrench Space W.
    Args:
        W: (B, 6, num_friction_vectors*N) Wrench Space
    Returns:
        volumes: (B,) tensor of convex hull volumes
    """
    W = W.permute(0, 2, 1)

    hull = None
    radius = torch.zeros(W.shape[0], dtype=torch.float32, device=W.device)
    for b in range(W.shape[0]):
        W_batch = W[b]  # (num_friction_vectors*N, 6)

        try:
            hull = ConvexHull(W_batch.cpu().numpy())
            # A point p is inside if dot(normal, p) + d <= 0. Here p=0: d <= 0.
            origin_inside = all(hull.equations[i, -1] <= np.finfo(np.float32).eps 
                            for i in range(len(hull.equations)))
            if origin_inside:
                # Compute the signed distances from the origin to each facet
                distances = torch.from_numpy(np.abs(hull.equations[:, -1]) / np.linalg.norm(hull.equations[:, :-1], axis=1)).to(torch.float32).to(W.device)
                radius[b] = torch.min(distances)
        except:
            continue

    return radius, hull

#####################################################################################################
## METRICS: Related to Hand Configuration (H Matrix)
#####################################################################################################

def distance_singular_configuration(H):
    """ Compute the distance to the nearest singular configuration.
    Args:
        H: (B, 6, num_joints) Hand-object Jacobian
    Returns:
        dist_sing_config: (B,) tensor of minimum singular values
    """
    sv = torch.linalg.svdvals(H)   # (B, min(m, n))
    dist_sing_config = sv.min(dim=1).values  # (B,)
    return dist_sing_config

def uniformity_transformation(H):
    """ Compute the uniformity of the hand-object Jacobian H: ratio of minimum to maximum singular value.
    Args:
        H: (B, 6, num_joints) Hand-object Jacobian
    Returns:
        uniformity: (B,) tensor of uniformity measures
    """
    sv = torch.linalg.svdvals(H)   # (B, min(m, n))
    uniformity = sv.min(dim=1).values / (sv.max(dim=1).values + 1e-12)  # (B,)
    return uniformity

def manipulability_ellipsoid(H):
    """ Compute the manipulability ellipsoid of the hand configuration: sqrt(det(H * H^T))
    Args:
        H: (B, 6, num_joints) Hand-object Jacobian
    Returns:
        manipulability: (B,) tensor of manipulability measures
    """
    HH_T = H @ H.transpose(-2, -1)    # (B, 6, 6)

    det_HH_T = torch.linalg.det(HH_T + 1e-6 * torch.eye(HH_T.shape[-1], device=HH_T.device).unsqueeze(0))  # (B,)
    manipulability = torch.sqrt(torch.clamp(det_HH_T, min=0.0))
    return manipulability

#####################################################################################################
## METRICS: Wrapper to compute all metrics
#####################################################################################################

def grasp_metrics(G, J):
    """ Compute various grasp metrics given the grasp matrix G and hand-object Jacobian J. 
    Args:
        G ((B, 6, 3*N)): Grasp matrix
        J ((B, 3*N, num_joints)): Jacobian matrix
    Returns:
        metrics (dictionary of grasp metrics):  \\
        can_be_force_closure, rank_H, is_manipulable, \\
        min_singular_value, volume_ellipsoid_grasp_space, grasp_isotropy_index, \\
        volume_wrench_space, epsilon_quality_measure, \\
        distance_singular_configuration, uniformity_transformation, manipulability_ellipsoid
        
    """
    G_FC = wrench_space(G)
    H = hand_object_jacobian(G, J)

    can_be_fc = can_be_force_closure(G_FC)
    rank_H, is_manipulable = how_manipulable(H)

    if not can_be_fc.sum() == 0:
        min_sing_val = minimum_singular_value(G)
        volume_ws = volume_ellipsoid_grasp_space(G)
        isotropy = grasp_isotropy_index(G)
        vol_convex_hull, _ = volume_wrench_space(G_FC)
        epsilon_quality, _ = epsilon_quality_measure(G_FC)

        dist_sing_config = distance_singular_configuration(H)
        uniformity = uniformity_transformation(H)
        manipulability = manipulability_ellipsoid(H)
    else:
        min_sing_val = torch.zeros(G.shape[0], device=G.device)
        volume_ws = torch.zeros(G.shape[0], device=G.device)
        isotropy = torch.zeros(G.shape[0], device=G.device)
        vol_convex_hull = torch.zeros(G.shape[0], device=G.device)
        epsilon_quality = torch.zeros(G.shape[0], device=G.device)
        dist_sing_config = torch.zeros(G.shape[0], device=G.device)
        uniformity = torch.zeros(G.shape[0], device=G.device)
        manipulability = torch.zeros(G.shape[0], device=G.device)
    

    metrics = {
        "can_be_force_closure": can_be_fc,
        "rank_H": rank_H,
        "is_manipulable": is_manipulable,
        "min_singular_value": min_sing_val,
        "volume_ellipsoid_grasp_space": volume_ws,
        "grasp_isotropy_index": isotropy,
        "volume_wrench_space": vol_convex_hull,
        "epsilon_quality_measure": epsilon_quality,
        "distance_singular_configuration": dist_sing_config,
        "uniformity_transformation": uniformity,
        "manipulability_ellipsoid": manipulability
    }

    return metrics
