import torch
import torch.nn.functional as F
import time
from tqdm import tqdm

from utils.get_models import get_handmodel
from utils_model.HandModel import HandModel
from utils_data.custom_bps import compute_aligned_dist_v2
from utils.constants import ROBOTS_GRASPS_LABELS, ROBOTS_PALM_LABELS

class GripperOpt:
    """
    Run the optimization for the gripper. Uses Adam optimizer on q based on an energy function.
    """
    def __init__(self, cfg, logger=None):

        self.device = cfg['device']
        self.learning_rate = float(cfg['learning_rate'])
        self.w_pen = float(cfg['W_PENETRATION'])
        self.w_spen = float(cfg['W_SELF_PENETRATION'])
        self.w_joints = float(cfg['W_JOINTS'])
        self.w_type = float(cfg['W_TYPE'])
        self.max_iter = cfg['max_iterations']
        self.num_clones = cfg['num_clones']
        self.dist_type = cfg['distance_type']
        assert self.dist_type in ['euclidean', 'aligned'], "Distance type in config file must be either 'euclidean' or 'aligned'."

        self.batch_size = None
        self.batch_original_size = None
        self.robot_name = None
        self.object_name = None
        self.asked_type_idx = None
        self.object_pcd = None
        self.object_normals = None
        self.target_contact_points = None
        self.labels = None
        self.q_start = None

        self.has_reset = False

        self.q_current = None
        self.compute_energy = None
        self.optimizer = None
        self.scheduler = None
        self.energy = None
        self.global_step = None
        self.target_points = None
        self.target_labels = None
        self.time = 0

        self.logger = logger


    def reset(self, robot_name, object_name, asked_type_idx=None, object_pcd=None, object_normals=None, contact_points=None, labels=None):
        """Reset and prepare GripperOpt state.

        Args:
            robot_name (str): Name of the robot/gripper.
            object_name (str): Name of the object.
            asked_type_idx (torch.Tensor): Indices of the desired grasp types. (B,)
            object_pcd (B, N, 3): Object point cloud.
            object_normals (B, N, 3): Object normals corresponding to the point cloud.
            contact_points (B, M, 3+): Target contact points for the gripper to reach.
            labels (B, M): Labels indicating which contact points belong to which finger.
            q_start (B, n_joints): Initial joint values for the gripper.
        """
        self.batch_original_size = contact_points.shape[0]

        self.robot_name = robot_name
        self.object_name = object_name
        self.asked_type_idx = asked_type_idx        # (B,)
        self.object_pcd = object_pcd.unsqueeze(0) if object_pcd is not None and object_pcd.dim() == 2 else object_pcd                                   # (B, 2048, 3)
        self.object_normals = object_normals.unsqueeze(0) if object_normals is not None and object_normals.dim() == 2 else object_normals               # (B, 2048, 3)
        self.target_contact_points = contact_points.unsqueeze(0) if contact_points is not None and contact_points.dim() == 2 else contact_points        # (B, N, 3)
        if self.target_contact_points.shape[-1] > 3:
            self.target_contact_points = self.target_contact_points[:, :, :3]
        self.labels = labels.unsqueeze(0) if labels is not None and labels.dim() == 1 else labels                                                       # (B, N)

        self.asked_type_idx = self.asked_type_idx.repeat_interleave(self.num_clones, dim=0)                 # (B * num_clones,)
        self.object_pcd = self.object_pcd.repeat_interleave(self.num_clones, dim=0)                         # (B * num_clones, 2048, 3)
        self.object_normals = self.object_normals.repeat_interleave(self.num_clones, dim=0)                 # (B * num_clones, 2048, 3)
        self.target_contact_points = self.target_contact_points.repeat_interleave(self.num_clones, dim=0)   # (B * num_clones, N, 3)
        self.labels = self.labels.repeat_interleave(self.num_clones, dim=0)                                 # (B * num_clones, N)

        self.batch_size = self.target_contact_points.shape[0] # Number of batches

        self.hand_model: HandModel = get_handmodel(self.robot_name, batch_size=self.batch_size, hand_scale=1.0, device=self.device, num_points=512)
        self.q_upper_bound = self.hand_model.revolute_joints_q_upper.detach()  # (B, n_joints)
        self.q_lower_bound = self.hand_model.revolute_joints_q_lower.detach()  # (B, n_joints)
        self.q_start = self.hand_model.opti_start_pose.clone()[:, 9:]  # (B, n_joints)
        self.q_start[1] = self.hand_model.straight_pose.clone()[1, 9:]  # Fix second clone to straight pose
        # self.q_start = torch.rand_like(self.hand_model.straight_pose.clone()[:, 9:]) * (self.q_upper_bound - self.q_lower_bound) + self.q_lower_bound  # Random initialization within bounds

        # Precompute intra-finger mask for self-penetration energy
        if self.robot_name == 'allegro_right':
            # finger_ids = torch.arange(20, device=self.device) // 5              # Finger indices: 0-4 (LF), 5-9 (MF), 10-14 (IF), 15-19 (TH)
            finger_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], device=self.device)
        elif self.robot_name == 'shadowhand':
            finger_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4], device=self.device)
        else:
            raise ValueError(f"Unsupported robot name: {self.robot_name}")
            # Create mask: True for intra-finger pairs, False otherwise
        self.intra_finger_mask = (finger_ids.unsqueeze(0) == finger_ids.unsqueeze(1)).unsqueeze(0)      # (K-1, K-1)
        
        # Self-Penetration Margin
        if self.robot_name == 'allegro_right':
            self.spen_margin = 0.03
        elif self.robot_name == 'shadowhand':
            self.spen_margin = 0.019
        else:
            raise ValueError(f"Unsupported robot name: {self.robot_name}")

        # Type of Grasp conformity: Create mask for active contact points based on asked type => True if point belongs to asked type
        idx_2_grasp = {idx: grasp for idx, grasp in enumerate(ROBOTS_GRASPS_LABELS[robot_name].keys())}
        num_types = len(idx_2_grasp)
        max_label = max([max(v) for v in ROBOTS_GRASPS_LABELS[robot_name].values()]) + 1
        grasp_to_label_mask = torch.zeros((num_types, max_label), dtype=torch.bool, device=self.device)   # (Num Types, Total Zones)
        for idx, name in idx_2_grasp.items():
            labels = ROBOTS_GRASPS_LABELS[robot_name][name]
            grasp_to_label_mask[idx, labels] = True
        # Include entire links for each grasp type  
        link_label_adj = F.one_hot(self.hand_model.label_to_link_idx.to(self.device).long()).float()        # (Total Zones, Num Links)
        active_links_mask = (grasp_to_label_mask.float() @ link_label_adj) > 0                              # (Num Types, Num Links)
        grasp_to_label_mask = (active_links_mask.float() @ link_label_adj.T) > 0                            # (Num Types, Total Zones)
        # Exclude palm labels from active points
        palm_labels = [p for p in ROBOTS_PALM_LABELS[robot_name] if p < grasp_to_label_mask.shape[1]]
        grasp_to_label_mask[:, palm_labels] = False   
        # Get active points
        self.batch_active_zones = grasp_to_label_mask[self.asked_type_idx]                                                                  # (B, Total Zones)
        handprint_labels = self.hand_model.get_handprint_points(self.hand_model.straight_pose, label=True)[:, :, -1].long().clone()     # (B, 2048)
        self.active_points_mask = torch.gather(self.batch_active_zones, 1, handprint_labels)                                                # (B, 2048)

        # Init Energy Optimization
        self.global_step = 0
        if self.dist_type == 'euclidean':
            self.compute_energy = self.compute_energy_euclidean_dist
        elif self.dist_type == 'aligned':
            self.compute_energy = self.compute_energy_aligned_dist

        self.q_current = self.q_start.clone().to(self.device)
        self.q_current = torch.nn.Parameter(self.q_current, requires_grad=True)  # Make it a parameter for optimization
        self.optimizer = torch.optim.Adam([self.q_current], lr=self.learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=15, threshold=0.0005, verbose=True)

        self.hand_model.update_kinematics_no_base(jv=self.q_current)  # Update hand model with initial joint values

        self.has_reset = True

    def get_current_q(self):
        """
        Get the current joint values.
        """
        return self.q_current

    def update_hand(self):
        """
        Update derived geometric points from the hand model:
        1. Hand Surface Points: self.hand_surface_points_
        2. Zones Centers: self.zones_centers
        3. Keypoints without base: self.keypoints_no_base
        """
        self.hand_model.update_kinematics_no_base(jv=self.q_current)

        # Hand Surface Points
        surface_data = self.hand_model.get_handprint_points(label=True)
        self.hand_surface_points_ = surface_data[..., :3]
        self.hand_surface_labels = surface_data[..., -1].long()     # (B, N)

        # Keypoints without base
        self.keypoints_no_base = self.hand_model.get_keypoints_differentiable(with_base=False)

    def compute_energy_euclidean_dist(self, step):
        """
        Compute the energy function based on the current gripper pose.
        """
        # Energy: EUCLIDEAN DISTANCE
        dist_matrix = torch.cdist(self.target_contact_points, self.hand_surface_points_, p=2)                                       # (B, M, N)
        label_match_mask = (self.labels.unsqueeze(-1) == self.hand_surface_labels.unsqueeze(1))                                     # (B, M, N)
        masked_dist_matrix = dist_matrix.clone()
        masked_dist_matrix[~label_match_mask] = float('inf')
        min_dists, _ = torch.min(masked_dist_matrix, dim=-1)                                                                        # (B, M)
        min_dists = torch.where(torch.isinf(min_dists), torch.zeros_like(min_dists), min_dists)

        # Mask distances where self.labels == palm and filter out non-active points
        active_targets_mask = torch.gather(self.batch_active_zones.float(), 1, self.labels.long())                                  # (B, N)
        if (self.robot_name == 'shadowhand') or (self.robot_name == 'allegro_right'):
            palm_mask = (self.labels > 7).float()
            active_targets_mask = active_targets_mask * palm_mask
        else:
            raise ValueError(f"Unsupported robot name: {self.robot_name}")  
        
        min_dists = min_dists * active_targets_mask
        energy_dist = min_dists.mean(dim=1)  # (B,)

        # Energy: HAND-OBJECT PENETRATION
            # Compute distances between hand surface points and object points
        hand_object_distances = torch.cdist(self.hand_surface_points_, self.object_pcd)                                 # (B, 2048, 2048)
        hand_object_distances, hand_object_indices = torch.min(hand_object_distances, dim=2)                            # (B, 2048), (B, 2048)
            # Get the closest object points and normals per hand point
        hand_object_indices_expanded = hand_object_indices.unsqueeze(-1).expand(-1, -1, 3)
        hand_object_points = torch.gather(self.object_pcd, 1, hand_object_indices_expanded)                             # (B, 2048, 3)
        hand_object_normals = torch.gather(self.object_normals, 1, hand_object_indices_expanded)                        # (B, 2048, 3)
        hand_object_signs = torch.sum((hand_object_points - self.hand_surface_points_) * hand_object_normals, dim=-1)   # (B, 2048)
        hand_object_signs = (hand_object_signs > 0.0).float()                                                           # (B, 2048)
            # Compute penetration energy
        energy_pen = torch.mean(hand_object_signs * hand_object_distances, dim=1)                                       # (B,)

        # Energy: GRASP TYPE CONFORMITY
        margin = 0.005
        hand_points_non_type_dist = F.relu(margin - torch.abs(hand_object_distances))
        energy_type = (hand_points_non_type_dist * (~self.active_points_mask).float()).sum(dim=1)

        # Energy: SELF-PENETRATION
        distances_keypoints = torch.cdist(self.keypoints_no_base, self.keypoints_no_base, p=2)                                  # (B, K-1, K-1)
        distances_keypoints = torch.where(self.intra_finger_mask, torch.ones_like(distances_keypoints), distances_keypoints)    # (B, K-1, K-1)
        s_norm = F.relu(self.spen_margin - distances_keypoints)
        energy_spen = s_norm.sum(dim=(1, 2))                                                                                    # (B,)

        # Energy: JOINTS => upper | lower
        z_norm = F.relu(self.q_current - self.q_upper_bound) + F.relu(self.q_lower_bound - self.q_current)
        energy_joints = z_norm.sum(dim=1)

        # Energy: TOTAL
        energy = energy_dist 
        energy += self.w_joints * energy_joints
        energy += self.w_pen * energy_pen
        if step >= self.max_iter // 3:
            energy += self.w_spen * energy_spen
            energy += self.w_type * energy_type

        self.energy = energy

        if self.logger is not None:
            self.logger.log_metrics({
                'energy': energy.mean().item(),
                'energy_dist': energy_dist.mean().item(),
                'energy_pen': self.w_pen * energy_pen.mean().item(),
                'energy_spen': self.w_spen * energy_spen.mean().item(),
                'energy_joints': self.w_joints * energy_joints.mean().item(),
                'energy_type': self.w_type * energy_type.mean().item(),
            }, step=self.global_step)

        return energy

    def compute_energy_aligned_dist(self, step):
        """
        Compute the energy function based on the current gripper pose.
        """        
        # Energy: ALIGN DISTANCE
            # Project target contact points onto object surface normals to get aligned target points
        d_idx = torch.cdist(self.target_contact_points, self.object_pcd).min(dim=-1).indices  # (B, N)
        target_normals = -self.object_normals[torch.arange(d_idx.shape[0]).unsqueeze(1), d_idx]  # (B, N, 3)
        target_points = torch.cat([self.target_contact_points, target_normals], dim=-1)  # (B, N, 6)
            # Compute aligned distance matrix between target points and hand surface points, then mask by labels
        aligned_dist_matrix = compute_aligned_dist_v2(X=target_points, Y=self.hand_surface_points_, gamma=2.0, delta=0.1, use_sqrt=True, return_distances=True)  # (B, M, N)
        label_match_mask = (self.labels.unsqueeze(-1) == self.hand_surface_labels.unsqueeze(1))                                     # (B, M, N)
        masked_dist_matrix = aligned_dist_matrix.clone()
        masked_dist_matrix[~label_match_mask] = float('inf')
        min_dists, _ = torch.min(masked_dist_matrix, dim=-1)                                                                        # (B, M)
        min_dists = torch.where(torch.isinf(min_dists), torch.zeros_like(min_dists), min_dists)

        # Mask distances where self.labels == palm and filter out non-active points
        active_targets_mask = torch.gather(self.batch_active_zones.float(), 1, self.labels.long())                                  # (B, N)
        if (self.robot_name == 'shadowhand') or (self.robot_name == 'allegro_right'):
            palm_mask = (self.labels > 7).float()
            active_targets_mask = active_targets_mask * palm_mask
        else:
            raise ValueError(f"Unsupported robot name: {self.robot_name}")  

        min_dists = min_dists * active_targets_mask
        energy_dist = min_dists.mean(dim=1)                                                                             # (B,)

        # Energy: HAND-OBJECT PENETRATION
            # Compute distances between hand surface points and object points
        hand_object_distances = torch.cdist(self.hand_surface_points_, self.object_pcd)                                 # (B, 2048, 2048)
        hand_object_distances, hand_object_indices = torch.min(hand_object_distances, dim=2)                            # (B, 2048), (B, 2048)
            # Get the closest object points and normals per hand point
        hand_object_indices_expanded = hand_object_indices.unsqueeze(-1).expand(-1, -1, 3)
        hand_object_points = torch.gather(self.object_pcd, 1, hand_object_indices_expanded)                             # (B, 2048, 3)
        hand_object_normals = torch.gather(self.object_normals, 1, hand_object_indices_expanded)                        # (B, 2048, 3)
        hand_object_signs = torch.sum((hand_object_points - self.hand_surface_points_) * hand_object_normals, dim=-1)   # (B, 2048)
        hand_object_signs = (hand_object_signs > 0.0).float()                                                           # (B, 2048)
            # Compute penetration energy
        energy_pen = torch.mean(hand_object_signs * hand_object_distances, dim=1)                                       # (B,)

        # Energy: GRASP TYPE CONFORMITY
        margin = 0.005
        hand_points_non_type_dist = F.relu(margin - torch.abs(hand_object_distances))
        energy_type = (hand_points_non_type_dist * (~self.active_points_mask).float()).sum(dim=1)

        # Energy: SELF-PENETRATION
        distances_keypoints = torch.cdist(self.keypoints_no_base, self.keypoints_no_base, p=2)                                  # (B, K-1, K-1)
        distances_keypoints = torch.where(self.intra_finger_mask, torch.ones_like(distances_keypoints), distances_keypoints)    # (B, K-1, K-1)
        s_norm = F.relu(self.spen_margin - distances_keypoints)
        energy_spen = s_norm.sum(dim=(1, 2))                                                                                    # (B,)

        # Energy: JOINTS => upper | lower
        z_norm = F.relu(self.q_current - self.q_upper_bound) + F.relu(self.q_lower_bound - self.q_current)
        energy_joints = z_norm.sum(dim=1)

        # Energy: TOTAL
        energy = energy_dist 
        energy += self.w_joints * energy_joints
        energy += self.w_pen * energy_pen
        if step >= self.max_iter // 3:
            energy += self.w_spen * energy_spen
            energy += self.w_type * energy_type

        self.energy = energy

        if self.logger is not None:
            self.logger.log_metrics({
                'energy': energy.mean().item(),
                'energy_dist': energy_dist.mean().item(),
                'energy_pen': self.w_pen * energy_pen.mean().item(),
                'energy_spen': self.w_spen * energy_spen.mean().item(),
                'energy_joints': self.w_joints * energy_joints.mean().item(),
                'energy_type': self.w_type * energy_type.mean().item(),
            }, step=self.global_step)

        return energy

    def step(self, iteration):
        self.optimizer.zero_grad()
        self.update_hand()
        energy = self.compute_energy(iteration)
        energy_mean = energy.mean()
        energy_mean.backward()
        self.optimizer.step()
        self.global_step += 1

    def run(self, verbose=True):
        """
        Run the optimization process.
        Args:
            verbose (bool): Whether to display a progress bar.
        Returns:
            best_q (torch.Tensor): Optimized joint values for the gripper. (B, n_joints)
            best_energies (torch.Tensor): Corresponding energies for the optimized joint values. (B,)
        """
        if not self.has_reset:
            raise RuntimeError("GripperOpt has not been reset. Call reset() before run().")

        q_traj = []
        energy_per_iter = []
        time_per_step = []
        if verbose:
            with tqdm(total=self.max_iter, desc=f"[{self.robot_name}/{self.object_name}] Optimization - Energy: n/a (avg)") as pbar:
                for i in range(self.max_iter):
                    t0 = time.time()
                    self.step(i)
                    t1 = time.time()
                    time_per_step.append(t1 - t0)
                    with torch.no_grad():
                        q = self.get_current_q()
                        q_traj.append(q.clone().detach())
                        energy = self.energy.detach().cpu()
                        energy_per_iter.append(energy)
                        pbar.set_description(f"[{self.robot_name}/{self.object_name}] Optimization - Energy: {energy.mean():.4f} (avg)")
                        pbar.update(1)
        else:
            for i in range(self.max_iter):
                t0 = time.time()
                self.step(i)
                t1 = time.time()
                time_per_step.append(t1 - t0)
                with torch.no_grad():
                    q = self.get_current_q()
                    q_traj.append(q.clone().detach())
                    energy = self.energy.detach().cpu()
                    energy_per_iter.append(energy)
        self.time = sum(time_per_step) / len(time_per_step)
        # print(f"Optimization finished. Mean time per iter: {self.time:.4f} Energy: {energy.min():.4f} (min)")

        q_traj = torch.stack(q_traj, dim=0).transpose(0, 1)                     # (B, self.max_iter, num_joints)
        energy_per_iter = torch.stack(energy_per_iter, dim=0).transpose(0, 1)   # (B, self.max_iter)

        # Get the optimized gripper positions (last iteration)
        q_opti = q_traj[:, -1, :].view(self.batch_original_size, self.num_clones, -1)         # (B, num_clones, num_joints)
        energy_opti = energy_per_iter[:, -1].view(self.batch_original_size, self.num_clones)  # (B, num_clones)

        best_energies, best_clone_indices = torch.min(energy_opti, dim=1)                     # (B,), (B,)
        batch_indices = torch.arange(self.batch_original_size, device=q_opti.device)
        best_q = q_opti[batch_indices, best_clone_indices]

        return best_q, best_energies  # Return the best among the clones