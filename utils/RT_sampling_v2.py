# import open3d as o3d
import torch
# import kaolin
from utils.tools import rotate_pcd_6d, voxel_downsampling, compute_object_pose_6d, move_pcd_6d, compute_6d_gripper_pose
import trimesh
from scipy.spatial.transform import Rotation
import numpy as np

def check_collision_pytorch(points, vertices, faces, margin=0.0, chunk_size=500):
    """
    Checks if points are intersecting the mesh OR are too close to it.
    
    Args:
        points: (B, N, 3) or (P, 3) - Points to check
        vertices: (V, 3) - Mesh vertices
        faces: (F, 3) - Mesh face indices
        margin: float - Minimum allowed distance to the mesh surface
        chunk_size: int - Number of points to process at once to avoid OOM
        
    Returns:
        collide_mask: Boolean tensor same shape as input points (True = collision/too close)
    """
    # Standardize input to (Total_Points, 3)
    original_shape = points.shape
    points_flat = points.reshape(-1, 3)
    num_points = points_flat.shape[0]
    
    device = points.device
    vertices = vertices.to(device).float()
    faces = faces.to(device)
    
    # Pre-fetch face vertices: (F, 3, 3)
    # V0=(F,3), V1=(F,3), V2=(F,3)
    face_verts = vertices[faces] 
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    
    collide_results = []
    
    # Process in chunks to save memory
    for i in range(0, num_points, chunk_size):
        # Slice current chunk of points
        p_chunk = points_flat[i : i + chunk_size] # (M, 3)
        
        # PROXIMITY CHECK
        dists = torch.cdist(p_chunk, vertices)
        min_dists, _ = dists.min(dim=1)
        
        # Mark as colliding if too close
        is_too_close = min_dists < margin
        
        # INSIDE CHECK        
        A = v0[None, :, :] - p_chunk[:, None, :] 
        B = v1[None, :, :] - p_chunk[:, None, :]
        C = v2[None, :, :] - p_chunk[:, None, :]
        
        # Norms
        al = A.norm(dim=-1)
        bl = B.norm(dim=-1)
        cl = C.norm(dim=-1)
        
        # Solid Angle Formula
        # numerator = Det([A, B, C]) = dot(cross(A, B), C)
        numerator = torch.sum(torch.cross(A, B, dim=-1) * C, dim=-1)
        
        denominator = (al * bl * cl) + \
                      torch.sum(A * B, dim=-1) * cl + \
                      torch.sum(B * C, dim=-1) * al + \
                      torch.sum(C * A, dim=-1) * bl
        
        solid_angles = 2 * torch.atan2(numerator, denominator)
        
        # Sum over faces (axis 1)
        winding_number = solid_angles.sum(dim=1) / (4 * torch.pi)
        
        is_inside = torch.abs(winding_number) > 0.5
        
        # Combine: Collision if Inside OR Too Close
        chunk_collision = is_inside | is_too_close
        collide_results.append(chunk_collision)
        
    # Reassemble
    collide_flat = torch.cat(collide_results, dim=0)
    return collide_flat.reshape(original_shape[:-1]) # Return mask (B, N) or (P)

def check_sign_pytorch(vertices, faces, points):
    """
    A PyTorch alternative to kaolin.ops.mesh.check_sign
    points: (P, 3)
    vertices: (V, 3) - The actual coordinates of the vertices for each face
    faces: (F, 3) - The actual indices of the vertices for each face
    """
    face_vertices = vertices[faces, :]  # (F, 3, 3)
        # 1. Calculate vectors from points to face vertices
    # This can be memory intensive for large meshes; 
    A = face_vertices[None, :, 0] - points[:, None, :] # (P, F, 3)
    B = face_vertices[None, :, 1] - points[:, None, :]
    C = face_vertices[None, :, 2] - points[:, None, :]

    # 2. Compute the solid angle subtended by each triangle
    # Formula: tan(omega/2) = [a b c] / (abc + (a.b)c + (b.c)a + (c.a)b)
    al = A.norm(dim=-1)
    bl = B.norm(dim=-1)
    cl = C.norm(dim=-1)
    
    numerator = torch.sum(torch.cross(A, B, dim=-1) * C, dim=-1)
    denominator = (al * bl * cl) + torch.sum(A * B, dim=-1) * cl + \
                  torch.sum(B * C, dim=-1) * al + torch.sum(C * A, dim=-1) * bl
    
    solid_angles = 2 * torch.atan2(numerator, denominator)
    
    # 3. Sum solid angles and check against threshold (typically 2*pi or 4*pi)
    winding_number = solid_angles.sum(dim=-1) / (4 * torch.pi)
    
    # winding_number > 0.5 usually indicates 'inside'
    return winding_number > 0.5

def create_workspace_type(full_labeled_pcd, type_labels, voxel_size=0.005):
    """
    Creates a workspace point cloud for a specific grasp type by filtering points
    from a full labeled point cloud.

    Args:
        full_labeled_pcd (torch.Tensor): (N, 7) array of point cloud coordinates and labels.
        type_labels (torch.Tensor): (K,) array of integer labels corresponding to the desired grasp type.
        voxel_size (float): Size of the voxel grid for downsampling
    Returns:
        pcd (torch.Tensor): (M, 6) downsampled point cloud for the specified grasp type.
    """
    mask = torch.isin(full_labeled_pcd[:, 6], type_labels)
    pcd = full_labeled_pcd[mask]  # (M, 7)
    pcd, _ = voxel_downsampling(pcd, voxel_size=voxel_size)
    return pcd[:, :6]  # Return only XYZ and normals


def sample_object_poses(robot_name, collision_mesh, workspace, object_pcd_normals, total_samples=200):
    """
    Samples uniformly object poses within the given workspace.

    Args:
        :collision_mesh (Tuple[torch.Tensor, torch.Tensor]): The mesh of the robot's collision object (vertices: (V, 3), faces: (F, 3)).
        :workspace (torch.Tensor): (M, 6) The effective robot's workspace relative to the desired type.
        :object_pcd_normals (torch.Tensor): (N, 6) array of point cloud coordinates and normals.
        :total_samples (int): Number of poses to sample, max=10000.
    Returns:
        :object pcd (torch.Tensor): (total_samples, N, 3) A tensor of moved object point clouds.
        :object poses (torch.Tensor): (total_samples, 9) The corresponding object poses (translation + rot6d).
        :gripper poses (torch.Tensor): (total_samples, 9) The corresponding gripper poses (translation + rot6d).
        :workspace_batch (torch.Tensor): (batch_size, M, 3) The workspace repeated for the batch.
    """
    total_samples = min(total_samples, 10000)
    device = object_pcd_normals.device

    collision_vertices, collision_faces = collision_mesh
    if len(collision_vertices.shape) > 2:
        collision_vertices = collision_vertices[0]
    if len(collision_faces.shape) > 2:
        collision_faces = collision_faces[0]
    
    object_pcd_normals = object_pcd_normals.unsqueeze(0).repeat(total_samples, 1, 1)
    output_objects = torch.zeros_like(object_pcd_normals, device=device)
    output_objects_pose = torch.zeros((total_samples, 9), device=device)

    # Check collisions with hand and if object not too far from hand
    if robot_name == 'shadowhand':
        workspace_center = torch.tensor([0.01, 0.0, 0.1], device=device)
        palm_normal_axis, palm_normal_dir = 1, -1  # Y-axis, dir: +/- 1
        dist_threshold, z_threshold = 0.08, 0.07
    elif robot_name == 'allegro_right':
        workspace_center = torch.tensor([0.0, 0.025, 0.0], device=device)
        palm_normal_axis, palm_normal_dir = 0, 1  # X-axis, dir: +/- 1
        dist_threshold, z_threshold = 0.11, -0.01
    else:
        raise NotImplementedError(f"Robot {robot_name} not implemented for collision checking.")

    # Filter workspace points too far from center
    distances = torch.norm(workspace[:, :3] - workspace_center.unsqueeze(0), dim=-1)
    workspace = workspace[distances < dist_threshold]
    # Filter by "in front of hand"
    workspace = workspace[workspace[:, palm_normal_axis] * palm_normal_dir >= 0.0]
    # Filter by above palm height
    workspace = workspace[workspace[:, 2] >= z_threshold]
    # Batch workspace
    workspace_batch = workspace.unsqueeze(0).repeat(total_samples, 1, 1)  # (total_samples, M', 6)
    
    
    num_samples = 0
    while num_samples < total_samples:
        
        # Select Random Candidates
        batch_indices = torch.arange(total_samples, device=device)
        obj_indices = torch.randint(0, object_pcd_normals.shape[1], (total_samples,), device=device)
        ws_indices = torch.randint(0, workspace.shape[0], (total_samples,), device=device)
        
        # Get Candidate Workspace Points
        workspace_candidates_points = workspace_batch[batch_indices, ws_indices, :3] # (B, 3)

        # Determine Random Rotation Matrix
        random_rot6d = torch.randn((total_samples, 6), device=device)
        rotated_object, rotated_normals = rotate_pcd_6d(object_pcd_normals[:, :, :3], random_rot6d, object_pcd_normals[:, :, 3:])
        
        # Translate object
        candidate_p_rot = rotated_object[batch_indices, obj_indices]  # (B, 3)
        translation = workspace_candidates_points[:, :3] - candidate_p_rot      # (B, 3)
        moved_objects = rotated_object + translation.unsqueeze(1)  # (total_samples, N, 3)

        # Collision Check & Filtering
        batch_collision_mask = check_collision_pytorch(
            moved_objects, 
            collision_vertices, 
            collision_faces, 
            margin=0.01,
            chunk_size=500 # Adjust based on your VRAM
        )

        collide = batch_collision_mask.any(dim=1)  # (total_samples,) True if any point collides
        if torch.all(collide):
            continue
        
        batch_valid_samples = ~collide
        batch_valid_objects = moved_objects[batch_valid_samples]
        batch_valid_normals = rotated_normals[batch_valid_samples]
        batch_valid_translation = translation[batch_valid_samples]
        batch_valid_rot6d = random_rot6d[batch_valid_samples]
        
        n_valid = batch_valid_objects.shape[0]
        if num_samples + n_valid > total_samples:
            remaining = total_samples - num_samples
            output_objects[num_samples:] = torch.cat((batch_valid_objects[:remaining], batch_valid_normals[:remaining]), dim=-1)
            output_objects_pose[num_samples:, :3] = batch_valid_translation[:remaining]
            output_objects_pose[num_samples:, 3:] = batch_valid_rot6d[:remaining]
            break
        else:
            output_objects[num_samples:num_samples + n_valid] = torch.cat((batch_valid_objects, batch_valid_normals), dim=-1)
            output_objects_pose[num_samples:num_samples + n_valid, :3] = batch_valid_translation
            output_objects_pose[num_samples:num_samples + n_valid, 3:] = batch_valid_rot6d
        
        num_samples += n_valid

    gripper_poses_6d = compute_6d_gripper_pose(object_xyz=output_objects_pose[:, :3], object_rot=output_objects_pose[:, 3:], rot_type='rot6d')
    return output_objects, output_objects_pose, gripper_poses_6d, workspace_batch


def sample_object_poses_on_palm(robot_name, object_pcd_normals, num_samples=200):
    """
    Samples poses uniformly on the convex hull of a given point cloud.
    Each pose's x or y-axis is directed towards the hull face normal (inward).

    Args:
        :robot_name (str): The name of the robot (e.g., 'allegro_right', 'barrett', 'shadowhand').
        :object_pcd_normals (torch.Tensor): Nx6 array of point cloud coordinates and normals.
        :num_samples (int): Number of poses to sample.

    Returns:
        :object_pcd (torch.Tensor): (total_samples, N, 3) A tensor of moved object point clouds.
        :object_poses (torch.Tensor): (total_samples, 9) The corresponding object poses (translation + rot6d).
        :gripper_poses (torch.Tensor): (total_samples, 9) The corresponding gripper poses (translation + rot6d).
    """
    # total_samples = max(num_samples, 1000)
    total_samples = num_samples

    r = torch.full((total_samples,), 0.005, dtype=torch.float32)

    pcd_np = object_pcd_normals[:, :3].to(torch.float32).cpu().numpy()
    hull_mesh = trimesh.points.PointCloud(pcd_np).convex_hull
    samples_np, face_indices = trimesh.sample.sample_surface(hull_mesh, total_samples)
    normals_np = hull_mesh.face_normals[face_indices]
    sampled_points = torch.from_numpy(samples_np).float()
    sampled_points_normals = torch.from_numpy(normals_np).float()

    def get_arbitrary_axis(arbitrary, primary, idx, alt):
        arbitrary_axis = torch.tensor(arbitrary, dtype=torch.float32).repeat(total_samples, 1)
        mask = torch.abs(primary[:, idx]) >= 0.99
        arbitrary_axis[mask] = torch.tensor(alt, dtype=torch.float32)
        return arbitrary_axis

    # Vectorized frame creation
    if robot_name == 'allegro_right':
        primary_axis = -sampled_points_normals
        arbitrary = get_arbitrary_axis([0, 1, 0], primary_axis, 1, [1, 0, 0])
        y_axis = torch.cross(primary_axis, arbitrary, dim=1)
        y_axis /= torch.norm(y_axis, dim=1, keepdim=True)
        z_axis = torch.cross(primary_axis, y_axis, dim=1)
        z_axis /= torch.norm(z_axis, dim=1, keepdim=True)
        axes = (primary_axis, y_axis, z_axis)
        translation_dir = primary_axis
        offset = -r.unsqueeze(1) * translation_dir
        rot_axis = primary_axis
    elif robot_name == 'shadowhand':
        primary_axis = sampled_points_normals
        arbitrary = get_arbitrary_axis([1, 0, 0], primary_axis, 0, [0, 1, 0])
        x_axis = torch.cross(primary_axis, arbitrary, dim=1)
        x_axis /= torch.norm(x_axis, dim=1, keepdim=True)
        z_axis = torch.cross(x_axis, primary_axis, dim=1)
        z_axis /= torch.norm(z_axis, dim=1, keepdim=True)
        axes = (x_axis, primary_axis, z_axis)
        translation_dir = -primary_axis
        offset = -(r + 0.01).unsqueeze(1) * translation_dir
        rot_axis = primary_axis
    else:
        raise ValueError("Not Implemented for this Robot.")

    # Random rotation around primary axis
    angle = torch.rand(total_samples, 1) * 2 * np.pi
    sin_half, cos_half = torch.sin(angle / 2), torch.cos(angle / 2)
    quat = torch.cat([
        rot_axis * sin_half,    # (total_samples, 3)
        cos_half                # (total_samples, 1)
    ], dim=1)
    R_rand = torch.from_numpy(Rotation.from_quat(quat.numpy()).as_matrix()).float()

    R = torch.stack(axes, dim=2)
    R = torch.matmul(R_rand, R)

    frame_xyz = sampled_points + offset
    if robot_name == 'shadowhand':
        offset_point = torch.tensor([0.02, 0.0, 0.07], device=R.device)
        frame_xyz = frame_xyz - torch.matmul(R, offset_point)
    elif robot_name == 'allegro_right':
        offset_point = torch.tensor([0.0, 0.025, 0.0], device=R.device)
        frame_xyz = frame_xyz - torch.matmul(R, offset_point)

    frame_rot6d = torch.cat((R[:, :, 0], R[:, :, 1]), dim=1)
    poses_6d = torch.cat((frame_xyz, frame_rot6d), dim=1)

    indices = torch.randperm(poses_6d.shape[0])[:num_samples]
    poses_6d = poses_6d[indices].to(object_pcd_normals.device)

    object_pos = compute_object_pose_6d(poses_6d[:, :3], poses_6d[:, 3:9])                                  # (B, 6)
    moved_objects = object_pcd_normals.unsqueeze(0).repeat(num_samples, 1, 1)  # (B, N, 6)
    moved_objects, moved_objects_normals = move_pcd_6d(xyz=object_pos[:, :3], rot=object_pos[:, 3:], pcd=moved_objects[..., :3], normals=moved_objects[..., 3:])

    return torch.cat((moved_objects, moved_objects_normals), dim=-1), object_pos, poses_6d                          # (B, N, 6)









if __name__ == "__main__":
    import os, sys
    from time import time
    from utils.constants import DATA_PATH, ROOT_PATH, ROBOTS_PALM_LABELS, ROBOTS_GRASPS_LABELS
    from utils.visualize_plotly import plot_point_cloud, plot_frame, plot_data, save_data, plot_point_cloud_label
    from utils.get_models import get_handmodel
    from utils_model.HandModel import HandModel

    robot_name = 'shadowhand'   # ['allegro_right', 'shadowhand', 'barrett']
    num_samples = 20
    type_name = 'm19'

    if robot_name == 'shadowhand':
        cam = dict(x=0, y=-2, z=0)
    elif robot_name == 'allegro_right':
        cam = dict(x=2, y=0, z=0)
    else:
        raise NotImplementedError(f"Camera position not defined for robot {robot_name}.")

    hand_model : HandModel = get_handmodel(robot=robot_name, batch_size=num_samples, device='cuda')

    hand_verts, hand_faces = hand_model.get_hand_mesh(q=hand_model.straight_pose)
    # hand_verts, hand_faces = hand_model.get_link_mesh('palm', q=hand_model.straight_pose)

    # Simplify Hand Mesh for Collision Checking
    hand_mesh = trimesh.Trimesh(vertices=hand_verts[0].cpu().numpy(), faces=hand_faces[0].cpu().numpy())
    hand_mesh = hand_mesh.simplify_quadric_decimation(face_count=1000)
    hand_verts, hand_faces = torch.from_numpy(hand_mesh.vertices).to('cuda'), torch.from_numpy(hand_mesh.faces).to('cuda')

    # palm_verts, palm_faces = hand_model.get_link_mesh('palm', q=hand_model.straight_pose)

    object_name = 'ycb/tomato_soup_can'
    object_name_split = object_name.split('+')
    object_path = os.path.join(DATA_PATH, 'pointclouds', 'multidex', f'{object_name}.pt')

    object_pcd_normals = torch.load(object_path, map_location='cuda').unsqueeze(0).repeat(num_samples, 1, 1)
    object_pcd = object_pcd_normals[:, :, :3]
    object_normals = object_pcd_normals[:, :, 3:]

    print("Sampling poses...")
    start_time = time()
    on_palm = False
    type_labels = torch.tensor(ROBOTS_GRASPS_LABELS[robot_name][type_name], device='cuda')
    palm_labels = torch.tensor(ROBOTS_PALM_LABELS[robot_name], device='cuda')
    if torch.isin(type_labels, palm_labels).any():
        on_palm = True
        obj_moved, obj_pose, gripper_pose = sample_object_poses_on_palm(robot_name, object_pcd_normals[0], num_samples=num_samples)
    else:
        metadata = torch.load(os.path.join(DATA_PATH, f'handprint/{robot_name}_10000_handprints_normals.pt'), map_location='cuda')         # N (pcd, joint_values)
        full_workspace_labeled = torch.stack([data[1] for data in metadata]).to('cuda')  # (N, P, 7)
        full_workspace_labeled = full_workspace_labeled.view(-1, 7) # Flatten to (Total_Points, 7)
        workspace = create_workspace_type(full_workspace_labeled, type_labels, voxel_size=0.005)
        obj_moved, obj_pose, gripper_pose, workspace_res = sample_object_poses(robot_name, (hand_verts, hand_faces), workspace, object_pcd_normals[0], total_samples=num_samples)  # (B, 9)
    print(f"Sampling done in {time() - start_time:.2f} seconds.")

    #############
    # VISU => Gripper Centric
    # save_folder = os.path.join(ROOT_PATH, 'outputs/290126/sampled_poses/') 
    # if os.path.exists(save_folder):
    #     os.system(f'rm -rf "{save_folder}"')
    # os.makedirs(save_folder, exist_ok=True)

    workspace_pts = torch.load(os.path.join(DATA_PATH, f'workspaces/{robot_name}_workspace_8192_pts.pt'), weights_only=True)
    q = hand_model.straight_pose[0].unsqueeze(0)
    
    # Grasp
    handprint_labels = hand_model.get_handprint_points(q=q, label=True)
    handprint, _, labels = torch.split(handprint_labels, [3, 3, 1], dim=-1)
    grasp = torch.isin(labels.squeeze(-1), torch.tensor(ROBOTS_GRASPS_LABELS[robot_name][type_name], device=labels.device))
    grasp_points = handprint[grasp].cpu().numpy()
    grasp_lbl = labels[grasp].squeeze(-1).cpu().numpy()

    # VISU
    for b in range(min(20, num_samples)):
        vis_data = []
        vis_data += hand_model.get_plotly_data(q=q, color='lightgrey', opacity=0.5, name=f'Hand', show=True)
        vis_data += [plot_point_cloud(workspace_pts.cpu().numpy(), color='grey', size=1.5, name='Workspace', show=True)]
        vis_data += [plot_point_cloud_label(grasp_points, grasp_lbl, text=grasp_lbl, size=5, name='Grasp Points', show=True)]
        if not on_palm:
            vis_data += [plot_point_cloud(workspace.cpu().numpy(), color='lightblue', size=1.5, name='Workspace type', show='legendonly')]
            vis_data += [plot_point_cloud(workspace_res[b].cpu().numpy(), color='red', size=2, name='Sampled workspace pts', show=True)]
        vis_data += [plot_point_cloud(obj_moved[b].cpu().numpy(), color='green', size=3, name='Object', show=True)]
        # save_data(vis_data, os.path.join(save_folder, f'{object_name.replace("/", "_")}_pose_{b}.html'), plot_title=f'Sampled Pose {b}', grid=True, cam=dict(x=0, y=-2, z=0))
        plot_data(vis_data, plot_title=f'Sampled Pose {b}', grid=False, cam=cam)
    
    # VISU => Object Centric
    # q = hand_model.straight_pose[0].unsqueeze(0)
    # for k in range(min(5, num_samples)):
    #     q[:, :9] = gripper_pose[k]
    #     vis_data = []
    #     vis_data += hand_model.get_plotly_data(q=q, color='lightgray', opacity=0.5, name=f'Hand', show=True)
    #     vis_data += [plot_point_cloud(object_pcd_normals[0].cpu().numpy(), color='green', size=3, name='Object', show=True)]
    #     plot_data(vis_data, plot_title=f'Sampled Pose {k}', grid=True, cam=dict(x=0, y=-2, z=0))