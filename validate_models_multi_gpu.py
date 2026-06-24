import torch
import torch.multiprocessing as mp
import os
import math
import yaml
import json
import argparse
import trimesh
from tqdm import tqdm
from datetime import datetime
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary

from utils_model.CVAE import CVAE
from utils_model.GripperOpt import GripperOpt
from utils.metrics import grasp_matrix_with_object_center, wrench_space, epsilon_quality_measure
from utils.get_models import get_handmodel
from utils.constants import DATA_PATH, ROOT_PATH, ROBOTS_GRASPS_LABELS, IDX_2_GRASP_TYPE, GRASP_TYPE_2_IDX, ROBOTS_PALM_LABELS
from utils.tools import move_pcd_6d, compute_label_barycenters, type_checker_vec
from utils.rot6d import q_rot6d_to_q_euler
from utils.RT_sampling_v2 import sample_object_poses, sample_object_poses_on_palm, create_workspace_type


def get_object_pcd(dataset, name, device, batch_size, pos=[0.0, 0.0, 0.0], rot6d=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]):
    # Load object point cloud
    object_path = os.path.join(DATA_PATH, 'pointclouds', dataset, f'{name}.pt')
    object_pcd_normals = torch.load(object_path, map_location=device, weights_only=False).to(torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)           # (B, 2048, 6)

    # Position and rotation of the object
    if type(pos) is list:
        pos = torch.tensor([pos]).to(torch.float32).repeat(batch_size, 1).to(device)
    if type(rot6d) is list:
        rot6d = torch.tensor([rot6d]).to(torch.float32).repeat(batch_size, 1).to(device)

    object_pcd = object_pcd_normals[:, :, :3]  # (B, 2048, 3)
    object_normals = object_pcd_normals[:, :, 3:]  # (B, 2048, 3)

    object_pcd_moved, object_normals_moved = move_pcd_6d(xyz=pos, rot=rot6d, pcd=object_pcd, normals=object_normals)
    object_pcd_moved_normals = torch.cat((object_pcd_moved, -object_normals_moved), dim=-1)  # (B, 2048, 6)

    return object_pcd_moved_normals


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='shadowhand', type=str, help='Name of the gripper to use')
    parser.add_argument('--dataset', default='multidex', type=str, help='Dataset name')
    parser.add_argument('--object_name', default=None, type=str, help='Name of the object to validate')
    parser.add_argument('--grasp_type', default=None, type=str, help='Grasp type')
    parser.add_argument('--num_samples_per_type', default=100, type=int, help='Number of grasp samples per type per object')
    parser.add_argument('--set_id', default='0', type=str, help='Object set ID: object_list is separated in 4 sets to allow distributed processing')
    parser.add_argument('--num_sets', default=4, type=int, help='Total number of sets the object list is separated into for distributed processing')
    parser.add_argument('--labels_check', default=False, action='store_true', help='Whether to perform grasp type labels check')
    parser.add_argument('--fc_check', default=False, action='store_true', help='Whether to perform force closure check')
    args_ = parser.parse_args()
    return args_

def worker_process(rank, gpu_id, date_str, object_subset, args):
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Worker {rank} starting on GPU {gpu_id} processing {len(object_subset)} objects.")

    if args.robot_name == 'shadowhand':
        train_name = f"{args.robot_name}_goag_dgcnn_types_2_0128"
    elif args.robot_name == 'allegro_right':
        train_name = f"{args.robot_name}_goag_dgcnn_types_2_0209"
    else:
        raise ValueError(f"Robot name {args.robot_name} not recognized. Available options: ['shadowhand', 'allegro_right']")
    
    ## CVAE
    model_basedir = f'logs/{train_name}'
    config_path = os.path.join(model_basedir, "config.yaml")
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    cfg['robot_name'] = args.robot_name
    # cfg['batch_size'] = 10

    ckpt_path = os.path.join(model_basedir, 'ckpts_dir', 'last.ckpt')
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CVAE.load_from_checkpoint(ckpt_path, cfg=cfg).to(device)
    model.eval()

    basis_set = model.bps_feat.basis_set.to(device)

    print(f"Loaded {train_name} at epoch {ckpt['epoch']}")
    print(f"\n{ModelSummary(model, max_depth=1)}")

    ## Optimization Model
    opti_config_path = os.path.join(ROOT_PATH, 'configs/optimization.yaml')
    with open(opti_config_path, "r") as file:
        opti_cfg = yaml.safe_load(file)
    opti_logs_name = None       #f'logs_opti_{args.robot_name}'
    if opti_logs_name:
        opti_tb_logger = TensorBoardLogger(save_dir=os.path.join('logs_opti'), name=opti_logs_name, log_graph=False)
    else:
        opti_tb_logger = None
    opt_model = GripperOpt(cfg=opti_cfg, logger=opti_tb_logger)

    ## Hand Model
    hand_model = get_handmodel(args.robot_name, batch_size=cfg['batch_size'], device=device, num_points=2048)
    hand_verts, hand_faces = hand_model.get_hand_mesh(q=hand_model.straight_pose)
    hand_mesh = trimesh.Trimesh(vertices=hand_verts[0].cpu().numpy(), faces=hand_faces[0].cpu().numpy())
    hand_mesh = hand_mesh.simplify_quadric_decimation(face_count=1000)
    hand_verts, hand_faces = torch.from_numpy(hand_mesh.vertices).to(device), torch.from_numpy(hand_mesh.faces).to(device)

    if args.grasp_type is not None:
        DESIRED_TYPES = [GRASP_TYPE_2_IDX[args.robot_name][args.grasp_type]]
    else:
        DESIRED_TYPES = list(range(len(ROBOTS_GRASPS_LABELS[args.robot_name].keys())))

    metadata = torch.load(os.path.join(DATA_PATH, f'handprint/{args.robot_name}_10000_handprints_normals.pt'), map_location=device, weights_only=True)
    full_workspace_labeled = torch.stack([data[1] for data in metadata]).to(device).view(-1, 7)

    workspace_per_type = {}
    for desired_type in DESIRED_TYPES:
        type_name = IDX_2_GRASP_TYPE[args.robot_name][desired_type]
        type_labels = torch.tensor(ROBOTS_GRASPS_LABELS[args.robot_name][type_name], device=device)
        palm_labels = torch.tensor(ROBOTS_PALM_LABELS[args.robot_name], device=device)
        if torch.isin(type_labels, palm_labels).any():
            workspace_per_type[desired_type] = None
        else:
            workspace_per_type[desired_type] = create_workspace_type(full_workspace_labeled, type_labels, voxel_size=0.005)

    predicted_data = []

    # iterator = tqdm(object_subset, position=rank, desc=f"GPU {gpu_id}") if rank == 0 else object_subset
    for object_name in tqdm(object_subset, desc=f"Set {args.set_id}"):
        object_name_clean = object_name[:-3] if object_name.endswith('.pt') else object_name
        
        try:
            # Pass device explicitly
            object_pcd_normals = get_object_pcd(args.dataset, object_name_clean, device, cfg['batch_size'])
            
            final_results = {
                'labels': {t: [] for t in DESIRED_TYPES},
                'pcd': {t: [] for t in DESIRED_TYPES},
                'poses': {t: [] for t in DESIRED_TYPES},
                'count': {t: 0 for t in DESIRED_TYPES}
            }

            total_needed = args.num_samples_per_type * len(DESIRED_TYPES)
            max_iter = 15

            while (sum(final_results['count'].values()) < total_needed) and (max_iter > 0):
                max_iter -= 1

                batch_pcd_list, batch_type_list, batch_poses_list = [], [], []
                type_slice_indices = [] 
                current_idx = 0

                for desired_type in DESIRED_TYPES:
                    needed = args.num_samples_per_type - final_results['count'][desired_type]
                    if needed <= 0:
                        continue              
                    
                    # Pose Sampling
                    if workspace_per_type[desired_type] is None:
                        pcd_moved, _, poses = sample_object_poses_on_palm(args.robot_name, object_pcd_normals[0], num_samples=needed * 2)
                    else:
                        pcd_moved, _, poses, _ = sample_object_poses(args.robot_name, (hand_verts, hand_faces), workspace_per_type[desired_type], object_pcd_normals[0], total_samples=needed * 2)
                    
                    batch_size_curr = pcd_moved.shape[0]
                    type_tensor = torch.tensor([desired_type], device=device).repeat(batch_size_curr)
                    
                    batch_pcd_list.append(pcd_moved)
                    batch_type_list.append(type_tensor)
                    batch_poses_list.append(poses)
                    
                    # Store metadata to reconstruct types later: (type_id, start_index, end_index)
                    type_slice_indices.append((desired_type, current_idx, current_idx + batch_size_curr))
                    current_idx += batch_size_curr
                    
                if not batch_pcd_list:
                    break
                
                big_pcd = torch.cat(batch_pcd_list, dim=0)          # (Total_Samples, N, 6)
                big_types = torch.cat(batch_type_list, dim=0)       # (Total_Samples,)
                big_poses = torch.cat(batch_poses_list, dim=0)      # (Total_Samples, 9)
                
                labels_hat_list = []
            
                for i in range(0, big_pcd.shape[0], cfg['batch_size']):
                    pcd_chunk = big_pcd[i : i + cfg['batch_size']]
                    type_chunk = big_types[i : i + cfg['batch_size']]
                    chunk_size = pcd_chunk.shape[0]
                    
                    with torch.no_grad():
                        z_latent = torch.randn((chunk_size, model.latent_size), device=device)
                        x_hat = model.inference(pcd_chunk, type_chunk, z_latent)
                        
                    labels_hat_list.append(torch.argmax(x_hat, dim=-1))
                    
                big_labels_hat = torch.cat(labels_hat_list, dim=0)
                
                # Filter by Type Check
                if args.labels_check:
                    valid_type_mask = type_checker_vec(args.robot_name, big_types, big_labels_hat)
    
                    if not valid_type_mask.any():
                        continue

                    big_labels_hat = big_labels_hat[valid_type_mask]
                    big_pcd = big_pcd[valid_type_mask]
                    big_types = big_types[valid_type_mask]
                    big_poses = big_poses[valid_type_mask]
                
                # B. Force Closure Check
                if args.fc_check:
                    workspace_pts = basis_set.clone().unsqueeze(0).repeat(big_labels_hat.shape[0], 1, 1)
                    bary, bary_lbls = compute_label_barycenters(workspace_pts, big_labels_hat)
                    bary_mask = bary_lbls > 0
                    distances = torch.cdist(bary, big_pcd[:, :, :3])
                    closest_distances, closest_indices = torch.min(distances, dim=2)
                    batch_indices = torch.arange(big_pcd.shape[0], device=device).unsqueeze(1).repeat(1, bary.shape[1])
                    projected_bary = big_pcd[batch_indices, closest_indices]
                    projected_bary = projected_bary * (closest_distances <= 0.03).unsqueeze(-1) * bary_mask.unsqueeze(-1)
                    G = grasp_matrix_with_object_center(projected_bary, big_pcd[:, :, :3].mean(dim=1))
                    G_FC = wrench_space(G, mu=0.3) 
                    eps_quality, _ = epsilon_quality_measure(G_FC)
                    valid_fc_mask = eps_quality > 0.0
                    
                    valid_fc_mask = torch.ones(big_labels_hat.shape[0], dtype=torch.bool, device=device)
                    if not valid_fc_mask.any():
                        continue

                    # Final Filter
                    big_labels_hat = big_labels_hat[valid_fc_mask]
                    big_pcd = big_pcd[valid_fc_mask]
                    big_types = big_types[valid_fc_mask]
                    big_poses = big_poses[valid_fc_mask]
                
                for t in DESIRED_TYPES:
                    # Find samples in the valid batch that match this type
                    type_mask = (big_types == t)
                    
                    if not type_mask.any():
                        continue
                        
                    # Extract samples for this type
                    new_labels = big_labels_hat[type_mask]
                    new_pcd = big_pcd[type_mask]
                    new_poses = big_poses[type_mask]
                    
                    # Calculate how many we actually need
                    needed = args.num_samples_per_type - final_results['count'][t]
                    
                    # Take only what we need
                    take = min(needed, new_labels.shape[0])
                    
                    final_results['labels'][t].append(new_labels[:take])
                    final_results['pcd'][t].append(new_pcd[:take])
                    final_results['poses'][t].append(new_poses[:take])
                    final_results['count'][t] += take
                    
            all_labels_cat = []
            all_pcd_cat = []
            all_types_cat = []
            all_poses_cat = []
            
            for t in DESIRED_TYPES:
                if final_results['labels'][t]:
                    all_labels_cat.append(torch.cat(final_results['labels'][t], dim=0))
                    all_pcd_cat.append(torch.cat(final_results['pcd'][t], dim=0))
                    all_types_cat.append(torch.full((final_results['count'][t],), t, device=device))
                    all_poses_cat.append(torch.cat(final_results['poses'][t], dim=0))
                    
            if not all_labels_cat: 
                continue

            labels_hat_cat = torch.cat(all_labels_cat, dim=0)                                                       # (NUM_TYPES * GRASPS_PER_TYPE, bps_pts)
            object_pcd_cat = torch.cat(all_pcd_cat, dim=0)                                                       # (NUM_TYPES * GRASPS_PER_TYPE, N, 6)
            asked_types_cat = torch.cat(all_types_cat, dim=0)                                                       # (NUM_TYPES * GRASPS_PER_TYPE)
            gripper_poses_6d_cat = torch.cat(all_poses_cat, dim=0)                                                   # (NUM_TYPES * GRASPS_PER_TYPE, 9)

            # Optimize Grasps - Per Step to avoid OOM
            workspace_pts = model.bps_feat.basis_set.clone().unsqueeze(0).repeat(labels_hat_cat.shape[0], 1, 1)                     # (GRASPS_PER_TYPE, bps_pts, 3)
            batch_step = 100 * 5    # ~13GB RAM for 100 grasps
            all_q_opti = []
            # for b in tqdm(range(0, labels_hat_cat.shape[0], batch_step), desc=f"[{args.robot_name}/{object_name.replace('/', '+')}] Optimizing Grasps"):
            for b in range(0, labels_hat_cat.shape[0], batch_step):
                opt_model.reset(args.robot_name, object_name.replace("/", "+"), asked_types_cat[b:b+batch_step], object_pcd_cat[b:b+batch_step, :, :3], -object_pcd_cat[b:b+batch_step, :, 3:6], workspace_pts[b:b+batch_step], labels_hat_cat[b:b+batch_step])
                q_opti, energy = opt_model.run(verbose=False)                                                                        # (B, num_joints)
                all_q_opti.append(q_opti)

            all_q_opti = torch.cat(all_q_opti, dim=0)                                                                               # (NUM_TYPES * GRASPS_PER_TYPE, num_joints)
            all_gripper_pos_euler = q_rot6d_to_q_euler(gripper_poses_6d_cat)                                                        # (NUM_TYPES * GRASPS_PER_TYPE, 9)

            predicted_q_full = torch.cat([all_gripper_pos_euler, all_q_opti], dim=1)
            predicted_data.append({
                'object_name': object_name.replace("/", "+"),
                'object_pcd': object_pcd_cat.cpu(),
                'labels_hat': labels_hat_cat.cpu(),
                'grasp_types': asked_types_cat.cpu(),
                'predicted_q': predicted_q_full.cpu(),
            })
        except Exception as e:
            print(f"Error processing {object_name} on GPU {gpu_id}: {e}")
            continue
    
    result_path = os.path.join('logs_inference_grasps', f'{date_str}')
    os.makedirs(result_path, exist_ok=True)
    
    # Save as a partial file
    file_name = f'{args.set_id}_{args.dataset}_{args.robot_name}_labels_{args.labels_check}_fc_{args.fc_check}_part_{rank}.pt'
    torch.save(predicted_data, os.path.join(result_path, file_name))
    print(f"Worker {rank} finished. Saved to {file_name}")



if __name__ == "__main__":
    args = get_parser()

    date_str = datetime.now().strftime('%m%d%Y')

    ## Objects List
    dataset_folder = os.path.join(DATA_PATH, 'pointclouds', args.dataset)
    with open(os.path.join(DATA_PATH, 'pointclouds', f'split_{args.dataset}.json'), 'r') as f:
        split_data = json.load(f)
    objects_list = split_data['test_split']

    # Separate objects into 4 sets for distributed processing
    objects_list = objects_list[int(args.set_id)::int(args.num_sets)]  # Take every 4th object starting from set_id (0, 1, 2, or 3)

    if args.object_name:
        # Filter if specific object requested
        objects_list = [o for o in objects_list if (o[:-3] if o.endswith('.pt') else o).replace('/', '+') == args.object_name.replace('/', '+')]

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected! This script requires CUDA.")
    
    # Create list of IDs [0, 1, 2, ... N-1]
    gpu_list = list(range(num_gpus))
    num_gpus = len(gpu_list)
    print(f"Found {len(objects_list)} objects. Distributing across {num_gpus} GPUs: {gpu_list}")    

    chunk_size = math.ceil(len(objects_list) / num_gpus)
    processes = []

    for rank, gpu_id in enumerate(gpu_list):
        start_idx = rank * chunk_size
        end_idx = min((rank + 1) * chunk_size, len(objects_list))
        subset = objects_list[start_idx:end_idx]
        
        if not subset:
            continue
            
        p = mp.Process(target=worker_process, args=(rank, gpu_id, date_str, subset, args))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    print("All workers finished. Merging results...")

    # 4. Merge Results
    result_path = os.path.join('logs_inference_grasps', f'{date_str}')
    all_data = []
    
    for rank in range(len(gpu_list)):
        part_file = os.path.join(result_path, f'{args.set_id}_{args.dataset}_{args.robot_name}_labels_{args.labels_check}_fc_{args.fc_check}_part_{rank}.pt')
        if os.path.exists(part_file):
            data = torch.load(part_file)
            all_data.extend(data)
            os.remove(part_file)
            
    final_file = os.path.join(result_path, f'{args.set_id}_{args.dataset}_{args.robot_name}_labels_{args.labels_check}_fc_{args.fc_check}_predicted_q.pt')
    torch.save(all_data, final_file)
    print(f"Saved merged results to {final_file}")

    for rank in range(len(gpu_list)):
        part_file = os.path.join(result_path, f'{args.set_id}_{args.dataset}_{args.robot_name}_labels_{args.labels_check}_fc_{args.fc_check}_part_{rank}.pt')
        if os.path.exists(part_file):
            os.remove(part_file)