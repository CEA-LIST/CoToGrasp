import torch
import os
import yaml
import time
import json
import argparse
import trimesh
from tqdm import tqdm
from tabulate import tabulate
from termcolor import cprint
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


def get_object_pcd(dataset='multidex', name='ycb/baseball', pos=[0.0, 0.0, 0.0], rot6d=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]):
    # Load object point cloud
    object_path = os.path.join(DATA_PATH, 'pointclouds', dataset, f'{name}.pt')
    object_pcd_normals = torch.load(object_path, map_location=device, weights_only=False).to(torch.float32).unsqueeze(0).repeat(cfg['batch_size'], 1, 1)           # (B, 2048, 6)

    # Position and rotation of the object
    if type(pos) is list:
        pos = torch.tensor([pos]).to(torch.float32).repeat(cfg['batch_size'], 1).to(device)
    if type(rot6d) is list:
        rot6d = torch.tensor([rot6d]).to(torch.float32).repeat(cfg['batch_size'], 1).to(device)

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
    
    args_ = parser.parse_args()
    return args_

if __name__ == "__main__":
    args = get_parser()

    device = torch.device('cuda')

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
    q = hand_model.canonical_pose[0].clone().unsqueeze(0)
    hand_verts, hand_faces = hand_model.get_hand_mesh(q=hand_model.straight_pose)
    hand_mesh = trimesh.Trimesh(vertices=hand_verts[0].cpu().numpy(), faces=hand_faces[0].cpu().numpy())
    hand_mesh = hand_mesh.simplify_quadric_decimation(face_count=1000)
    hand_verts, hand_faces = torch.from_numpy(hand_mesh.vertices).to(device), torch.from_numpy(hand_mesh.faces).to(device)

    basis_set = model.bps_feat.basis_set.clone().to(device)  # (bps_pts, 3)

    ## Objects List
    dataset_folder = os.path.join(DATA_PATH, 'pointclouds', args.dataset)
    with open(os.path.join(DATA_PATH, 'pointclouds', f'split_{args.dataset}.json'), 'r') as f:
        split_data = json.load(f)
    objects_list = split_data['test_split']

    ## Types
    if args.grasp_type is not None:
        if args.grasp_type not in GRASP_TYPE_2_IDX[args.robot_name]:
            raise ValueError(f"Grasp type {args.grasp_type} not recognized for robot {args.robot_name}. Available types: {list(ROBOTS_GRASPS_LABELS[args.robot_name].keys())}")
        DESIRED_TYPES = [GRASP_TYPE_2_IDX[args.robot_name][args.grasp_type]]
    else:
        DESIRED_TYPES = [k for k in range(len(ROBOTS_GRASPS_LABELS[args.robot_name].keys()))]

    metadata = torch.load(os.path.join(DATA_PATH, f'handprint/{args.robot_name}_10000_handprints_normals.pt'), map_location=device, weights_only=True)         # N (pcd, joint_values)
    full_workspace_labeled = torch.stack([data[1] for data in metadata]).to(device)  # (N, P, 7)
    full_workspace_labeled = full_workspace_labeled.view(-1, 7) # Flatten to (Total_Points, 7)

    workspace_per_type = {}
    for desired_type in DESIRED_TYPES:
        type_name = IDX_2_GRASP_TYPE[args.robot_name][desired_type]
        type_labels = torch.tensor(ROBOTS_GRASPS_LABELS[args.robot_name][type_name], device=device)
        palm_labels = torch.tensor(ROBOTS_PALM_LABELS[args.robot_name], device=device)
        if torch.isin(type_labels, palm_labels).any():
            workspace_per_type[desired_type] = None
        else:
            workspace_per_type[desired_type] = create_workspace_type(full_workspace_labeled, type_labels, voxel_size=0.005)

    cprint(f"******** [{args.robot_name.upper()} - Prediction]", 'magenta', attrs=['bold'])
    predicted_data, time_sampling, time_model, time_fc, time_check_type, time_opti = [], [], [], [], [], []
    # for object_name in tqdm(objects_list, desc=f"[{args.robot_name}] Validating on {args.dataset} dataset: ", unit='objects', total=len(objects_list)):
    for object_name in objects_list:
        object_name = object_name[:-3] if object_name.endswith('.pt') else object_name
        if args.object_name is not None and (object_name.replace('/', '+') != args.object_name.replace('/', '+')):
            continue
        
        object_pcd_normals = get_object_pcd(args.dataset, object_name)

        final_results = {
            'labels': {t: [] for t in DESIRED_TYPES},
            'pcd': {t: [] for t in DESIRED_TYPES},
            'poses': {t: [] for t in DESIRED_TYPES},
            'count': {t: 0 for t in DESIRED_TYPES}
        }

        total_needed = args.num_samples_per_type * len(DESIRED_TYPES)

        pbar = tqdm(total=total_needed, desc=f"[{object_name}] All Types", unit='grasps')
    
        max_iter = 15 # Safety break

        while (sum(final_results['count'].values()) < total_needed) and (max_iter > 0):
            max_iter -= 1

            batch_pcd_list, batch_type_list, batch_poses_list = [], [], []
            type_slice_indices = [] 
            current_idx = 0

            start_time_sample = time.time()
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
            
            time_sampling.append((time.time() - start_time_sample) / big_pcd.shape[0])

            labels_hat_list = []
        
            for i in range(0, big_pcd.shape[0], cfg['batch_size']):
                pcd_chunk = big_pcd[i : i + cfg['batch_size']]
                type_chunk = big_types[i : i + cfg['batch_size']]
                chunk_size = pcd_chunk.shape[0]
                
                with torch.no_grad():
                    z_latent = torch.randn((chunk_size, model.latent_size), device=device)
                    start_model = time.time()
                    x_hat = model.inference(pcd_chunk, type_chunk, z_latent)
                    end_model = time.time() - start_model
                    
                labels_hat_list.append(torch.argmax(x_hat, dim=-1))
                time_model.append(end_model / chunk_size)
                
            big_labels_hat = torch.cat(labels_hat_list, dim=0)

            start_check = time.time()
            
            valid_type_mask = type_checker_vec(args.robot_name, big_types, big_labels_hat)
            time_check_type.append((time.time() - start_check) / big_types.shape[0])
            
            if not valid_type_mask.any():
                continue

            # Filter by Type Check
            big_labels_hat = big_labels_hat[valid_type_mask]
            big_pcd = big_pcd[valid_type_mask]
            big_types = big_types[valid_type_mask]
            big_poses = big_poses[valid_type_mask]
            
            # B. Force Closure Check
            start_fc = time.time()
            # ... (Your FC logic here, mostly unchanged, just running on the filtered batch) ...
            workspace_pts = basis_set.clone().unsqueeze(0).repeat(big_labels_hat.shape[0], 1, 1)
            bary, bary_lbls = compute_label_barycenters(workspace_pts, big_labels_hat)
            bary_mask = bary_lbls > 0
            distances = torch.cdist(bary, big_pcd[:, :, :3])
            closest_distances, closest_indices = torch.min(distances, dim=2)
            
            # Indexing trick for batch
            batch_indices = torch.arange(big_pcd.shape[0], device=device).unsqueeze(1).repeat(1, bary.shape[1])
            projected_bary = big_pcd[batch_indices, closest_indices]
            projected_bary = projected_bary * (closest_distances <= 0.03).unsqueeze(-1) * bary_mask.unsqueeze(-1)
            
            G = grasp_matrix_with_object_center(projected_bary, big_pcd[:, :, :3].mean(dim=1))
            # Note: If wrench_space is slow, verify it can handle this batch size. 
            # If not, you might need a small loop here. Assuming it is vectorized:
            G_FC = wrench_space(G, mu=0.3) 
            eps_quality, _ = epsilon_quality_measure(G_FC)
            valid_fc_mask = eps_quality > 0.0
            time_fc.append((time.time() - start_fc) / big_labels_hat.shape[0])
            
            if not valid_fc_mask.any():
                continue

            # Final Filter
            valid_labels = big_labels_hat[valid_fc_mask]
            valid_pcd = big_pcd[valid_fc_mask]
            valid_types = big_types[valid_fc_mask]
            valid_poses = big_poses[valid_fc_mask]
            
            for t in DESIRED_TYPES:
                # Find samples in the valid batch that match this type
                type_mask = (valid_types == t)
                
                if not type_mask.any():
                    continue
                    
                # Extract samples for this type
                new_labels = valid_labels[type_mask]
                new_pcd = valid_pcd[type_mask]
                new_poses = valid_poses[type_mask]
                
                # Calculate how many we actually need
                needed = args.num_samples_per_type - final_results['count'][t]
                
                # Take only what we need
                take = min(needed, new_labels.shape[0])
                
                final_results['labels'][t].append(new_labels[:take])
                final_results['pcd'][t].append(new_pcd[:take])
                final_results['poses'][t].append(new_poses[:take])
                final_results['count'][t] += take
                
                pbar.update(take)

        pbar.close()

            
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
        for b in tqdm(range(0, labels_hat_cat.shape[0], batch_step), desc=f"[{args.robot_name}/{object_name.replace('/', '+')}] Optimizing Grasps"):
            opt_model.reset(args.robot_name, object_name.replace("/", "+"), asked_types_cat[b:b+batch_step], object_pcd_cat[b:b+batch_step, :, :3], -object_pcd_cat[b:b+batch_step, :, 3:6], workspace_pts[b:b+batch_step], labels_hat_cat[b:b+batch_step])
            start_time = time.time()
            q_opti, energy = opt_model.run(verbose=False)                                                                        # (B, num_joints)
            time_opti.append((time.time() - start_time) / q_opti.shape[0])
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

    # Print overall statistics (in ms/grasps)
    mean_time_sampling = (sum(time_sampling) / len(time_sampling)) * 1e3 if len(time_sampling) > 0 else 0
    mean_time_model = (sum(time_model) / len(time_model)) * 1e3 if len(time_model) > 0 else 0
    mean_time_fc = (sum(time_fc) / len(time_fc)) * 1e3 if len(time_fc) > 0 else 0 
    mean_time_check_type = (sum(time_check_type) / len(time_check_type)) * 1e3 if len(time_check_type) > 0 else 0
    mean_time_opti = (sum(time_opti) / len(time_opti)) * 1e3 if len(time_opti) > 0 else 0
    
    table_data = [
        ["Sampling", f"{mean_time_sampling:.4f}", "ms/grasps"],
        ["Model", f"{mean_time_model:.4f}", "ms/grasps"],
        ["Type Checking", f"{mean_time_check_type:.4f}", "ms/grasps"],
        ["Force Closure", f"{mean_time_fc:.4f}", "ms/grasps"],
        ["Optimization", f"{mean_time_opti:.4f}", "ms/grasps"],
        ["Overall time", f"{mean_time_sampling + mean_time_model + mean_time_fc + mean_time_check_type + mean_time_opti:.4f}", "ms/grasps"]
    ]
    time_table = tabulate(table_data, headers=[f"[{args.robot_name.upper()}]", "Time", "Unit"], tablefmt="heavy_outline", stralign="left")
    cprint(time_table, 'cyan', attrs=['bold'])

    date_str = datetime.now().strftime('%m%d%Y')
    result_path = os.path.join('logs_inference_grasps', f'{date_str}')
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=False)
    file_name = f'{args.dataset}_{args.robot_name}_predicted_q.pt'
    torch.save((predicted_data, time_table), os.path.join(result_path, file_name))