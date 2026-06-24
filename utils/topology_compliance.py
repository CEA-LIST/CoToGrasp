from typing import Type
import torch
import os
import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm
import trimesh as tm
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from termcolor import colored
from utils.tools import retrieve_grasp_type_from_grasp, compute_object_pose, move_pcd
from utils.constants import DATA_PATH, ROOT_PATH, ROBOTS_GRASPS_LABELS, IDX_2_GRASP_TYPE
from utils.get_models import get_handmodel
from utils.visualize_plotly import plot_data, plot_point_cloud, plot_point_cloud_label
from utils.rot6d import q_euler_to_q_rot6d

def get_normalized_entropy(data_dict):
    counts = np.array(list(data_dict.values())[:-2]) # Exclude 'unknown' and 'None'
    total_count = np.sum(counts)
    if total_count == 0:
        return 0.0
    probs = counts / total_count
    probs = probs[probs > 0]
    shannon_entropy = -np.sum(probs * np.log(probs))
    num_categories = len(data_dict) - 2 # Exclude 'unknown' and 'None' from the category count
    max_entropy = np.log(num_categories)
    return shannon_entropy / max_entropy

def type_compliance(robot_name, dataset, validation_results_path, contact_threshold=0.005, rate_threshold=0.0):
    type_count = {k: 0 for k in ROBOTS_GRASPS_LABELS[robot_name].keys()}
    type_count['unknown'] = 0
    type_count['None'] = 0

    type_count_gt = {k: 0 for k in ROBOTS_GRASPS_LABELS[robot_name].keys()}

    type_conformity_info = {k: type_count.copy() for k in ROBOTS_GRASPS_LABELS[robot_name].keys()}

    global_batch = 0
    list_objects, list_q, list_type_gt = [], [], []
    validate_data = torch.load(validation_results_path, map_location='cuda')
    for i in tqdm(range(len(validate_data)), desc=f"Processing objects"):
        data_i = validate_data[i]
        object_name = data_i['object_name']
        
        # INFERENCE: predicted grasps
        # q = data_i['predicted_q'][data_i['success']]
        # grasp_type_gt = data_i['grasp_types'][data_i['success']]

        # ISAAC: successful grasps only
        q = data_i['q_isaac'][data_i['success']]
        grasp_type_gt = data_i['grasp_types'][data_i['success']]

        
        if q.shape[0] == 0:
            continue
        q = q_euler_to_q_rot6d(q)
        
        batch_size = q.shape[0]

        if dataset == 'multidex':
            object_name_split = object_name.split('+')
            object_path = os.path.join(DATA_PATH, 'urdf/objects/multidex', f'{object_name_split[0]}/{object_name_split[1]}/{object_name_split[1]}.obj')
        else:
            object_path = os.path.join(DATA_PATH, f'urdf/objects/{dataset}', f'{object_name}.obj')
        obj_mesh = tm.load(object_path, force='mesh')
        # obj_mesh.vertices *= scale_batch
        obj_pcd, face_indices = tm.sample.sample_surface_even(mesh=obj_mesh, count=10000)
        if len(obj_pcd) < 10000:
            diff = 10000 - len(obj_pcd)
            indices = np.random.choice(len(obj_pcd), diff)
            obj_pcd = np.concatenate([obj_pcd, obj_pcd[indices]], axis=0)
            face_indices = np.concatenate([face_indices, face_indices[indices]], axis=0)
        elif len(obj_pcd) > 10000:
            indices = np.random.permutation(len(obj_pcd))[:10000]
            obj_pcd = obj_pcd[indices]
            face_indices = face_indices[indices]
        obj_normals = obj_mesh.face_normals[face_indices]
        object_pcd = torch.from_numpy(obj_pcd).float().unsqueeze(0).repeat(batch_size, 1, 1).to(device)            # Swap Y and Z axes
        object_normals = torch.from_numpy(obj_normals).float().unsqueeze(0).repeat(batch_size, 1, 1).to(device)    # Swap Y and Z axes
        object_pcd_normals = torch.cat((object_pcd, -object_normals), dim=-1)                           # (B, 10000, 6)

        list_objects.append(object_pcd_normals)
        list_q.append(q)
        list_type_gt.append(grasp_type_gt)

        global_batch += batch_size

        if global_batch >= 200:

            object_pcd_normals = torch.cat(list_objects, dim=0)
            q = torch.cat(list_q, dim=0)
            grasp_type_gt = [t for sublist in list_type_gt for t in sublist]

            list_objects, list_q, list_type_gt = [], [], []
            global_batch = 0

            # Handprint + labels
            hand_model = get_handmodel(robot_name, device=device, batch_size=q.shape[0], num_points=100000)
            handprint_labels = hand_model.get_handprint_points(q, label=True)                  # (1, 2048, 7)
            
            batch_grasp_types = retrieve_grasp_type_from_grasp(robot_name, handprint_labels, object_pcd_normals, contact_threshold, rate_threshold, verbose=False)

            # Count types
            # n_visu = 0
            for b in range(q.shape[0]):
                
                type_gt = IDX_2_GRASP_TYPE[robot_name][grasp_type_gt[b].item()]
                type_pred_list = batch_grasp_types[b]
                
                if type_pred_list is None:
                    type_count['None'] += 1
                elif len(type_pred_list) == 1:
                    type_count[type_pred_list[0][0]] += 1
                else:
                    n = len(type_pred_list)
                    for t in type_pred_list:
                        type_count[t[0]] += 1 / n
                
                type_count_gt[type_gt] += 1
                
                # Type conformity info
                if type_pred_list is None:
                    type_conformity_info[type_gt]['None'] += 1
                elif len(type_pred_list) == 1:
                    type_conformity_info[type_gt][type_pred_list[0][0]] += 1
                else:
                    n = len(type_pred_list)
                    for t in type_pred_list:
                        type_conformity_info[type_gt][t[0]] += 1 / n

                # VISUALIZATION
                # hand_points = handprint_labels[b, :, :3].cpu().numpy()
                # hand_labels = masked_labels[b].cpu().numpy()
                # object_points = object_pcd_reversed_normals[b, :, :3].cpu().numpy()

                # vis_data = []
                # vis_data += [plot_point_cloud(object_points, color='green', name='Object PCD')]
                # # vis_data += [plot_point_cloud(hand_points, color='black', name='Handprint PCD')]
                # vis_data += [plot_point_cloud_label(hand_points, hand_labels, text=hand_labels, size=5, name='Handprint PCD')]

                # plot_data(vis_data, plot_title=f'{b}: Object: {object_name} - GT: {IDX_2_GRASP_TYPE[robot_name][grasp_type_gt[b].item()]} - Grasp Type: {grasp_type_list} - Labels: {batch_unique_labels[b]}')

                # if n_visu+1>=1:
                #     break
                # n_visu+=1
            # break

    return type_count, type_count_gt, type_conformity_info



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--robot_name', default='shadowhand', type=str, help='Name of the robot')
    args = parser.parse_args()

    device = 'cuda'
    robot_name = args.robot_name

    contact_threshold = 0.005       # Distance threshold to consider a contact point
    rate_threshold = 0.0            # Minimum percentage of contact points on a link to consider it in the grasp type determination

    # dataset = 'multidex'
    dataset='dexgraspnet'
    validation_results_path = os.path.join(ROOT_PATH, 'logs_isaac', '02182026', f'{robot_name}_validation_results_{dataset}.pt')

    type_count_dict, type_count_gt_dict, type_conformity_info = type_compliance(robot_name, dataset, validation_results_path, contact_threshold, rate_threshold)

    print(f"Type Count (GT): {type_count_gt_dict}")
    print(f"Type Count (Pred): {type_count_dict}")

    ################################
    ## ANALYSIS

    # Type conformity: for each GT type, how many times each predicted type was retrieved (including None and unknown)
    type_conformity = {}
    for gt_type, pred_counts in type_conformity_info.items():
        total_gt = type_count_gt_dict[gt_type]
        good_pred = pred_counts[gt_type]
        type_conformity[gt_type] = {'total_gt': total_gt, 'good_pred': good_pred, 'conformity_rate': good_pred / total_gt * 100 if total_gt > 0 else 0.0}
    
    # Type conformity info table
    inner_keys = list(type_conformity_info['m1'].keys())
    headers = ['Types'] + ["Total"] + ["TC (%)"] + inner_keys
    table_rows = []
    for gt_type, pred_counts in type_conformity_info.items():
        row = [f"{gt_type}"] + [f"{type_count_gt_dict[gt_type]}"] + [f"{type_conformity[gt_type]['conformity_rate']:.2f}"]
        for pred_type in inner_keys:
            count = pred_counts[pred_type]
            if gt_type == pred_type:
                row.append(colored(f"{count:.1f}", attrs=['bold']))
            else:
                row.append(f"{count:.1f}" if count > 0 else "-")
        table_rows.append(row)
    print(tabulate(table_rows, headers=headers, tablefmt='heavy_outline', stralign='center'))

    # Average conformity rate
    avg_conformity_rate = sum(info['conformity_rate'] for info in type_conformity.values()) / len(type_conformity)
    print(f"Average Conformity Rate: {avg_conformity_rate:.2f}%")


    # GT
    type_name_gt = [k for k in type_count_gt_dict.keys()]
    type_count_gt = [round(v, 2) for v in type_count_gt_dict.values()]
    total_counts_gt = sum(type_count_gt)
    type_freq_gt = [v / total_counts_gt * 100 for v in type_count_gt]
    
    # Pred
    type_name = [k for k in type_count_dict.keys()]
    type_count = [round(v, 2) for v in type_count_dict.values()]
    total_counts = sum(type_count)
    type_freq = [v / total_counts * 100 for v in type_count]   
    
    entropy_gt = get_normalized_entropy(type_count_gt_dict)
    entropy_pred = get_normalized_entropy(type_count_dict)
    
    print(f"Entropy (GT): {entropy_gt}")
    print(f"Entropy (Pred): {entropy_pred}")
    
    # Plotting
    fig = plt.figure(figsize=(15, 6))
    bars = plt.bar(type_name, type_freq, color='skyblue', label=f'Effective Topology (Entropy: {entropy_pred:.4f})')
    bars_gt = plt.bar(type_name_gt, type_freq_gt, fill=False, edgecolor='red', linewidth=2, label=f'Attempted Topology (Entropy: {entropy_gt:.4f})')
    # plt.title(f'CoToGrasp - TC = 17.18%', fontweight='bold', fontsize=24) # -- Grasps: {int(round(total_counts, 0))}
    plt.title(f'Dexonomy - TC = 14.28%', fontweight='bold', fontsize=24)
    plt.xlabel('Contact Topologies', fontweight='bold', fontsize=20)
    plt.ylabel(f'Frequency (%)', fontweight='bold', fontsize=20)
    plt.grid(axis='y', linestyle='--', alpha=0.2)
    plt.tick_params(axis='x', rotation=45, labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, 0.05, f"{type_freq[i]:.2f}", ha='center', va='bottom', fontsize=11)

    for i, bar in enumerate(bars_gt):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{type_freq_gt[i]:.2f}", ha='center', va='bottom', fontsize=11, color='red')

    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.show()