import os
import torch
from tqdm import tqdm
import argparse
from utils.get_models import get_handmodel
from utils.constants import DATA_PATH
from utils_model.HandModel import HandModel
from utils.tools import farthest_point_sampling, sort_points_using_hilbert


def generate_random_config_pcd_label(N, robot_name):
    base_hand_model : HandModel = get_handmodel(robot_name, hand_scale=1.0, device=device, num_points=2048)
    upper = base_hand_model.revolute_joints_q_upper
    lower = base_hand_model.revolute_joints_q_lower
    q = base_hand_model.canonical_pose.clone()  # (1, 25)

    n_joints = q.shape[1] - 9

    all_data = []

    pbar = tqdm(total=N, desc=f"[{robot_name}] Creating pointclouds: ")
    while len(all_data) < N:
        
        joint_values = torch.rand(n_joints, device=device)
        joint_values = joint_values.clone() * (upper - lower) + lower
        
        q[:, 9:] = joint_values

        if base_hand_model.penetrate_itself(q=q):
            continue

        hand_model : HandModel = get_handmodel(robot_name, hand_scale=1.0, device=device, num_points=2048)
        pcd_label = hand_model.get_handprint_points(q=q, label=True).squeeze(0)      # (2048, 7)
        all_data.append((joint_values, pcd_label))

        pbar.update(1)
    pbar.close()

    print(f"[{robot_name}] Saving...")
    torch.save(all_data, os.path.join(DATA_PATH, f'handprint/{robot_name}_{N}_handprints_normals.pt'))
    print(f"[{robot_name}] Saved.")



def create_workspace(robot_name, n_pts=4096): # 2048 / 4096 / 8192 / 16384 / 32768
    print(f"[{robot_name}] Loading metadata...", end='\r')
    metadata = torch.load(os.path.join(DATA_PATH, f'handprint/{robot_name}_10000_handprints_normals.pt'), map_location=device)         # N (pcd, joint_values)
    print(f"[{robot_name}] Loading metadata... done.")
    print(f"[{robot_name}] There are {len(metadata)} configs.")
    
    all_pcd = []
    for data in metadata:
        pcd = data[1]                     # (2048, 7)
        all_pcd.append(pcd)

    full_pcd = torch.cat(all_pcd, dim=0)[:, :3]
    pcd, _ = farthest_point_sampling(full_pcd, num_points=n_pts)
    pcd, _ = sort_points_using_hilbert(pcd)
    torch.save(pcd, os.path.join(DATA_PATH, f'workspaces/{robot_name}_workspace_{n_pts}_pts.pt'))
    print(f"[{robot_name}] Workspace saved.")

def update_handprints(N, robot_name):
    print(f"[{robot_name}] Loading metadata...", end=' ')
    data = torch.load(os.path.join(DATA_PATH, f'handprint/{robot_name}_{N}_handprints_normals.pt'), map_location=device)         # N (joint_values, pcd)
    print(f"[{robot_name}] There are {len(data)} configs.")

    all_data = []
    for joint_values, _ in tqdm(data, desc=f"[{robot_name}] Updating handprints: "):
        
        hand_model : HandModel = get_handmodel(robot_name, hand_scale=1.0, device=device, num_points=2048)

        q = hand_model.canonical_pose.clone()
        q[:, 9:] = joint_values

        pcd = hand_model.get_handprint_points(q=q, label=True).squeeze(0)      # (2048, 7)
        all_data.append((joint_values, pcd))

    print(f"[{robot_name}] Saving updated handprints...")
    torch.save(all_data, os.path.join(DATA_PATH, f'handprint/{robot_name}_{N}_handprints_normals.pt'))





def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='shadowhand', type=str, help='Name of the gripper to use')
    parser.add_argument('--n_handprints', default=10000, type=int, help='Number of handprints to generate')
    parser.add_argument('--n_workspace_pts', default=4096, type=int, help='Number of points in the workspace pointcloud')
    args_ = parser.parse_args()
    return args_


if __name__=="__main__":
    
    device = torch.device('cuda')
    args = get_parser()

    print("[INFO] Uncomment the corresponding lines to generate handprints and workspaces.")

    # generate_random_config_pcd_label(args.n_handprints, args.robot_name)
    # create_workspace(args.robot_name, n_pts=args.n_workspace_pts)
    # update_handprints(args.n_handprints, args.robot_name)
