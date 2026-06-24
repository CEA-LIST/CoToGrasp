import torch
from torch.utils.data import Dataset
import os

from utils.constants import DATA_PATH, ROBOTS_GRASPS_LABELS, GRASP_TYPE_2_IDX

class CoToGraspDataset(Dataset):
    def __init__(self, robot_name='shadowhand', fixed_rt=True, num_handprints=None):
        super(CoToGraspDataset, self).__init__()
        
        self.robot_name = robot_name

        if fixed_rt:
            self.handprints = torch.load(os.path.join(DATA_PATH, f'handprint/{self.robot_name}_10000_handprints_normals.pt'), map_location='cpu', weights_only=True)
        else:
            raise NotImplementedError("Only fixed_rt=True is implemented for now.")

        if num_handprints is not None:
            self.handprints = self.handprints[:num_handprints]

        self.grasps_labels = ROBOTS_GRASPS_LABELS[self.robot_name]
        self.grasp_2_idx = GRASP_TYPE_2_IDX[self.robot_name]

    def __len__(self):
        self.size = len(self.handprints) * len(self.grasps_labels.keys())
        return self.size

    def __getitem__(self, index):
        # Calculate which handprint and which set within that handprint
        handprint_idx = index // len(self.grasps_labels.keys())
        set_idx = index % len(self.grasps_labels.keys())
        # Determine which grasp based on set_idx
        cumulative_sets = 0
        grasp_name = None
        for grasp in self.grasps_labels.keys():
            if set_idx == cumulative_sets:
                grasp_name = grasp
                break
            cumulative_sets += 1
        
        # Joint Values
        joint_values = self.handprints[handprint_idx][0].squeeze(0)  # (N_JOINTS)

        # Retrieve the handprint data
        handprint_label = self.handprints[handprint_idx][1]                     # (2048, 7)
        handprint, label = torch.split(handprint_label, [6, 1], dim=1)          # (2048, 6), (2048, 1)

        # Create grasp mask
        vals = torch.tensor(self.grasps_labels[grasp_name], device=label.device)
        grasp = torch.isin(label.squeeze(-1), vals).to(dtype=label.dtype).unsqueeze(-1)  # (2048,1)

        target_labels = torch.zeros_like(label)
        target_labels[grasp == 1.0] = label[grasp == 1.0]                       # (2048, 1)

        # handprint: (2048, 6), grasp: (2048, 1), target_labels: (2048, 1), grasp_idx: int, joint_values: (N_JOINTS)
        return handprint, grasp, target_labels, self.grasp_2_idx[grasp_name], joint_values
