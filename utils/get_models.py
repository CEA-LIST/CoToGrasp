import json
import torch
import os

from utils_model.HandModel import HandModel
from utils.constants import DATA_PATH

def get_handmodel(robot: str, batch_size: int = 1, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), hand_scale=1.0, num_points=2048):
    urdf_assets_meta = json.load(open(os.path.join(DATA_PATH, "urdf/robot/urdf_assets_meta.json")))
    urdf_path = urdf_assets_meta['urdf_path'][robot]
    meshes_path = urdf_assets_meta['meshes_path'][robot]
    hand_model = HandModel(robot, urdf_path, meshes_path, batch_size=batch_size, device=device, hand_scale=hand_scale, num_points=num_points)
    return hand_model

