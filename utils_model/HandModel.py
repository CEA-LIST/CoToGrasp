"""
Truly inspired from CenterGrasp repository
"""

import os
import torch
import transforms3d
import numpy as np
import trimesh as tm
import trimesh.sample
import matplotlib as mpl
import pytorch_kinematics as pk
from plotly import graph_objects as go
import urdf_parser_py.urdf as URDF_PARSER
from pytorch_kinematics.urdf_parser_py.urdf import URDF, Mesh, Cylinder, Box, Sphere

from utils.rot6d import *
from utils.tools import farthest_point_sampling
from utils.constants import DATA_PATH
from utils.constants import ALLEGRO_CANONICAL_HAND_POSE, ALLEGRO_FULL_STRAIGHT_POSE, ALLEGRO_OPTI_START_POSE
from utils.constants import SHADOWHAND_CANONICAL_HAND_POSE, SHADOWHAND_FULL_STRAIGHT_POSE, SHADOWHAND_OPTI_START_POSE
from utils.constants import ROBOTS_GRASPS_LABELS, ROBOTS_LINK_NAMES

class HandModel:
    def __init__(self, robot_name, urdf_filename, mesh_path,
                 batch_size=1, 
                 device=torch.device('cuda'),
                 hand_scale=1.0,
                 num_points=2048,
                 ):
        self.device = device
        
        self.robot_name = robot_name
        self.urdf_filename = urdf_filename
        self.mesh_path = mesh_path

        assert self.robot_name in ['allegro_right', 'shadowhand'], "Robot not supported yet. Robot name must be one of: allegro, shadowhand"

        self.batch_size = batch_size
        self.num_points = num_points
        self.current_status = None
        self.scale = hand_scale
        
        # prepare model
        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(dtype=torch.float, device=self.device)
        self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)
        # print(f"There are {len(self.robot.get_joint_parameter_names())} joints: {self.robot.get_joint_parameter_names()}")

        # prepare geometries for visualization
        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)

        self.link_names = []
        self.labels_to_link = {}
        self.surface_points = {}
        self.surface_points_normal = {}
        self.handprint_points = {}
        self.zones_labels = {}
        self.zones_normals = {}
        self.handprint_points_normal = {}

        self.finger_colors = {
            "palm": 0,
            "little": 1,
            "middle": 2,
            "index": 3,
            "thumb": 4,
        }

        visual = URDF.from_xml_string(open(urdf_filename).read())
        self.mesh_verts = {}
        self.mesh_faces = {}
        self.key_points = {}
        self.complete_vertices = {}
        
        banned_link = []
        if robot_name == 'shadowhand':
            banned_link = ["forearm", "wrist", "fftip", "mftip", "rftip", "lftip", "thbase", "thub", "thtip", 'ffknuckle', 'mfknuckle', 'rfknuckle', 'lfknuckle']
            # banned_link = ["forearm", "wrist", "ffknuckle", "mfknuckle", "rfknuckle", "lfknuckle", "thbase"] --> DRO

        if robot_name == 'allegro_right':
            self.canonical_pose = ALLEGRO_CANONICAL_HAND_POSE.unsqueeze(0).to(device).repeat(self.batch_size, 1)
            self.straight_pose = ALLEGRO_FULL_STRAIGHT_POSE.unsqueeze(0).to(device).repeat(self.batch_size, 1)
            self.opti_start_pose = ALLEGRO_OPTI_START_POSE.unsqueeze(0).to(device).repeat(self.batch_size, 1)
        if robot_name == "shadowhand":
            self.canonical_pose = SHADOWHAND_CANONICAL_HAND_POSE.unsqueeze(0).to(device).repeat(self.batch_size, 1)
            self.straight_pose = SHADOWHAND_FULL_STRAIGHT_POSE.unsqueeze(0).to(device).repeat(self.batch_size, 1)
            self.opti_start_pose = SHADOWHAND_OPTI_START_POSE.unsqueeze(0).to(device).repeat(self.batch_size, 1)

        self.grasp_type = ROBOTS_GRASPS_LABELS[robot_name]
        self.finger_link_names = ROBOTS_LINK_NAMES[robot_name]

        for link in visual.links:

            if (link.name in banned_link) or (len(link.visuals) == 0):
                continue
            # print(f"Processing link: {link.name}")
            self.link_names.append(link.name)

            # MESH VERTICES AND FACES
            mesh, scale, rotation, translation = self.load_mesh(link, mesh_path)
            self.process_meshes(link, mesh, scale, rotation, translation)
            # Keypoints: Geometric center of each mesh
            self.process_keypoints(link, mesh, scale, rotation, translation)
            # Surface point
            self.process_surface_points(link, mesh, scale, rotation, translation)
        
        # Handprint points: based on surface points
        for link_name in self.surface_points.keys():
            self.process_handprint_points(link_name)
        self.downsample_handprint()

        # Zones labels: based on handprint points
        for link_name in self.surface_points.keys():
            self.process_zones(link_name)
        self.init_relations_labels_link_name()
        
        # Joints limits
        self.process_joint_limits()
    
    def normalize_joint_values(self, jv):
        device = jv.device
        jv_normed = (jv.to(self.device) - self.revolute_joints_q_lower) / (self.revolute_joints_q_upper - self.revolute_joints_q_lower)
        return jv_normed.to(device)

    def unnormalize_joint_values(self, jv):
        device = jv.device
        jv_unnormed = jv.to(self.device) * (self.revolute_joints_q_upper - self.revolute_joints_q_lower) + self.revolute_joints_q_lower
        return jv_unnormed.to(device)

    def update_kinematics(self, q):
        self.global_translation = q[:, :3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(q[:,3:9])
        self.current_status = self.robot.forward_kinematics(q[:,9:])

    def update_kinematics_no_base(self, jv):
        self.global_translation = torch.zeros((self.batch_size, 3), device=self.device)
        self.global_rotation = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.current_status = self.robot.forward_kinematics(jv)

    def get_joint_limits(self, cpu=False):
        if cpu:
            return self.revolute_joints_q_lower.cpu(), self.revolute_joints_q_upper.cpu()
        return self.revolute_joints_q_lower, self.revolute_joints_q_upper
    
    def load_mesh(self, link, mesh_path):
        if type(link.visuals[0].geometry) == Mesh:
            filename = link.visuals[0].geometry.filename
            mesh = tm.load(os.path.join(mesh_path, filename), force='mesh', process=False)
        elif type(link.visuals[0].geometry) == Cylinder:
            mesh = tm.primitives.Cylinder(
                radius=link.visuals[0].geometry.radius, height=link.visuals[0].geometry.length)
        elif type(link.visuals[0].geometry) == Box:
            mesh = tm.primitives.Box(extents=link.visuals[0].geometry.size)
        elif type(link.visuals[0].geometry) == Sphere:
            mesh = tm.primitives.Sphere(
                radius=link.visuals[0].geometry.radius)
        else:
            print(type(link.visuals[0].geometry))
            raise NotImplementedError
        try:
            scale = np.array(
                link.visuals[0].geometry.scale).reshape([1, 3])
        except:
            scale = np.array([[1, 1, 1]])
        try:
            rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
            translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
        except AttributeError:
            rotation = transforms3d.euler.euler2mat(0, 0, 0)
            translation = np.array([[0, 0, 0]])
        
        return mesh, scale, rotation, translation

    def process_meshes(self, link, mesh, scale, rotation, translation):
        self.mesh_verts[link.name] = np.array(mesh.vertices) * scale
        self.mesh_verts[link.name] = np.matmul(rotation, self.mesh_verts[link.name].T).T + translation
        self.mesh_faces[link.name] = np.array(mesh.faces)
        complete_vertices = mesh.sample(512)
        complete_vertices = np.matmul(rotation, complete_vertices.T).T + translation
        complete_vertices = np.concatenate([complete_vertices, np.ones([len(complete_vertices), 1])], axis=-1)
        self.complete_vertices[link.name] = torch.from_numpy(complete_vertices).to(self.device).float().unsqueeze(0).repeat(self.batch_size, 1, 1)

    def process_surface_points(self, link, mesh, scale, rotation, translation):
        if self.robot_name == 'shadowhand':
            if link.name == "palm":
                pts, pts_face_index = trimesh.sample.sample_surface_even(mesh=mesh, count=10000, radius=0.001)
            else:
                pts, pts_face_index = trimesh.sample.sample_surface_even(mesh=mesh, count=1000, radius=0.002)
            pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)
        elif self.robot_name == 'allegro_right':
            if link.name == "base_link":
                pts, pts_face_index = trimesh.sample.sample_surface_even(mesh=mesh, count=5000, radius=0.002)
            else:
                pts, pts_face_index = trimesh.sample.sample_surface_even(mesh=mesh, count=500, radius=0.002)
            pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)
        
        pts *= scale
        pts = np.matmul(rotation, pts.T).T + translation

        pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
        pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
        self.surface_points[link.name] = torch.from_numpy(pts).to(
            self.device).float()
        self.surface_points_normal[link.name] = torch.from_numpy(pts_normal).to(
            self.device).float()

    def process_handprint_points(self, link_name):
        if self.robot_name == 'shadowhand':
            # Load handprint points from file
            data = torch.load(os.path.join(DATA_PATH, f'urdf/robot/shadowhand/handprint/{link_name}_filtered.pt'), map_location=self.device, weights_only=True)
            base_handprint_pts, base_handprint_normals = torch.split(data, [3, 3], dim=-1)
            
            pts = self.surface_points[link_name]
            
            # Points downward in the y-axis
            reference_direction = torch.tensor([0, -1, 0], dtype=torch.float32, device=self.device)
            # Compute dot product between normals and reference direction
            dot_products = torch.matmul(pts[:, :3], reference_direction)
            # Keep points where the dot product is positive
            if link_name in ['ffknuckle', 'mfknuckle', 'rfknuckle', 'lfknuckle']:
                handprint_indices = dot_products > 1.0          # no points in these links
            elif link_name in ['palm', 'ee_link', 'imu', 'lfmetacarpal']:
                handprint_indices = (dot_products > 7.0e-3) & (pts[:, 2] >= 0.01)
            else:
                handprint_indices = dot_products > -1.0e-3

            handprint_pts = pts[handprint_indices]
            
            # For each point in pts, find the closest point in base_handprint_pts and assign its normal
            dists = torch.cdist(handprint_pts[:, :3], base_handprint_pts.to(self.device))  # (N_pts, N_base)
            closest_idx = torch.argmin(dists, dim=1)  # (N_pts,)
            pts_normals = base_handprint_normals[closest_idx]

            # handprint_pts = torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)
            handprint_normals = torch.cat([pts_normals, torch.ones_like(pts_normals[..., :1])], dim=-1)
            self.handprint_points[link_name] = handprint_pts.to(self.device).float()
            self.handprint_points_normal[link_name] = handprint_normals.to(self.device).float()
        elif self.robot_name == 'allegro_right':
            if link_name == "link_12.0":
                reference_direction = torch.tensor([0, 1, 0], dtype=torch.float32, device=self.device)
            else:
                reference_direction = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device) # Points downward in the x-axis
            # Normalize the reference direction
            reference_direction = reference_direction / torch.norm(reference_direction)
            pts = self.surface_points[link_name]
            pts_normal = self.surface_points_normal[link_name]
            if link_name in ['link_3.0', 'link_7.0', 'link_11.0']:
                mask_z = pts[:, 2] > 0.01
                pts = pts[mask_z]
                pts_normal = pts_normal[mask_z]

            # Compute dot product between normals and reference direction
            dot_products = torch.matmul(pts[:, :3], reference_direction)
            # Keep points where the dot product is positive
            if link_name in ["base_link", "link_12.0"]:
                handprint_indices = dot_products > 1.0e-2
            else:
                handprint_indices = dot_products > -5.0e-3     # 5.0e-3
            handprint_pts = pts[handprint_indices]
            handprint_normals = pts_normal[handprint_indices]
            self.handprint_points[link_name] = handprint_pts.to(self.device).float()
            self.handprint_points_normal[link_name] = handprint_normals.to(self.device).float()
        else:
            raise NotImplementedError("HANDPRINT not implemented for this robot")
            
    def process_zones(self, link_name):
        pts = self.handprint_points[link_name]
        if self.robot_name == 'shadowhand':
            if link_name == 'palm': # Palm
                labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 1, dtype=torch.float32, device=pts.device)
                pts_x = pts[:, :, 0]
                pts_z = pts[:, :, 2]
                # Define key points
                x0, x1, x2, x3, x4 = -0.039, -0.02, 0.002, 0.022, 0.012
                z0, z1, z2, z3, z4 = 0.04, 0.05, 0.06, 0.074, 0.01
                # Calculate line equations
                f4_a, f4_b = (z1 - z0) / (x3 - x2), z1 - ((z1 - z0) / (x3 - x2)) * x3
                f5_a, f5_b = (z4 - z0) / (x4 - x2), z4 - ((z4 - z0) / (x4 - x2)) * x4
                # Update labels based on condition
                labels_tensor[(pts_z < f4_a * pts_x + f4_b) & (pts_z >= f5_a * pts_x + f5_b) & (pts_x >= x2)] = 2
                labels_tensor[(pts_z >= f4_a * pts_x + f4_b) & (pts_x >= x2) & (pts_x < x3) & (pts_z < z3)] = 3
                labels_tensor[(pts_z >= z1) & (pts_z < z3) & (pts_x >= x3)] = 4
                # 8 belongs to little finger base region
                labels_tensor[(pts_z >= z3) & (pts_x < x2)] = 5
                labels_tensor[(pts_z >= z3) & (pts_x >= x2) & (pts_x < x3)] = 6
                labels_tensor[(pts_z >= z3) & (pts_x >= x3)] = 7
            else: # Fingers
                if link_name == 'lfmetacarpal':
                    # Little finger base region
                    f_a, f_b = (0.018 - 0.031) / (0.043 - 0.029), 0.018 - ((0.018 - 0.031) / (0.043 - 0.029)) * 0.043
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 1, dtype=torch.float32, device=pts.device)
                    labels_tensor[pts[:, :, 2] > f_a * pts[:, :, 0] + f_b] = 5
                elif link_name in ['ffproximal', 'ffmiddle']:
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 8, dtype=torch.float32, device=pts.device)
                    labels_tensor[pts[:, :, 0] > 0] = 9
                elif link_name == 'ffdistal':
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 10, dtype=torch.float32, device=pts.device)
                    labels_tensor[pts[:, :, 0] > 0] = 11
                elif link_name in ['mfproximal', 'mfmiddle']:
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 12, dtype=torch.float32, device=pts.device)
                    labels_tensor[pts[:, :, 0] > 0] = 13
                elif link_name == 'mfdistal':
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 14, dtype=torch.float32, device=pts.device)
                    labels_tensor[pts[:, :, 0] > 0] = 15
                elif link_name in ['rfproximal', 'rfmiddle']:
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 16, dtype=torch.float32, device=pts.device)
                elif link_name == 'rfdistal':
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 17, dtype=torch.float32, device=pts.device)
                elif link_name in ['lfproximal', 'lfmiddle']:
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 18, dtype=torch.float32, device=pts.device)
                elif link_name == 'lfdistal':
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 19, dtype=torch.float32, device=pts.device)
                elif link_name == 'thproximal':
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 20, dtype=torch.float32, device=pts.device)
                elif link_name == 'thmiddle':
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 21, dtype=torch.float32, device=pts.device)
                elif link_name == 'thdistal':
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 22, dtype=torch.float32, device=pts.device)
                else:
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 0, dtype=torch.float32, device=pts.device)
        elif self.robot_name == 'allegro_right':
            if link_name == 'base_link':
                labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 1, dtype=torch.float32, device=pts.device)
                pts_y = pts[:, :, 1]
                pts_z = pts[:, :, 2]
                # Keypoints
                y0, y1 = -0.013, 0.027
                z0, z1 = -0.024, -0.048
                # Update labels based on condition
                labels_tensor[(pts_y < y0) & (pts_z < z0)] = 1
                labels_tensor[(pts_y >= y0) & (pts_z < z1)] = 1
                labels_tensor[(pts_y >= y0) & (pts_y < y1) & (pts_z >= z1) & (pts_z < z0)] = 3
                labels_tensor[(pts_y >= y1) & (pts_z >= z1) & (pts_z < z0)] = 4
                labels_tensor[(pts_y < y0) & (pts_z >= z0)] = 5
                labels_tensor[(pts_y >= y0) & (pts_y < y1) & (pts_z >= z0)] = 6
                labels_tensor[(pts_y >= y1) & (pts_z >= z1) & (pts_z >= z0)] = 7
            else:
                if link_name in ['link_0.0', 'link_1.0', 'link_2.0']:
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 8, dtype=torch.float32, device=pts.device)
                    labels_tensor[pts[:, :, 1] > 0.0] = 9
                elif link_name in ['link_3.0', 'link_3.0_tip']:
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 10, dtype=torch.float32, device=pts.device)
                    labels_tensor[pts[:, :, 1] > 0] = 11
                elif link_name in ['link_4.0', 'link_5.0', 'link_6.0']:
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 12, dtype=torch.float32, device=pts.device)
                    labels_tensor[pts[:, :, 1] > 0.0] = 13
                elif link_name in ['link_7.0', 'link_7.0_tip']:
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 14, dtype=torch.float32, device=pts.device)
                    labels_tensor[pts[:, :, 1] > 0] = 15
                elif link_name in ['link_8.0', 'link_9.0', 'link_10.0']:
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 16, dtype=torch.float32, device=pts.device)
                elif link_name in ['link_11.0', 'link_11.0_tip']:
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 17, dtype=torch.float32, device=pts.device)
                elif link_name == 'link_12.0':
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 2, dtype=torch.float32, device=pts.device)
                    labels_tensor[pts[:, :, 2] > 0.02] = 18
                elif link_name == 'link_13.0':
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 18, dtype=torch.float32, device=pts.device)
                elif link_name == 'link_14.0':
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 18, dtype=torch.float32, device=pts.device)
                    labels_tensor[pts[:, :, 2] > 0.035] = 19
                elif link_name == 'link_15.0':
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 19, dtype=torch.float32, device=pts.device)
                    labels_tensor[pts[:, :, 2] > 0.02] = 20
                elif link_name == 'link_15.0_tip':
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 20, dtype=torch.float32, device=pts.device)
                else:
                    labels_tensor = torch.full((pts.shape[0], pts.shape[1]), 0, dtype=torch.float32, device=pts.device)
        else:
            raise NotImplementedError("Zones not implemented for this robot")
        self.zones_labels[link_name] = labels_tensor.to(self.device)
        self.labels_to_link[link_name] = torch.unique(labels_tensor).to(self.device)
    
    def process_keypoints(self, link, mesh, scale, rotation, translation):
        bounds_min, bounds_max = mesh.bounds[0], mesh.bounds[1]
        center = (bounds_min + bounds_max) / 2 * scale

        if self.robot_name == 'shadowhand' and link.name in ['lfmetacarpal']:
            center[:, 0] -= (bounds_max[0] - bounds_min[0]) * scale[:, 0] / 4
            center[:, 2] += 0.005

        center = np.matmul(rotation, center.T).T + translation
        center = np.concatenate([center, np.ones([len(center), 1])], axis=-1)
        self.key_points[link.name] = torch.from_numpy(center).to(self.device).float().unsqueeze(0).repeat(self.batch_size, 1, 1)
    
    def init_relations_labels_link_name(self):
        self.all_link_names = sorted(list(self.labels_to_link.keys()))
        # Link name to link index
        self.link_name_to_idx = {name: i for i, name in enumerate(self.all_link_names)}
        # Label to link index
        max_label = max([torch.max(v).item() for v in self.labels_to_link.values()])
        self.label_to_link_idx = torch.full((int(max_label) + 1,), 0, dtype=torch.long)
        for link_name, labels in self.labels_to_link.items():
            idx = self.link_name_to_idx[link_name]
            for label in labels:
                self.label_to_link_idx[label.to(torch.long)] = idx      
        # Link index to root link index
        self.link_idx_to_root_idx = torch.zeros(len(self.all_link_names), dtype=torch.long)
        finger_root_map = {}
        for _, links in self.finger_link_names.items():
            root = links[0]
            for link in links:
                finger_root_map[link] = root
        for i, name in enumerate(self.all_link_names):
            root_name = finger_root_map.get(name, "palm")
            if root_name in self.link_name_to_idx:
                self.link_idx_to_root_idx[i] = self.link_name_to_idx[root_name]
            else:
                self.link_idx_to_root_idx[i] = 0

    def process_joint_limits(self):
        self.revolute_joints = []
        for i in range(len(self.robot_full.joints)):
            if self.robot_full.joints[i].joint_type == 'revolute':
                self.revolute_joints.append(self.robot_full.joints[i])
        self.revolute_joints_q_mid = []
        self.revolute_joints_q_var = []
        self.revolute_joints_q_upper = []
        self.revolute_joints_q_lower = []
        for i in range(len(self.robot.get_joint_parameter_names())):
            for j in range(len(self.revolute_joints)):
                if self.revolute_joints[j].name == self.robot.get_joint_parameter_names()[i]:
                    joint = self.revolute_joints[j]
            assert joint.name == self.robot.get_joint_parameter_names()[i]
            self.revolute_joints_q_mid.append(
                (joint.limit.lower + joint.limit.upper) / 2)
            self.revolute_joints_q_var.append(
                ((joint.limit.upper - joint.limit.lower) / 2) ** 2)
            self.revolute_joints_q_lower.append(joint.limit.lower)
            self.revolute_joints_q_upper.append(joint.limit.upper)

        self.revolute_joints_q_lower = torch.Tensor(
            self.revolute_joints_q_lower).repeat([self.batch_size, 1]).to(self.device)
        self.revolute_joints_q_upper = torch.Tensor(
            self.revolute_joints_q_upper).repeat([self.batch_size, 1]).to(self.device)

    def downsample_handprint(self):
        self.update_kinematics(self.straight_pose)

        handprint_points = []
        for link_name in self.handprint_points:
            trans_matrix = self.current_status[link_name].get_matrix().to(self.device)
            pts = torch.matmul(trans_matrix[0], self.handprint_points[link_name].unsqueeze(0).transpose(1, 2)).transpose(1, 2)[..., :3]
            handprint_points.append(pts)

        handprint_points = torch.cat(handprint_points, 1).to(self.device)
        handprint_points = torch.matmul(self.global_rotation, handprint_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        handprint_points = handprint_points[0] * self.scale
        
        # Perform farthest point sampling to keep only self.num_points
        _, downsampled_indices = farthest_point_sampling(handprint_points, self.num_points)
        # Update self.handprint_points with the downsampled points
        start_idx = 0
        for link in self.handprint_points:
            num_points = self.handprint_points[link].shape[0]
            link_indices = downsampled_indices[(downsampled_indices >= start_idx) & (downsampled_indices < start_idx + num_points)] - start_idx
            self.handprint_points[link] = self.handprint_points[link][link_indices, :].unsqueeze(0).repeat(self.batch_size, 1, 1)
            start_idx += num_points
        
        # Update self.handprint_points_normal with the same indices
        start_idx = 0
        for link in self.handprint_points_normal:
            num_points = self.handprint_points_normal[link].shape[0]
            link_indices = downsampled_indices[(downsampled_indices >= start_idx) & (downsampled_indices < start_idx + num_points)] - start_idx
            self.handprint_points_normal[link] = self.handprint_points_normal[link][link_indices, :].unsqueeze(0).repeat(self.batch_size, 1, 1)
            start_idx += num_points

    def get_surface_points_differentiable(self):
        surface_points = []
        for i, link_name in enumerate(self.surface_points):
            trans_matrix = self.current_status[link_name].get_matrix()
            pts = torch.matmul(trans_matrix, self.surface_points[link_name].unsqueeze(0).repeat(self.batch_size, 1, 1).transpose(1, 2)).transpose(1, 2)[..., :3]
            surface_points.append(pts)
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)

        return surface_points * self.scale

    def get_handprint_points_differentiable(self):
        handprint_points = []
        for link_name in self.handprint_points:
            trans_matrix = self.current_status[link_name].get_matrix()
            pts = torch.matmul(trans_matrix, self.handprint_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3]
            handprint_points.append(pts)
        handprint_points = torch.cat(handprint_points, 1)
        handprint_points = torch.matmul(self.global_rotation, handprint_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        
        return handprint_points * self.scale

    def get_handprint_points(self, q=None, label=False):
        """Return the handprint points of the hand model.

        Args:
            q (torch.tensor): size (B, 16/22/8). Defaults to None.
            label (bool, optional): Defaults to False.

        Returns:
            pcd (torch.tensor): size (B, N, 7) if label is True, else (B, N, 6).
        """
        if q is not None:
            B = min(q.shape[0], self.batch_size)
            self.update_kinematics(q[:B])
        else:
            B = self.batch_size

        handprint_labels = []
        handprint_points = []
        handprint_normals = []
        for link_name in self.handprint_points:
            # print(f"{link_name}")
            
            # Get handprint points and normals with link at origin
            pts = self.handprint_points[link_name][:B]
            pts_normal = self.handprint_points_normal[link_name][:B]
            labels = self.zones_labels[link_name][:B]

            # Transform points to global frame
            trans_matrix = self.current_status[link_name].get_matrix().to(self.device)
            pts = torch.matmul(trans_matrix, pts.transpose(1, 2)).transpose(1, 2)
            pts_normal = torch.matmul(trans_matrix, pts_normal.transpose(1, 2)).transpose(1, 2)
                
            handprint_points.append(pts[..., :3])
            handprint_normals.append(pts_normal[..., :3])
            handprint_labels.append(labels)

        handprint_points = torch.cat(handprint_points, dim=1).to(self.device)
        handprint_normals = torch.cat(handprint_normals, dim=1).to(self.device)
        handprint_labels = torch.cat(handprint_labels, dim=1).to(self.device)
        handprint_points = torch.matmul(self.global_rotation, handprint_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        handprint_normals = torch.matmul(self.global_rotation, handprint_normals.transpose(1, 2)).transpose(1, 2)

        handprint_points = handprint_points * self.scale
        handprint_points_normals = torch.cat((handprint_points, handprint_normals), dim=-1)
        handprint_labels = handprint_labels.unsqueeze(-1)

        if label:
            handprint_points_normals = torch.cat((handprint_points_normals, handprint_labels), dim=-1)

        return handprint_points_normals

    def get_zones_normals(self, q=None):
        if q is not None:
            B = min(q.shape[0], self.batch_size)
            self.update_kinematics(q[:B])
        else:
            B = self.batch_size
        
        normals = []
        for link_name in self.zones_normals:
            # Get zones normals with link at origin
            pts_normal = self.zones_normals[link_name][:B]
            # Transform normals to global frame
            trans_matrix = self.current_status[link_name].get_matrix().to(self.device)
            pts_normal = torch.matmul(trans_matrix, pts_normal.transpose(1, 2)).transpose(1, 2)
            normals.append(pts_normal[..., :3])
        
        zones_normals = torch.cat(normals, dim=1).to(self.device)
        zones_normals = torch.matmul(self.global_rotation, zones_normals.transpose(1, 2)).transpose(1, 2)
        return zones_normals
    
    def get_zones_centers(self, q=None):
        H = self.get_handprint_points(q, label=True)
        handprint, labels = H[:, :, :3], H[:, :, -1].long()
        B, N, D = handprint.shape
        flat_points = handprint.view(-1, 3)      # (B*N, 3)
        num_zones = labels.max(dim=-1).values.max().item() + 1
        batch_offsets = (torch.arange(B, device=handprint.device) * num_zones).view(-1, 1)
        unique_indices = (labels.long() + batch_offsets).view(-1)
        sum_points = torch.zeros((B * num_zones, D), device=handprint.device)
        sum_points.index_add_(0, unique_indices, flat_points)
        counts = torch.zeros(B * num_zones, device=handprint.device)
        ones = torch.ones(B * N, device=handprint.device)
        counts.index_add_(0, unique_indices, ones)
        counts = counts.clamp(min=1).unsqueeze(1)
        means = sum_points / counts
        zones_centers = means.view(B, num_zones, D)
        return zones_centers

    def get_link_from_labels(self, labels: torch.Tensor):
        """
        Args:
            labels (B, N): Labels indicating which contact points belong to which finger.
        
        Returns:
            link_names (list of list of str): List of link names corresponding to each label for each batch.
        """
        link_names = []
        for b in range(labels.shape[0]):
            link_names_per_batch = []
            for label in labels[b]:  # Assuming all batches have the same labels
                for link_name in self.labels_to_link:
                    if label.item() in self.labels_to_link[link_name]:
                        link_names_per_batch.append(link_name)
                        break
            link_names.append(link_names_per_batch)
        return link_names

    def get_link_indices_from_labels(self, labels: torch.Tensor):
        """
        Args:
            labels (B, N): Labels indicating which contact points belong to which finger.
        
        Returns:
            link_indices (torch.Tensor): Link indices corresponding to each label. Size (B, N).
        """
        if self.label_to_link_idx.device != labels.device:
            self.label_to_link_idx = self.label_to_link_idx.to(labels.device)
        link_indices = self.label_to_link_idx[labels]
        return link_indices

    def get_key_points(self, q=None, tip_only=False, with_base=True):
        if q is not None:
            self.update_kinematics(q)
        B = min(q.shape[0], self.batch_size)
        key_points = []
        for link_name in self.key_points:
            # print(link_name)
            trans_matrix = self.current_status[link_name].get_matrix()
            link_kp = self.key_points[link_name].transpose(1, 2)[:B]
            key_points.append(torch.matmul(trans_matrix, link_kp).transpose(1, 2)[..., :3])
        key_points = torch.cat(key_points, dim=1).to(self.device)
        key_points = torch.matmul(self.global_rotation, key_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        key_points = key_points * self.scale
        if tip_only:
            return key_points[:, 5::5]
        if not with_base:
            return key_points[:, 1:]
        return key_points

    def get_keypoints_differentiable(self, with_base=False):
        key_points = []
        for link_name in self.key_points:
            # print(link_name)
            trans_matrix = self.current_status[link_name].get_matrix()
            link_kp = self.key_points[link_name].transpose(1, 2)
            key_points.append(torch.matmul(trans_matrix, link_kp).transpose(1, 2)[..., :3])
        key_points = torch.cat(key_points, dim=1).to(self.device)
        key_points = torch.matmul(self.global_rotation, key_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        key_points = key_points * self.scale
        if not with_base:
            return key_points[:, 1:]
        return key_points

    def get_keypoints_differentiable_from_labels(self, labels):
        """
        Args:
            labels (B, N): Labels indicating which contact points belong to which finger.
        
        Returns:
            key_points (B, N, 3): Keypoints for each finger based on the labels.
        """
        keypoints = self.get_keypoints_differentiable(with_base=True)  # (B, 21, 3)
        labels_exp = labels.unsqueeze(-1).expand(-1, -1, 3)
        selected_keypoints = torch.gather(keypoints, 1, labels_exp)
        return selected_keypoints

    def get_link_key_points_differentiable(self, link_name=None):
        """
        Returns the keypoint(s) for a given link in a differentiable way, given joint_values (B, N_joints).
        The returned tensor will have gradients with respect to joint_values.
        """
        if link_name not in self.key_points:
            raise ValueError(f"Link name {link_name} not found in key points.")

        # Get transformation matrix for the link
        trans_matrix = self.current_status[link_name].get_matrix()  # (B, 4, 4)
        link_kp = self.key_points[link_name].transpose(1, 2)  # (B, 4, 1)
        # Transform keypoint(s)
        kp = torch.matmul(trans_matrix, link_kp).transpose(1, 2)[..., :3]  # (B, 1, 3) or (B, N, 3)
        # Apply global rotation and translation
        kp = torch.matmul(self.global_rotation, kp.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return kp * self.scale

    def get_link_keypoint(self, q, link_name):
        if link_name not in self.key_points:
            raise ValueError(f"Link name {link_name} not found in key points.")
        if q is not None:
            self.update_kinematics(q)
        B = min(q.shape[0], self.batch_size)

        trans_matrix = self.current_status[link_name].get_matrix()
        link_kp = self.key_points[link_name].transpose(1, 2)[:B]
        kp = [torch.matmul(trans_matrix, link_kp).transpose(1, 2)[..., :3]]
        kp = torch.cat(kp, dim=1).to(self.device)
        kp = torch.matmul(self.global_rotation, kp.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return kp * self.scale

    def get_meshes_from_q(self, q=None, i=0):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        return data

    def penetrate_itself(self, q):
        
        if q.shape[0] != 1:
            raise ValueError("This function is designed for a batch of size 1.")
        
        # Get keypoints for each fingers 
        key_points = self.get_key_points(q).to(self.device)
        if self.robot_name == "allegro_right":
            threshold = 0.03
            kp_palm = key_points[:, 0:1, :]                         # Palm
            kp_lf = key_points[:, 1:6, :]                           # Little finger
            kp_mf = key_points[:, 6:11, :]                          # Middle finger
            kp_if = key_points[:, 11:16, :]                         # Index finger
            kp_th = key_points[:, 16:, :]                           # Thumb
            kp_th_no_root = key_points[:, 17:, :]                   # Thumb no root
            kps_fingers = [kp_lf, kp_mf, kp_if, kp_th]
            kps_fingers_no_finger_root = [kp_lf, kp_mf, kp_if, kp_th_no_root]
        elif self.robot_name == "shadowhand":
            threshold = 0.022
            kp_palm = key_points[:, 0:1, :]                         # Palm
            kp_if = key_points[:, 1:5, :]                           # Index finger
            kp_mf = key_points[:, 5:9, :]                           # Middle finger
            kp_rf = key_points[:, 9:13, :]                          # Ring finger
            kp_lf = key_points[:, 13:18, :]                         # Little finger
            kp_lf_no_root = key_points[:, 14:18, :]                 # Little finger no root
            kp_th = key_points[:, 18:, :]                           # Thumb
            kps_fingers = [kp_if, kp_mf, kp_rf, kp_lf, kp_th]
            kps_fingers_no_finger_root = [kp_if, kp_mf, kp_rf, kp_lf_no_root, kp_th]
        else:
            raise NotImplementedError("Robot not supported yet. Robot name must be one of: allegro, shadowhand, barrett")

        # Distance between each finger and the palm only
        for kps_finger in kps_fingers_no_finger_root:
            kps_finger = kps_finger.squeeze(0).to(self.device)
            distances = torch.cdist(kp_palm, kps_finger, p=2).to(self.device)
            # print(f"Distances between palm and finger keypoints: {distances}")
            if torch.any(distances < threshold):  # If any palm-finger distance is too small
                # print("Palm and finger are too close!")
                return True
        
        # Check distances between different fingers
        key_points_fingers_only = key_points[:, 1:, :]  # Remove palm
        distances = torch.cdist(key_points_fingers_only, key_points_fingers_only, p=2).to(self.device)

        # Create a mask to exclude intra-finger distances
        finger_labels = torch.cat([torch.full((kp.shape[1],), i) for i, kp in enumerate(kps_fingers)]).to(self.device)
        same_finger_mask = finger_labels[:, None] == finger_labels[None, :]
        distances[:, same_finger_mask] = float('inf')  # Ignore intra-finger distances

        # print(f"Distances between different fingers: {distances}")

        if (distances < threshold).any():
            # print("Fingers are too close to each other!")
            return True

        return False

    def get_hand_mesh(self, q=None):
        """
        Return the transformed vertices and faces of the entire hand mesh.
        
        :param q: Joint values to update the kinematics. If None, uses the current state.
        :returns: Transformed vertices and faces of the entire hand mesh.
        """
        if q is not None:
            B = min(q.shape[0], self.batch_size)
            self.update_kinematics(q[:B])
        else:
            B = self.batch_size
        all_verts = []
        all_faces = []
        vert_offset = 0
        for link_name in self.mesh_verts:
            trans_matrix = self.current_status[link_name].get_matrix()
            idx = min(len(trans_matrix) - 1, 0)
            trans_matrix = trans_matrix[idx] 
            v = torch.from_numpy(self.mesh_verts[link_name]).float().to(self.device)
            ones = torch.ones((v.shape[0], 1), device=v.device, dtype=v.dtype)
            v_homo = torch.cat([v, ones], dim=-1) # [N, 4]
            transformed_v = torch.matmul(trans_matrix, v_homo.T).T[..., :3] # [N, 3]
            global_rot = self.global_rotation[0]
            global_trans = self.global_translation[0]
            transformed_v = torch.matmul(global_rot, transformed_v.T).T + global_trans
            transformed_v = transformed_v * self.scale
            f = torch.from_numpy(self.mesh_faces[link_name]).long().to(self.device)
            all_verts.append(transformed_v)
            all_faces.append(f + vert_offset)
            vert_offset += transformed_v.shape[0]

        all_verts = torch.cat(all_verts, dim=0)
        all_faces = torch.cat(all_faces, dim=0)
        return all_verts.unsqueeze(0).repeat(B, 1, 1), all_faces.unsqueeze(0).repeat(B, 1, 1)


    def get_link_mesh(self, link_name, q=None):
        """
        Return the transformed vertices and faces of a specific link's mesh.
        
        :param link_name: Name of the link whose mesh is to be retrieved.
        :param q: Joint values to update the kinematics. If None, uses the current state.
        :returns: Transformed vertices and faces of the link's mesh.
        """
        if q is not None:
            B = min(q.shape[0], self.batch_size)
            self.update_kinematics(q[:B])
        else:
            B = self.batch_size
        if link_name in self.mesh_verts:
            trans_matrix = self.current_status[link_name].get_matrix()
            idx = min(len(trans_matrix) - 1, 0)
            trans_matrix = trans_matrix[idx] 
            v = torch.from_numpy(self.mesh_verts[link_name]).float().to(self.device)
            ones = torch.ones((v.shape[0], 1), device=v.device, dtype=v.dtype)
            v_homo = torch.cat([v, ones], dim=-1) # [N, 4]
            transformed_v = torch.matmul(trans_matrix, v_homo.T).T[..., :3] # [N, 3]
            global_rot = self.global_rotation[0]
            global_trans = self.global_translation[0]
            transformed_v = torch.matmul(global_rot, transformed_v.T).T + global_trans
            transformed_v = transformed_v * self.scale
            f = torch.from_numpy(self.mesh_faces[link_name]).long().to(self.device)
            return transformed_v.unsqueeze(0).repeat(B, 1, 1), f.unsqueeze(0).repeat(B, 1, 1)

    def get_plotly_data(self, q=None, link_names=None, display_joints=False, color='lightgreen', color_map=None, show_color_map=False, opacity=0.7, name='Hand', show=True):
        data = []
        if q is not None: 
            self.update_kinematics(q)

        if link_names is not None:
            if show_color_map:
                hsv_color_map = mpl.colormaps['hsv']
                color_map = [str(mpl.colors.to_hex(hsv_color_map(k_link / (len(link_names) - 1)))) for k_link in range(len(link_names) + 1)]
            if color_map is not None:
                show_color_map = True
            for idx, link_name in enumerate(link_names):
                vert, faces = self.get_link_mesh(link_name)
                vert, faces = vert[0].cpu().numpy(), faces[0].cpu().numpy()
                data.append(
                    go.Mesh3d(x=vert[:, 0], y=vert[:, 1], z=vert[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], 
                        color=color_map[idx] if show_color_map else color, 
                        opacity=opacity,
                        name=name,
                        legendgroup=name,
                        showlegend=True if idx==0 else False,
                        visible=show,
                        hovertemplate=f"{link_name}"
                    )
                )
        else:
            if show_color_map:
                hsv_color_map = mpl.colormaps['hsv']
                color_map = [str(mpl.colors.to_hex(hsv_color_map(k_link / (len(self.mesh_verts))))) for k_link in range(len(self.mesh_verts))]
            if color_map is not None:
                show_color_map = True
            for idx, link_name in enumerate(self.mesh_verts):
                vert, faces = self.get_link_mesh(link_name)
                vert, faces = vert[0].cpu().numpy(), faces[0].cpu().numpy()
                data.append(
                    go.Mesh3d(x=vert[:, 0], y=vert[:, 1], z=vert[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], 
                        color=color_map[idx] if show_color_map else color, 
                        opacity=opacity,
                        name=name,
                        legendgroup=name,
                        showlegend=True if idx==0 else False,
                        visible=show,
                        hovertemplate=f"{link_name}"
                    )
                )
        if display_joints:
            size = 0.02
            fk = self.robot.forward_kinematics(q[:, 9:])
            for k, v in fk.items():
                mat = v.get_matrix().squeeze(0).detach().cpu().numpy()
                xyz_pos = mat[:3, 3]
                rot_mat = torch.from_numpy(mat[:3, :3])

                # Define the local axes vectors and their colors
                axes_vectors_local = np.array([
                    [size, 0, 0],  # x-axis
                    [0, size, 0],  # y-axis
                    [0, 0, size]   # z-axis
                ])
                
                # Define colors and names for the traces
                colors = ['red', 'green', 'blue']
                axis_names = ['X-axis', 'Y-axis', 'Z-axis']
                
                # Rotate the local axes to the global frame
                axes_vectors_global = rot_mat @ axes_vectors_local
                
                # Add a trace for each axis
                for i, (vec, color, axis_name) in enumerate(zip(axes_vectors_global.T, colors, axis_names)):
                    data.append(go.Cone(
                        x=[xyz_pos[0]],
                        y=[xyz_pos[1]],
                        z=[xyz_pos[2]],
                        u=[vec[0]],
                        v=[vec[1]],
                        w=[vec[2]],
                        sizemode="absolute",
                        sizeref=size,
                        showscale=False,
                        name=f"joint_{k}_{axis_name}",
                        colorscale=[[0, color], [1, color]],
                        showlegend=True,
                        visible=show,
                    ))
        return data

    def get_trimesh_data(self, q=None, show_color_map=False):
        if q is not None: 
            self.update_kinematics(q)
        
        scene = tm.Scene()
        
        # Precompute color map if needed
        mesh_colors = None
        if show_color_map:
            hsv_color_map = mpl.colormaps['hsv']
            mesh_colors = [
            (np.array(mpl.colors.to_rgba(hsv_color_map(k_link / (len(self.mesh_verts) - 1)))) * 255).astype(np.uint8)
            for k_link in range(len(self.mesh_verts))
            ]
        
        for idx, link_name in enumerate(self.mesh_verts):
            vert, faces = self.get_link_mesh(link_name)
            vert, faces = vert.squeeze(0).cpu().numpy(), faces.squeeze(0).cpu().numpy()
            mesh_color = mesh_colors[idx] if mesh_colors is not None else None
            mesh = tm.Trimesh(vertices=vert, faces=faces, vertex_colors=mesh_color, process=False)
            scene.add_geometry(mesh)
        
        vertices = []
        faces = []
        colors = []
        vertex_offset = 0
        for idx, geom in enumerate(scene.geometry.values()):
            if isinstance(geom, tm.Trimesh):
                vertices.append(geom.vertices)
                faces.append(geom.faces + vertex_offset)
                if show_color_map and mesh_colors is not None:
                    mesh_color = mesh_colors[idx][:3]
                    colors.append(np.tile(mesh_color, (len(geom.vertices), 1)))
                vertex_offset += len(geom.vertices)
        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)
        if show_color_map and colors:
            all_colors = np.vstack(colors)
            return tm.Trimesh(vertices=all_vertices, faces=all_faces, vertex_colors=all_colors)
        else:
            return tm.Trimesh(vertices=all_vertices, faces=all_faces)

