from isaacgym import gymapi, gymtorch

import os
import json
import pytorch_kinematics as pk
import time
import numpy as np
from tqdm import tqdm
import torch
from scipy.spatial.transform import Rotation as R
import gc

from utils.constants import DATA_PATH
# from utils_validation.controller import controller


class IsaacValidator:
    def __init__(
        self,
        robot_name,
        batch_size,
        use_gui=False,
        use_controller=False,
        use_stiffness=True,
        robot_friction=3.,
        object_friction=3.,
        steps_per_sec=60,
        grasp_step=50,
        debug_interval=0.01
    ):
        self.gym = gymapi.acquire_gym()

        self.use_gui = use_gui
        self.use_controller = use_controller
        self.use_stiffness = use_stiffness
        self.robot_name = robot_name
        self.batch_size = batch_size
        self.gpu = 0
        self.robot_friction = robot_friction
        self.object_friction = object_friction
        self.steps_per_sec = steps_per_sec
        self.grasp_step = grasp_step if self.use_controller else 10
        self.debug_interval = debug_interval

        json_path = os.path.join(DATA_PATH, "urdf/robot/urdf_assets_meta_isaac.json")
        self.urdf_assets_meta = json.load(open(json_path))
        self.urdf_path = self.urdf_assets_meta['urdf_path'][self.robot_name]
        pk_chain = pk.build_chain_from_urdf(open(self.urdf_path).read()).to(dtype=torch.float32)
        self.joint_orders = [joint.name for joint in pk_chain.get_joints()]

        self.fixed_joints = ['virtual_joint_x', 'virtual_joint_y', 'virtual_joint_z', 'virtual_joint_roll', 'virtual_joint_pitch', 'virtual_joint_yaw']

        self.finger_names = []
        for name in [link.name for link in pk_chain.get_links()]:
            if (name != 'world') and ('virtual' not in name):
                self.finger_names.append(name)

        self.device = 'cuda'
        self.envs = []
        self.robot_handles = []
        self.object_handles = []
        self.robot_asset = None
        self.object_asset = None
        self.object_name = None
        self.object_scale = 1.0
        self.rigid_body_num = None
        self.object_force = None
        self.urdf2isaac_order = None
        self.isaac2urdf_order = None

        self.sim_params = gymapi.SimParams()
        # set common parameters
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.dt = 1 / steps_per_sec
        self.sim_params.substeps = 2
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        # self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # use gravity
        self.sim_params.use_gpu_pipeline = True
        
        # set PhysX-specific parameters
        self.sim_params.physx.use_gpu = True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_threads = 4
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.contact_offset = 0.01
        self.sim_params.physx.rest_offset = 0.0
        self.sim_params.physx.contact_collection = gymapi.ContactCollection.CC_LAST_SUBSTEP
        # self.sim_params.physx.max_depenetration_velocity = 10.0

        # Increase the buffer size to handle "exploding" physics
        self.sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024 
        # Increase the general memory allocation multiplier
        self.sim_params.physx.default_buffer_size_multiplier = 5.0

        # create sim
        self.sim = self.gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        # self._rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)

        self.viewer = None
        if self.use_gui:
            self.has_viewer = True
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 1920
            self.camera_props.height = 1080
            self.camera_props.use_collision_geometry = True
            self.viewer = self.gym.create_viewer(self.sim, self.camera_props)
            self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(0.5, 0, 0), gymapi.Vec3(0, 0, 0))
        else:
            self.has_viewer = False

        self.robot_asset_options = gymapi.AssetOptions()
        self.robot_asset_options.disable_gravity = True
        self.robot_asset_options.fix_base_link = True
        self.robot_asset_options.collapse_fixed_joints = True
        # self.robot_asset_options.vhacd_enabled = True

        self.object_asset_options = gymapi.AssetOptions()
        self.object_asset_options.override_com = True
        self.object_asset_options.override_inertia = True
        self.object_asset_options.density = 500
        # self.object_asset_options.vhacd_enabled = True

    def set_asset(self, object_path, object_file, scale=1.0):
        robot_path = os.path.dirname(self.urdf_path)
        robot_file = os.path.basename(self.urdf_path)
        self.object_name = '+'.join(object_file.split('/')[:-1])
        self.robot_asset = self.gym.load_asset(self.sim, robot_path, robot_file, self.robot_asset_options)
        self.object_asset = self.gym.load_asset(self.sim, object_path, object_file, self.object_asset_options)
        self.rigid_body_num = (self.gym.get_asset_rigid_body_count(self.robot_asset)
                               + self.gym.get_asset_rigid_body_count(self.object_asset))
        if not type(scale) == torch.Tensor:
            scale = torch.tensor([scale], device=self.device).repeat(self.batch_size)
        self.object_scale = scale

    def create_envs(self):
        for env_idx in range(self.batch_size):
            env = self.gym.create_env(
                self.sim,
                gymapi.Vec3(-1, -1, -1),
                gymapi.Vec3(1, 1, 1),
                int(self.batch_size ** 0.5)
            )
            self.envs.append(env)

            # draw world frame
            if self.has_viewer:
                x_axis_dir = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
                x_axis_color = np.array([1, 0, 0], dtype=np.float32)
                self.gym.add_lines(self.viewer, env, 1, x_axis_dir, x_axis_color)
                y_axis_dir = np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)
                y_axis_color = np.array([0, 1, 0], dtype=np.float32)
                self.gym.add_lines(self.viewer, env, 1, y_axis_dir, y_axis_color)
                z_axis_dir = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
                z_axis_color = np.array([0, 0, 1], dtype=np.float32)
                self.gym.add_lines(self.viewer, env, 1, z_axis_dir, z_axis_color)

            # object actor setting
            object_handle = self.gym.create_actor(
                env,
                self.object_asset,
                gymapi.Transform(),
                f'object_{env_idx}',
                env_idx
            )
            self.gym.set_actor_scale(env, object_handle, self.object_scale[env_idx].item())  # scale the object
            self.object_handles.append(object_handle)
        
            object_shape_properties = self.gym.get_actor_rigid_shape_properties(env, object_handle)
            for i in range(len(object_shape_properties)):
                object_shape_properties[i].friction = self.object_friction
            self.gym.set_actor_rigid_shape_properties(env, object_handle, object_shape_properties)

            # robot actor setting
            robot_handle = self.gym.create_actor(
                env,
                self.robot_asset,
                gymapi.Transform(),
                f'robot_{env_idx}',
                env_idx
            )
            self.robot_handles.append(robot_handle)

            robot_properties = self.gym.get_actor_dof_properties(env, robot_handle)
            robot_properties["driveMode"].fill(gymapi.DOF_MODE_POS)
            robot_properties["damping"].fill(200)
            if self.use_stiffness:
                robot_properties["stiffness"].fill(10000)
            else:
                robot_properties["stiffness"].fill(1000)
            self.gym.set_actor_dof_properties(env, robot_handle, robot_properties)

            robot_shape_properties = self.gym.get_actor_rigid_shape_properties(env, robot_handle)
            for i in range(len(robot_shape_properties)):
                robot_shape_properties[i].friction = self.robot_friction
            self.gym.set_actor_rigid_shape_properties(env, robot_handle, robot_shape_properties)

            # Robot color: white
            if self.use_gui:
                robot_rigid_body_count = self.gym.get_actor_rigid_body_count(env, robot_handle)
                for i in range(robot_rigid_body_count):
                    self.gym.set_rigid_body_color(env, robot_handle, i, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.8, 0.8, 0.8))


        self.object_force = torch.full((self.batch_size,), 0.05, device=self.device)

        self.urdf2isaac_order = torch.zeros(len(self.joint_orders), dtype=torch.int32, device=self.device)
        self.isaac2urdf_order = torch.zeros(len(self.joint_orders), dtype=torch.int32, device=self.device)
        for urdf_idx, joint_name in enumerate(self.joint_orders):
            isaac_idx = self.gym.find_actor_dof_index(self.envs[0], self.robot_handles[0], joint_name, gymapi.DOMAIN_ACTOR)
            self.urdf2isaac_order[isaac_idx] = urdf_idx
            self.isaac2urdf_order[urdf_idx] = isaac_idx
        
        self.finger_indices = torch.zeros(len(self.finger_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.finger_names):
            idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_handles[0], name, gymapi.DOMAIN_ENV)
            self.finger_indices[i] = idx

        self.gym.prepare_sim(self.sim)
        # 1. Root State (Base position/rotation for all actors)
        self._root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(self._root_state)
        
        # 2. DOF State (Joint positions/velocities for all actors)
        self._dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(self._dof_state)
        
        # 3. Rigid Body State (For checking success)
        self._rb_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(self._rb_state)

        # Acquire the contact force tensor
        self._contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(self._contact_forces) # Shape: (total_rigid_bodies, 3)

        # Refresh initially to populate them
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.saved_root_tensor = self.root_state.clone()
        self.saved_dof_tensor = self.dof_state.clone()

    def set_actor_pose_dof(self, q):
        self.gym.prepare_sim(self.sim)

        # set all actors to origin
        self.root_state[:] = self.saved_root_tensor
        self.gym.set_actor_root_state_tensor(self.sim, self._root_state)

        init_q, target_q = q.clone().detach(), q.clone().detach()
        # if self.use_controller:
        #     outer_q_joints, inner_q_joints = controller(self.robot_name, q[:, 6:])
        #     init_q[:, 6:] = outer_q_joints
        #     target_q[:, 6:] = inner_q_joints

        dof_view = self.dof_state.view(self.batch_size, -1, 2)
        dof_view[:, :, 0].copy_(init_q[:, self.urdf2isaac_order]) 
        dof_view[:, :, 1].fill_(0.0) # Zero velocity
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_view.flatten()))

        target_tensor = target_q[:, self.urdf2isaac_order].flatten()
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(target_tensor))

        if self.has_viewer:
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            t = time.time()
            while time.time() - t < self.debug_interval * self.grasp_step:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, render_collision=True)

    def run_sim(self, verbose=False, desc="[Isaac/Run]"):
        # controller phase
        for step in range(self.grasp_step):
            self.gym.simulate(self.sim)

            if self.has_viewer:
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
                self.gym.fetch_results(self.sim, True)
                t = time.time()
                while time.time() - t < self.debug_interval:
                    self.gym.step_graphics(self.sim)
                    self.gym.draw_viewer(self.viewer, self.sim, render_collision=True)


        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # start_pos = gymtorch.wrap_tensor(self._rigid_body_states)[::self.rigid_body_num, :3].clone()
        start_pos = self.rb_state[::self.rigid_body_num, :3].clone()

        force_tensor = torch.zeros([len(self.envs), self.rigid_body_num, 3], device=self.device)  # env, rigid_body, xyz
        x_pos_force = force_tensor.clone()
        x_pos_force[:, 0, 0] = self.object_force
        x_neg_force = force_tensor.clone()
        x_neg_force[:, 0, 0] = -self.object_force
        y_pos_force = force_tensor.clone()
        y_pos_force[:, 0, 1] = self.object_force
        y_neg_force = force_tensor.clone()
        y_neg_force[:, 0, 1] = -self.object_force
        z_pos_force = force_tensor.clone()
        z_pos_force[:, 0, 2] = self.object_force
        z_neg_force = force_tensor.clone()
        z_neg_force[:, 0, 2] = -self.object_force
        force_list = [x_pos_force, y_pos_force, z_pos_force, x_neg_force, y_neg_force, z_neg_force]

        # force phase
        sim_steps = range(self.steps_per_sec * 6)
        iterator = tqdm(sim_steps, desc=desc) if verbose else sim_steps
        for step in iterator:
            self.gym.apply_rigid_body_force_tensors(self.sim,
                                                    gymtorch.unwrap_tensor(force_list[step // self.steps_per_sec]),     # Force
                                                    None,                                                               # Torque
                                                    gymapi.ENV_SPACE)
            self.gym.simulate(self.sim)
            

            if self.has_viewer:
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
                self.gym.fetch_results(self.sim, True)
                t = time.time()
                while time.time() - t < self.debug_interval:
                    self.gym.step_graphics(self.sim)
                    self.gym.draw_viewer(self.viewer, self.sim, render_collision=True)

        # Object Pose
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        end_pos = self.rb_state[::self.rigid_body_num, :3].clone()
        distance = (end_pos - start_pos).norm(dim=-1)

        # Contact Forces
        self.gym.refresh_net_contact_force_tensor(self.sim)
        contact_forces_per_env = self.contact_forces.view(self.batch_size, self.rigid_body_num, 3)
        link_forces = contact_forces_per_env[:, self.finger_indices, :]
        # contact_forces_magnitude = contact_forces_per_env.norm(dim=-1).max(dim=-1).values  # (num_envs, num_bodies_per_env)
        link_forces_mag = torch.norm(link_forces, dim=-1) # (batch_size, num_links)
        too_much_force = torch.any(link_forces_mag > 500, dim=-1)  # (batch_size,)
        
        success = (distance <= 0.02) & (~too_much_force)

        # apply inverse object transform to robot to get new joint value
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        object_pose = self.rb_state.clone()[::self.rigid_body_num, :7]  # batch_size, 7 (xyz + quat)
        object_transform = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        object_transform[:, :3, 3] = object_pose[:, :3]
        object_transform[:, :3, :3] = torch.tensor(R.from_quat(object_pose[:, 3:7].cpu().numpy()).as_matrix(), device=self.device)

        self.gym.refresh_dof_state_tensor(self.sim)
        dof_states = self.dof_state.clone().reshape(len(self.envs), -1, 2)[:, :, 0]  # batch_size, DOF (xyz + euler + joint)
        robot_transform = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        robot_transform[:, :3, 3] = dof_states[:, :3]
        robot_transform[:, :3, :3] = torch.tensor(R.from_euler('XYZ', dof_states[:, 3:6].cpu().numpy()).as_matrix(), device=self.device)

        robot_transform = torch.linalg.inv(object_transform) @ robot_transform
        dof_states[:, :3] = robot_transform[:, :3, 3]
        dof_states[:, 3:6] = torch.tensor(R.from_matrix(robot_transform[:, :3, :3].cpu().numpy()).as_euler('XYZ'))
        q_isaac = dof_states[:, self.isaac2urdf_order].cpu()

        return success, q_isaac

    def reset(self):
        """
        Resets the simulation to the initial state and clears memory buffers.
        """
        # 1. Reset Physics State to "Home" Positions
        self.root_state[:] = self.saved_root_tensor
        self.dof_state[:] = self.saved_dof_tensor
        
        # Apply the reset immediately to PhysX
        self.gym.set_actor_root_state_tensor(self.sim, self._root_state)
        self.gym.set_dof_state_tensor(self.sim, self._dof_state)
        
        # 2. Clear Forces
        # Apply zero force to stop any lingering physics momentum
        self.gym.apply_rigid_body_force_tensors(
            self.sim, 
            gymtorch.unwrap_tensor(torch.zeros((len(self.envs), self.rigid_body_num, 3), device='cuda')), 
            None, 
            gymapi.ENV_SPACE
        )

        # 3. Clear Internal PyTorch Buffers (Crucial for VRAM)
        # Force a simulation step to flush the physics pipeline buffers
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # 4. Python/PyTorch Garbage Collection
        # This clears the "computation graph" history if any leaked
        torch.cuda.empty_cache()
        gc.collect()

    def destroy(self):
        for env in self.envs:
            self.gym.destroy_env(env)
        self.gym.destroy_sim(self.sim)
        if self.has_viewer:
            self.gym.destroy_viewer(self.viewer)
        del self.gym
        gc.collect()
        torch.cuda.empty_cache()