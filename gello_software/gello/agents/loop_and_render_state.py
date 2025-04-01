import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import pickle
import numpy as np
import sys
from gello.agents.agent import Agent
from gello.robots.dynamixel import DynamixelRobot
scene_path = "/mnt/data-3/users/mengxinpan/SplatSim/"
diffusion_policy_path = "/mnt/data-3/users/mengxinpan/diffusion_policy/"
# render_fk_path = "/mnt/data-3/users/mengxinpan/SplatSim/render_fk_all_highres.py"
# gaussian_path = "/mnt/data-3/users/mengxinpan/SplatSim/gaussian_renderer"
sys.path.append(diffusion_policy_path)
# sys.path.append(render_fk_path)
sys.path.append(scene_path)
# sys.path.append(gaussian_path)
import torch
import dill
import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import pybullet as p
import cv2
import pybullet_data
import copy
import einops
from pathlib import Path
from einops import einsum
# from render_fk_all_highres import transform_shs
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from tqdm import tqdm
from os import makedirs
import torchvision
import yaml
from utils_fk import compute_transformation
from e3nn import o3
from einops import einsum
from argparse import ArgumentParser  # A
# from render_fk_all_highres import transform_shs
@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    """The joint ids of GELLO (not including the gripper). Usually (1, 2, 3 ...)."""

    joint_offsets: Sequence[float]
    """The joint offsets of GELLO. There needs to be a joint offset for each joint_id and should be a multiple of pi/2."""

    joint_signs: Sequence[int]
    """The joint signs of GELLO. There needs to be a joint sign for each joint_id and should be either 1 or -1.

    This will be different for each arm design. Refernce the examples below for the correct signs for your robot.
    """

    gripper_config: Tuple[int, int, int]
    """The gripper config of GELLO. This is a tuple of (gripper_joint_id, degrees in open_position, degrees in closed_position)."""

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(
        self, port: str = "/dev/ttyUSB1", start_joints: Optional[np.ndarray] = None
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
        )


PORT_CONFIG_MAP: Dict[str, DynamixelRobotConfig] = {
    # xArm
    # "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0": DynamixelRobotConfig(
    #     joint_ids=(1, 2, 3, 4, 5, 6, 7),
    #     joint_offsets=(
    #         2 * np.pi / 2,
    #         2 * np.pi / 2,
    #         2 * np.pi / 2,
    #         2 * np.pi / 2,
    #         -1 * np.pi / 2 + 2 * np.pi,
    #         1 * np.pi / 2,
    #         1 * np.pi / 2,
    #     ),
    #     joint_signs=(1, 1, 1, 1, 1, 1, 1),
    #     gripper_config=(8, 279, 279 - 50),
    # ),
    # panda
    # "/dev/cu.usbserial-FT3M9NVB": DynamixelRobotConfig(
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6, 7),
        joint_offsets=(
            3 * np.pi / 2,
            2 * np.pi / 2,
            1 * np.pi / 2,
            4 * np.pi / 2,
            -2 * np.pi / 2 + 2 * np.pi,
            3 * np.pi / 2,
            4 * np.pi / 2,
        ),
        joint_signs=(1, -1, 1, 1, 1, -1, 1),
        gripper_config=(8, 195, 152),
    ),
    # Left UR
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            0,
            1 * np.pi / 2 + np.pi,
            np.pi / 2 + 0 * np.pi,
            0 * np.pi + np.pi / 2,
            np.pi - 2 * np.pi / 2,
            -1 * np.pi / 2 + 2 * np.pi,
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 20, -22),
    ),
    # Right UR
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
            np.pi + 0 * np.pi,
            2 * np.pi + np.pi / 2,
            2 * np.pi + np.pi / 2,
            2 * np.pi + np.pi / 2,
            1 * np.pi,
            3 * np.pi / 2,
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 286, 248),
    ),
    # Custom UR
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8ISNEF-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=(
         3*np.pi/2, 6*np.pi/2, 4*np.pi/2, 4*np.pi/2, 1*np.pi/2, 1*np.pi/2 
        ),
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 110, 68),
    ),
}

class DiffusionAgent(Agent):
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
        output_dir: str = "./output_images",
        model_path: str = "/shared_disk/datasets/public_datasets/SplatSim/output/robot_iphone",
        iteration: int = -1,
        traj_folder: str = "/mnt/data-3/users/mengxinpan/SplatSim/bc_data_run_by_render/",
        objects: str = "plastic_apple"
    ):
        # Diffusion Policy Setup
        print('Loading policy')
        ckpt_path = '/mnt/data-3/users/mengxinpan/code/diffusion_policy/data/outputs/2025.03.25/09.26.14_train_diffusion_unet_image_real_image/checkpoints/latest.ckpt'
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        if 'diffusion' in cfg.name:
            self.policy: BaseImagePolicy = workspace.model
            if cfg.training.use_ema:
                self.policy = workspace.ema_model
            device = torch.device('cuda')
            self.policy.eval().to(device)
            self.policy.num_inference_steps = 16
            self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1

        # PyBullet Setup
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.dummy_robot = p.loadURDF("/mnt/data-3/users/mengxinpan/SplatSim/pybullet-playground_2/urdf/sisbot.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(self.dummy_robot, [0, 0, -0.1], [0, 0, 0, 1])
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        initial_joint_state = [0, -1.57, 1.57, -1.57, -1.57, 0]
        self.initial_joint_state = initial_joint_state
        joint_signs = [1, 1, 1, 1, 1, 1]
        for i in range(1, 7):
            p.resetJointState(self.dummy_robot, i, initial_joint_state[i-1] * joint_signs[i-1])
        
        self.initial_joints = []
        for joint_index in range(19):
            link_state = p.getLinkState(self.dummy_robot, joint_index, computeForwardKinematics=True)
            self.initial_joints.append(link_state)
        # Gaussian Rendering Setup
        self.model_path = model_path
        self.iteration = iteration
        self.traj_folder = traj_folder
        self.robot_name = model_path.split('/')[-1]
        self.object_splat_folder = model_path.replace(self.robot_name, '')
        self.object_list = objects.split(' ') if objects else []

        # Create and configure ArgumentParser for ModelParams and PipelineParams
        parser = ArgumentParser(description="Diffusion Agent with Gaussian Rendering")
        model_params = ModelParams(parser)
        pipeline_params = PipelineParams(parser)
        # Do not manually add --model_path or other arguments here; ModelParams already defines them
        args = parser.parse_args([
            "--model_path", model_path,
            # "--iteration", str(iteration),
            "--resolution","1",
            "--source_path", "/shared_disk/datasets/public_datasets/SplatSim/test_data/robot_iphone",  # Placeholder, required by ModelParams
            "--sh_degree", "3"    # Default value
        ])
        dataset = model_params.extract(args)

        self.gaussians = GaussianModel(args.sh_degree)
        self.scene = Scene(dataset, self.gaussians, load_iteration=int(iteration), shuffle=False)
        self.pipeline = pipeline_params.extract(args)
        self.bg_color = [1, 1, 1]  # White background
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.gaussians_backup = copy.deepcopy(self.gaussians)
        self.object_gaussians = [GaussianModel(3) for _ in range(len(self.object_list))]
        for i in range(len(self.object_list)):
            self.object_gaussians[i].load_ply(f"{self.object_splat_folder}{self.object_list[i]}/point_cloud/iteration_7000/point_cloud.ply")
        self.object_gaussians_backup = copy.deepcopy(self.object_gaussians)
        with open('/mnt/data-3/users/mengxinpan/SplatSim/object_configs/objects.yaml', 'r') as f:
            self.object_config = yaml.safe_load(f)
            self.robot_transformation = self.object_config[self.robot_name]['transformation']['matrix']

        # Agent State
        self.cur_index = -1
        self.cur_joint_list = None
        self.last_image_obs = None
        self.last_image_obs_1 = None
        self.last_state_obs = None
        self.policy.reset()
        self.cur_total_steps = 0
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print('Policy and rendering setup complete')
        
    def act(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        for i in range(1, 7):
            p.resetJointState(self.dummy_robot, i, obs_dict['joint_positions'][i-1])
        # p.stepSimulation()
        ee_pos, ee_quat = p.getLinkState(self.dummy_robot, 6)[0], p.getLinkState(self.dummy_robot, 6)[1]
        ee_euler = p.getEulerFromQuaternion(ee_quat)
        obs_dict['state'] = np.array([ee_pos[0], ee_pos[1], ee_pos[2], ee_euler[0], ee_euler[1], ee_euler[2], obs_dict['gripper_position'][0]])

        image = cv2.resize(obs_dict['wrist_rgb'], (320, 240)).transpose(2, 0, 1)
        image = np.expand_dims(image, axis=(0, 1))
        image_tensor = torch.from_numpy(image).float() / 255.0
        image = image_tensor + 0.1 * torch.randn_like(image_tensor)

        image_2 = cv2.resize(obs_dict['base_rgb'], (320, 240)).transpose(2, 0, 1)
        image_2 = np.expand_dims(image_2, axis=(0, 1))
        image_2_tensor = torch.from_numpy(image_2).float() / 255.0
        image_2 = image_2_tensor + 0.1 * torch.randn_like(image_2_tensor)

        if self.last_image_obs is None:
            self.last_image_obs = image
            self.last_image_obs_1 = image_2
            # self.last_state_obs = torch.from_numpy(obs_dict['joint_positions'].reshape(1, 1, 7)).float()
            self.last_state_obs = torch.from_numpy(obs_dict['state'].reshape(1, 1, 7)).float()

        cur_state_obs = torch.from_numpy(obs_dict['state'].reshape(1, 1, 7)).float()
        # cur_state_obs = torch.from_numpy(obs_dict['joint_positions'].reshape(1, 1, 7)).float()
        image_out = torch.cat((self.last_image_obs, image), dim=1)
        image_out_1 = torch.cat((self.last_image_obs_1, image_2), dim=1)
        state_out = torch.cat((self.last_state_obs, cur_state_obs), dim=1)

        if self.cur_index == -1 or self.cur_joint_list is None:
            print('New prediction step')
            obs_dict_1 = {
                'image_1': image_out,
                'image_2': image_out_1,
                'state': state_out
            } # state_out是ee pos、ee quad与gripper
            result = self.policy.predict_action(obs_dict_1)
            self.cur_index = 0
            self.cur_joint_list = result['action'][0].detach().cpu().numpy()
        # action_pred = result['action'][0][0] = tensor([ 0.0337, -0.3405,  0.3198, -0.9509,  0.4033, -0.4675,  0.7788],
        action_pred = self.cur_joint_list[self.cur_index]
        ee_pose = action_pred[:3]
        ee_quat = p.getQuaternionFromEuler(action_pred[3:6])
        '''此处替换'''
        dummy_joint_pos = p.calculateInverseKinematics(
            self.dummy_robot, 6, ee_pose, ee_quat,
            residualThreshold=0.00001, maxNumIterations=100000,
            lowerLimits=[obs_dict['joint_positions'][k] - np.pi for k in range(6)],
            upperLimits=[obs_dict['joint_positions'][k] + np.pi for k in range(6)],
            jointRanges=[12.566] * 6,
            restPoses=self.initial_joint_state
        )  # len（dummy-joint pos）=12
        # if len(action_pred) == 6:  # 只有 6 个关节角度
        #     dummy_joint_pos = action_pred
        #     gripper_pos = obs_dict['gripper_position'][0]  # 夹爪位置保持不变
        # else:  # 长度为 7，包含夹爪位置
        #     dummy_joint_pos = action_pred[:6]
        #     gripper_pos = action_pred[-1]
            
        for i in range(1, 7):
            p.resetJointState(self.dummy_robot, i, dummy_joint_pos[i-1])
        # p.stepSimulation()

        joint_diff = np.array(dummy_joint_pos)[:6] - np.array(obs_dict['joint_positions'])[:6]
        if np.linalg.norm(joint_diff) < 0.01:
            self.cur_index += 1
            self.cur_total_steps = 0
        if self.cur_index >= len(self.cur_joint_list):
            self.cur_index = -1
            self.cur_joint_list = None
        print(self.cur_index)
        self.last_image_obs = image
        self.last_image_obs_1 = image_2
        self.last_state_obs = cur_state_obs

        return np.append(dummy_joint_pos[:6], action_pred[-1])

    def get_transformation_list(self, new_joint_poses):
        joint_poses = [0] * 19  # Initialize with 19 joints (adjust if needed)
        for i in range(min(len(new_joint_poses), len(joint_poses))):
            joint_poses[i] = new_joint_poses[i]
        for i in range(len(joint_poses)):
            p.resetJointState(self.dummy_robot, i, joint_poses[i])
        new_joints = []
        for joint_index in range(19):
            link_state = p.getLinkState(self.dummy_robot, joint_index, computeForwardKinematics=True)
            new_joints.append(link_state)
        transformations_list = []

        for joint_index in range(19):
            input_1 = (self.initial_joints[joint_index][0][0], self.initial_joints[joint_index][0][1], self.initial_joints[joint_index][0][2], np.array(self.initial_joints[joint_index][1]))
            input_2 = (new_joints[joint_index][0][0], new_joints[joint_index][0][1], new_joints[joint_index][0][2], np.array(new_joints[joint_index][1]))
            r_rel, t = compute_transformation(input_1, input_2)
            r_rel = torch.from_numpy(r_rel).to(device='cuda').float()
            t = torch.from_numpy(t).to(device='cuda').float()
            transformations_list.append((r_rel, t))
        return transformations_list

    def render_image(self,joint_positions,step, view_idx=5):  # view_idx: 5 for wrist, 254 for base
        view = self.scene.getTrainCameras()[view_idx]  # Select specific camera view
        # joint_positions =  np.array([0.47134819, -0.84844438,  1.24167812, -1.89515025, -1.61576983, 0.59970792])
        cur_joint = [0] + joint_positions[:6].tolist()  # TODO 转为20，不过按理说不会出问题，因为夹爪应该不怎么影响
        transformations_list = self.get_transformation_list(cur_joint)

        # Placeholder for object positions (assuming static for now)
        #TODO 改为从仿真中读取
        cur_object_position_list = [torch.tensor([0.4, 0.4, -0.01], device="cuda").float() for _ in self.object_list]
        cur_object_rotation_list = [torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").float() for _ in self.object_list]
        segmented_list, xyz = self.get_segmented_indices(self.gaussians_backup, self.robot_transformation)
        xyz, rot, opacity, shs_featrest, shs_dc = self.transform_means(self.gaussians_backup, xyz, segmented_list, transformations_list, self.robot_transformation)
        xyz_obj_list, rot_obj_list, opacity_obj_list, scales_obj_list, features_dc_obj_list, features_rest_obj_list = [], [], [], [], [], []
        for i in range(len(self.object_list)):
            xyz_obj, rot_obj, opacity_obj, scales_obj, features_dc_obj, features_rest_obj = self.transform_object(
                self.object_gaussians_backup[i], self.object_config[self.object_list[i]], cur_object_position_list[i], cur_object_rotation_list[i], self.robot_transformation
            )
            xyz_obj_list.append(xyz_obj)
            rot_obj_list.append(rot_obj)
            opacity_obj_list.append(opacity_obj)
            scales_obj_list.append(scales_obj)
            features_dc_obj_list.append(features_dc_obj)
            features_rest_obj_list.append(features_rest_obj)

        with torch.no_grad():
            self.gaussians._xyz = torch.cat([xyz] + xyz_obj_list, dim=0)
            self.gaussians._rotation = torch.cat([rot] + rot_obj_list, dim=0)
            self.gaussians._opacity = torch.cat([opacity] + opacity_obj_list, dim=0)
            self.gaussians._features_rest = torch.cat([shs_featrest] + features_rest_obj_list, dim=0)
            self.gaussians._features_dc = torch.cat([shs_dc] + features_dc_obj_list, dim=0)
            self.gaussians._scaling = torch.cat([self.gaussians_backup._scaling] + scales_obj_list, dim=0)
            rendering = render(view, self.gaussians, self.pipeline, self.background)["render"]
            torchvision.utils.save_image(rendering, f"/mnt/data-3/users/mengxinpan/SplatSim/output_images/step{step}_view{view_idx}.png")
            print("Rendered image saved")
            rendering = rendering.cpu().numpy().transpose(1, 2, 0) * 255.0  # Convert to HWC, [0, 255]
            rendering = rendering.astype(np.uint8)
            return rendering

    # Include transform_means, transform_object, get_segmented_indices, transform_shs from the new code
    def transform_means(self, pc, xyz, segmented_list, transformations_list, robot_transformation):
        Trans = torch.tensor(robot_transformation).to(device=xyz.device).float()
        scale_robot = torch.pow(torch.linalg.det(Trans[:3, :3]), 1/3)
        rotation_matrix = Trans[:3, :3] / scale_robot
        translation = Trans[:3, 3]
        inv_transformation_matrix = torch.inverse(Trans)
        inv_rotation_matrix = inv_transformation_matrix[:3, :3] 
        inv_translation = inv_transformation_matrix[:3, 3]
        
        # rot = copy.deepcopy(pc.get_rotation)
        rot = pc.get_rotation
        opacity = pc.get_opacity_raw
        with torch.no_grad():
            shs_dc = copy.deepcopy(pc._features_dc)
            shs_featrest = copy.deepcopy(pc._features_rest)

        for joint_index in range(7):
            r_rel, t = transformations_list[joint_index]
            segment = segmented_list[joint_index]
            transformed_segment = torch.matmul(r_rel, xyz[segment].T).T + t
            xyz[segment] = transformed_segment
            
            # Defining rotation matrix for the covariance
            rot_rotation_matrix = (inv_rotation_matrix*scale_robot) @ r_rel @ rotation_matrix
            
            tranformed_rot = rot[segment]  
            tranformed_rot = o3.quaternion_to_matrix(tranformed_rot) ### --> zyx    
            
            transformed_rot = rot_rotation_matrix  @ tranformed_rot # shape (N, 3, 3)
            
            transformed_rot = o3.matrix_to_quaternion(transformed_rot)
            
            rot[segment] = transformed_rot

            #transform the shs features
            shs_feat = shs_featrest[segment]
            shs_dc_segment = shs_dc[segment]
            shs_feat = self.transform_shs(shs_feat, rot_rotation_matrix)
            # print('shs_feat : ', shs_feat.shape)
            with torch.no_grad():
                shs_featrest[segment] = shs_feat
            # shs_dc[segment] = shs_dc_segment
            # shs_featrest[segment] = torch.zeros_like(shs_featrest[segment])
        cnt = 7
        for joint_index in [8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 12]:
            r_rel, t = transformations_list[joint_index]
            segment = segmented_list[cnt]
            transformed_segment = torch.matmul(r_rel, xyz[segment].T).T + t
            xyz[segment] = transformed_segment
            
            # Defining rotation matrix for the covariance
            rot_rotation_matrix = (inv_rotation_matrix*scale_robot) @ r_rel @ rotation_matrix
            
            tranformed_rot = rot[segment]  
            tranformed_rot = o3.quaternion_to_matrix(tranformed_rot) ### --> zyx    
            
            transformed_rot = rot_rotation_matrix  @ tranformed_rot # shape (N, 3, 3)
            
            transformed_rot = o3.matrix_to_quaternion(transformed_rot)
            
            rot[segment] = transformed_rot

            #transform the shs features
            shs_feat = shs_featrest[segment]
            shs_dc_segment = shs_dc[segment]
            shs_feat = self.transform_shs(shs_feat, rot_rotation_matrix)
            # print('shs_feat : ', shs_feat.shape)
            with torch.no_grad():
                shs_featrest[segment] = shs_feat
            # shs_dc[segment] = shs_dc_segment
            # shs_featrest[segment] = torch.zeros_like(shs_featrest[segment])
            cnt += 1
            
        #transform_back
        xyz = torch.matmul(inv_rotation_matrix, xyz.T).T + inv_translation
        
            
        return xyz, rot, opacity, shs_featrest, shs_dc

    def transform_object(self, pc, object_config, pos, quat, robot_transformation):
        Trans_canonical = torch.from_numpy(np.array(object_config['transformation']['matrix'])).to(device=pc.get_xyz.device).float() # shape (4, 4)

        
        
        rotation_matrix_c = Trans_canonical[:3, :3]
        translation_c = Trans_canonical[:3, 3]
        scale_obj = torch.pow(torch.linalg.det(rotation_matrix_c), 1/3)

        
        Trans_robot = torch.tensor(robot_transformation).to(device=pc.get_xyz.device).float()

        
        rotation_matrix_r = Trans_robot[:3, :3]
        scale_r = torch.pow(torch.linalg.det(rotation_matrix_r), 1/3)

        translation_r = Trans_robot[:3, 3]

        inv_transformation_r = torch.inverse(Trans_robot)
        inv_rotation_matrix_r = inv_transformation_r[:3, :3]
        inv_translation_r = inv_transformation_r[:3, 3]
        inv_scale = torch.pow(torch.linalg.det(inv_rotation_matrix_r), 1/3)

        # print('scale_obj : ', scale_obj)
        # print('inv_scale : ', inv_scale)
        
        xyz_obj = pc.get_xyz
        rotation_obj = pc.get_rotation
        opacity_obj = pc.get_opacity_raw
        scales_obj = pc.get_scaling
        scales_obj = scales_obj * scale_obj * inv_scale 
        scales_obj = torch.log(scales_obj)

        with torch.no_grad():
            features_dc_obj = copy.deepcopy(pc._features_dc)
            features_rest_obj = copy.deepcopy(pc._features_rest)
        
        #transform the object to the canonical frame
        xyz_obj = torch.matmul(rotation_matrix_c, xyz_obj.T).T + translation_c
        
        
        rot_rotation_matrix = ( inv_rotation_matrix_r/inv_scale) @ o3.quaternion_to_matrix(quat)  @  (rotation_matrix_c/scale_obj)
        rotation_obj_matrix = o3.quaternion_to_matrix(rotation_obj)
        rotation_obj_matrix = rot_rotation_matrix @ rotation_obj_matrix 
        rotation_obj = o3.matrix_to_quaternion(rotation_obj_matrix) 
        
        
        # aabb = ((-0.10300000149011612, -0.17799999701976776, -0.0030000000000000027), (0.10300000149011612, 0.028000000372529033, 0.022999999552965167))
        aabb = object_config['aabb']['bounding_box']
        #segment according to axis aligned bounding box
        segmented_indices = ((xyz_obj[:, 0] > aabb[0][0]) & (xyz_obj[:, 0] < aabb[1][0]) & (xyz_obj[:, 1] > aabb[0][1] ) & (xyz_obj[:, 1] < aabb[1][1]) & (xyz_obj[:, 2] > aabb[0][2] ) & (xyz_obj[:, 2] < aabb[1][2]))
        

        #offset the object by the position and rotation
        xyz_obj = torch.matmul(o3.quaternion_to_matrix(quat), xyz_obj.T).T + pos
        # xyz_obj = xyz_obj + pos
        
        xyz_obj = torch.matmul(inv_rotation_matrix_r, xyz_obj.T).T + inv_translation_r

        xyz_obj = xyz_obj[segmented_indices]
        rotation_obj = rotation_obj[segmented_indices]
        opacity_obj = opacity_obj[segmented_indices]
        scales_obj = scales_obj[segmented_indices]
        # cov3D_obj = cov3D_obj[segmented_indices]
        features_dc_obj = features_dc_obj[segmented_indices]
        features_rest_obj = features_rest_obj[segmented_indices]
        features_rest_obj= self.transform_shs( features_rest_obj, rot_rotation_matrix)
        # features_rest_obj = torch.zeros_like(features_rest_obj)
        
        return xyz_obj, rotation_obj, opacity_obj, scales_obj, features_dc_obj, features_rest_obj

    def get_segmented_indices(self, pc, robot_transformation):
        # [Copy from new code, unchanged]
        torch.cuda.empty_cache()
        means3D = pc.get_xyz # 3D means shape (N, 3)
        
        # Defining a cube in Gaussian space to segment out the robot
        xyz = pc.get_xyz # shape (N, 3)


        Trans = torch.tensor(robot_transformation).to(device=means3D.device).float() # shape (4, 4)
        
        #define a transformation matrix according to 90 degree rotation about z axis
        temp_matrix = torch.tensor([[0, -1, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]).to(device=means3D.device).float() # shape (4, 4)
        
        Trans = torch.matmul(temp_matrix, Trans)
        
        R = Trans[:3, :3]
        translation = Trans[:3, 3]
        
        
        points = copy.deepcopy(means3D)
        #transform the points to the new frame
        points = torch.matmul(R, points.T).T + translation
        
        
        centers = torch.tensor([[0, 0, 0.0213], [-0.0663-0.00785, 0 , .0892], [-0.0743, 0, .5142], [-0.0743 +0.0174 -0.00785, 0.39225, .5142], [-0.0743 +0.0174-0.0531, 0.04165+0.39225+0.00785, .5142], [-0.0743 +0.0174-0.0531, 0.04165+0.39225+0.0531 , .5142 -0.04165-0.00785]]) # length = 6
        centers = centers.to(device=xyz.device)
        segmented_points = []
        
        # Box condition
        box_condition = ((points[:, 0] > -0.25) * (points[:, 0] < 0.2) * (points[:, 1] > -0.3) * (points[:, 1] < 0.6) * (points[:, 2] > 0.0) * (points[:, 2] < 0.6))
        
        
        # Segment Base
        condition = torch.where((points[:, 2] < centers[0, 2]) * box_condition)[0]
        segmented_points.append(condition)
        
        # Segment Link 1
        condition = torch.where(((points[:, 2] > centers[0, 2])*(points[:, 0] > centers[1, 0])* (points[:, 2] < 0.2)) * box_condition
                        )[0]
        segmented_points.append(condition)
        
        # Segment Link 2
        condition1 = torch.where(((points[:,0] < centers[1,0]) * (points[:,2] > centers[0,2]) * (points[:,2] < 0.3) * (points[:,1] < 0.3))*box_condition)[0]
        condition2 = torch.where(((points[:,0] < centers[2,0]) * (points[:, 2] >= 0.3) * (points[:, 1] < 0.1))*box_condition)[0]
        condition = torch.cat([condition1, condition2])
        segmented_points.append(condition)
        
        # Segment Link 3
        condition1 = torch.where(((points[:,0] > centers[2,0]) * (points[:,1] > (centers[2,1] - 0.1)) * (points[:,1] < 0.3) * (points[:,2] > 0.4))*box_condition)[0]
        condition2 = torch.where(((points[:, 0] > centers[3, 0]) * (points[:, 1] >= 0.3) * (points[:, 2] > 0.4))*box_condition)[0]
        condition = torch.cat([condition1, condition2])
        
        segmented_points.append(condition)
        
        # Segment Link 4
        condition = torch.where(((points[:, 0] < centers[3, 0]) * (points[:, 1] > 0.25) * (points[:,1] < centers[4, 1]) * (points[:,2] > 0.3))*box_condition)[0]

        segmented_points.append(condition)
        
        # Segment Link 5
        condition = torch.where(((points[:, 0] < centers[3, 0]) * (points[:,1] > centers[4, 1]) * (points[:, 2] > centers[5, 2]))*box_condition)[0]
        segmented_points.append(condition)

        # Segment Link 6
        # condition = torch.where(((points[:, 0] < centers[3, 0]) * (points[:,1] > centers[4, 1]) * (points[:, 2] < centers[5, 2]))*box_condition)[0]
        condition = torch.where(((points[:, 0] < centers[3, 0]+0.2) * (points[:,1] > centers[4, 1]) * (points[:, 2] < centers[5, 2]) * (points[:, 2] > 0.4))*box_condition)[0]
        segmented_points.append(condition)


        #undo the temporary transformation
        points = torch.matmul(torch.inverse(temp_matrix)[:3, :3], points.T).T + torch.inverse(temp_matrix)[:3, 3]

        #load labels.npy
        labels = np.load('/mnt/data-3/users/mengxinpan/SplatSim/labels_iphone.npy')
        labels = torch.from_numpy(labels).to(device=xyz.device).long()

        # condition = (points[:, 2] > 0.2) & (points[:, 2] < 0.5) & (points[:, 1] < 0.2) & (points[:, 1] > 0.) & (points[:, 0] < 0.6) & (points[:, 0] > -0.)

        condition = (points[:, 2] > 0.2) & (points[:, 2] < 0.4) & (points[:, 1] < 0.2) & (points[:, 1] > 0.) & (points[:, 0] < 0.6) & (points[:, 0] > -0.)
        condition = torch.where(condition)[0]

        segmented_points.append(condition[labels== 1])
        segmented_points.append(condition[labels== 2])
        segmented_points.append(condition[labels== 3])
        segmented_points.append(condition[labels== 4])
        segmented_points.append(condition[labels== 5])
        segmented_points.append(condition[labels== 6])
        segmented_points.append(condition[labels== 7])
        segmented_points.append(condition[labels== 8])
        segmented_points.append(condition[labels== 9])
        segmented_points.append(condition[labels== 10])
        segmented_points.append(condition[labels== 11])


        
        return segmented_points, points

    def transform_shs(self, shs_feat, rotation_matrix):
        P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
        permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix.cpu().numpy() @ P
        rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix).to(device=shs_feat.device).float())
        
        # Construction coefficient
        D_1 = o3.wigner_D(1, rot_angles[0].cpu(), - rot_angles[1].cpu(), rot_angles[2].cpu()).to(device=shs_feat.device)
        D_2 = o3.wigner_D(2, rot_angles[0].cpu(), - rot_angles[1].cpu(), rot_angles[2].cpu()).to(device=shs_feat.device)
        D_3 = o3.wigner_D(3, rot_angles[0].cpu(), - rot_angles[1].cpu(), rot_angles[2].cpu()).to(device=shs_feat.device)

        #rotation of the shs features
        one_degree_shs = shs_feat[:, 0:3]
        one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        one_degree_shs = einsum(
                D_1,
                one_degree_shs,
                "... i j, ... j -> ... i",
            )
        one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 0:3] = one_degree_shs

        two_degree_shs = shs_feat[:, 3:8]
        two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        two_degree_shs = einsum(
                D_2,
                two_degree_shs,
                "... i j, ... j -> ... i",
            )
        two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 3:8] = two_degree_shs

        three_degree_shs = shs_feat[:, 8:15]
        three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        three_degree_shs = einsum(
                D_3,
                three_degree_shs,
                "... i j, ... j -> ... i",
            )
        three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 8:15] = three_degree_shs

        return shs_feat

    
    def run(self, initial_obs: Dict[str, np.ndarray]):
        obs_dict = initial_obs
        step = 0
        while True:
            print(f"Step {step}")
            action = self.act(obs_dict)
            # act函数正常来说action输出ee，但是这里临时改为关节角
            print('Action:', action)
            # Render images using Gaussian splatting
            wrist_rgb = self.render_image(action,step, view_idx=5)    # Wrist camera
            base_rgb = self.render_image(action,step, view_idx=254)  # Base camera

            # Save images
            # wrist_path = os.path.join(self.output_dir, f"step_{step:05d}_wrist.png")
            # base_path = os.path.join(self.output_dir, f"step_{step:05d}_base.png")
            # cv2.imwrite(wrist_path, cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(base_path, cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR))
            # print(f"Saved images: {wrist_path}, {base_path}")

            # Update observation
            joint_positions = action[:6]
            gripper_position = np.array([action[6]])
            
            obs_dict = {
                "wrist_rgb": wrist_rgb,
                "base_rgb": base_rgb,
                "joint_positions": joint_positions,    # TODO 原来是joint position，改为action
                "gripper_position": gripper_position
            }

            step += 1

if __name__ == "__main__":
    demo_agent = DiffusionAgent(
        port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0",
        output_dir="/mnt/data-3/users/mengxinpan/SplatSim/output_images",
        model_path="/shared_disk/datasets/public_datasets/SplatSim/output/robot_iphone",
        iteration=-1,
        traj_folder="/mnt/data-3/users/mengxinpan/SplatSim/bc_data_run_by_render/",
        objects="plastic_apple"
    )
    # base_path = "/mnt/data-3/users/mengxinpan/bc_data_il/gello/0"
    # wrist_rgb_path = os.path.join(base_path, "images_1", "00000.png")
    # base_rgb_path = os.path.join(base_path, "images_2", "00000.png")
    # pkl_path = os.path.join(base_path, "00001.pkl")

    # wrist_rgb = cv2.cvtColor(cv2.imread(wrist_rgb_path), cv2.COLOR_BGR2RGB)
    # base_rgb = cv2.cvtColor(cv2.imread(base_rgb_path), cv2.COLOR_BGR2RGB)
    # with open(pkl_path, 'rb') as f:
    #     pkl_data = pickle.load(f)
    #     joint_positions = pkl_data["joint_positions"]
    
    # initial_obs = {
    #     "wrist_rgb": np.array(wrist_rgb),
    #     "base_rgb": np.array(base_rgb),
    #     "joint_positions": np.array(joint_positions[:7]),  # TODO 原来是：6
    #     "gripper_position": np.array([joint_positions[6] if len(joint_positions) > 6 else 0])
    # }
    
    initial_joint_positions = np.array([-0.74960115, -2.13055178,  2.03415718, -1.47440167, -1.57079626,-0.74960115,  0.])
    initial_state = np.array([2.64249102e-01, -9.68567035e-02, 4.87049866e-01, -1.57079632e+00,8.60080139e-08, -1.57079633e+00, 2.93426317e-01])
    # 使用 render_image 生成初始图像
    wrist_rgb = demo_agent.render_image(initial_joint_positions, step=0, view_idx=5)   # 腕部相机
    base_rgb = demo_agent.render_image(initial_joint_positions, step=0, view_idx=254) # 基座相机

    # 构造 initial_obs
    initial_obs = {
        "wrist_rgb": wrist_rgb,                       # 腕部渲染图像 (H, W, 3)
        "base_rgb": base_rgb,                         # 基座渲染图像 (H, W, 3)
        "joint_positions": initial_joint_positions,   # 7 维关节位置
        "gripper_position": np.array([initial_joint_positions[6]])  # 夹爪位置 (1,)
    }
    
    demo_agent.run(initial_obs)