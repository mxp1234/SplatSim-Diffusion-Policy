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
sys.path.append(diffusion_policy_path)
import torch
import dill
import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import cv2

@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    joint_offsets: Sequence[float]
    joint_signs: Sequence[int]
    gripper_config: Tuple[int, int, int]

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
}

class DiffusionAgent(Agent):
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
    ):
        # Diffusion Policy Setup
        print('Loading policy')
        ckpt_path = '/mnt/data-3/users/mengxinpan/code/diffusion_policy/data/outputs/2025.03.26/05.29.02_train_diffusion_unet_image_real_image/checkpoints/epoch=0150-train_loss=0.001.ckpt'
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

        # Agent State
        self.last_image_obs = None
        self.last_image_obs_1 = None
        self.last_state_obs = None
        self.policy.reset()
        print('Policy setup complete')

    def act(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # Process images
        image = cv2.resize(obs_dict['wrist_rgb'], (320, 240)).transpose(2, 0, 1)
        image = np.expand_dims(image, axis=(0, 1))
        image_tensor = torch.from_numpy(image).float() / 255.0
        image = image_tensor 
        image_2 = cv2.resize(obs_dict['base_rgb'], (320, 240)).transpose(2, 0, 1)
        image_2 = np.expand_dims(image_2, axis=(0, 1))
        image_2_tensor = torch.from_numpy(image_2).float() / 255.0
        image_2 = image_2_tensor 

        if self.last_image_obs is None:
            self.last_image_obs = image
            self.last_image_obs_1 = image_2
            self.last_state_obs = torch.from_numpy(obs_dict['joint_positions'].reshape(1, 1, 7)).float()

        cur_state_obs = torch.from_numpy(obs_dict['joint_positions'].reshape(1, 1, 7)).float()
        image_out = torch.cat((self.last_image_obs, image), dim=1)
        image_out_1 = torch.cat((self.last_image_obs_1, image_2), dim=1)
        state_out = torch.cat((self.last_state_obs, cur_state_obs), dim=1)

        # Perform one-step inference
        print('Performing one-step prediction')
        obs_dict_1 = {
            'image_1': image_out,
            'image_2': image_out_1,
            'joint_positions': state_out
        }
        result = self.policy.predict_action(obs_dict_1)
        # print(result)
        action_pred = result['action'][0].detach().cpu().numpy()
        print('Prediction result:', action_pred)  # Print the first step of the predicted action sequence
        return action_pred[0]  # Return the first predicted action

if __name__ == "__main__":
    demo_agent = DiffusionAgent(
        port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0",
    )

    # Load images from specified paths
    wrist_rgb_path = "/mnt/data-3/users/mengxinpan/code/diffusion_policy/data/real/bc_data_il/gello/3/images_1/00000.png"  # Replace with actual path
    base_rgb_path = "/mnt/data-3/users/mengxinpan/code/diffusion_policy/data/real/bc_data_il/gello/3/images_2/00000.png"    # Replace with actual path
    wrist_rgb = cv2.cvtColor(cv2.imread(wrist_rgb_path), cv2.COLOR_BGR2RGB)
    base_rgb = cv2.cvtColor(cv2.imread(base_rgb_path), cv2.COLOR_BGR2RGB)

    # Specify joint positions
    joint_positions = np.array([-0.74960115, -2.13055178,2.03415718, -1.47440167, -1.57079626, -0.74960115, 0.])

    # Construct initial observation
    initial_obs = {
        "wrist_rgb": np.array(wrist_rgb),
        "base_rgb": np.array(base_rgb),
        "joint_positions": joint_positions,
        "gripper_position": np.array([joint_positions[6]])
    }

    # Perform one-step inference and print result
    action = demo_agent.act(initial_obs)