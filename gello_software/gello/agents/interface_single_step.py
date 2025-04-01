import os
import numpy as np
import torch
import dill
import hydra
import cv2
import click
import sys
diffusion_policy_path = "/mnt/data-3/users/mengxinpan/diffusion_policy/"
sys.path.append(diffusion_policy_path)
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

# /mnt/data-3/users/mengxinpan/code/diffusion_policy/data/outputs/2025.03.27/12.44.11_train_diffusion_unet_image_real_image/checkpoints/epoch=0350-train_loss=0.004.ckpt
OmegaConf.register_new_resolver("eval", eval, replace=True)
'''0.82开头'''
@click.command()
@click.option('--input', '-i', required=True, default='/mnt/data-3/users/mengxinpan/code/diffusion_policy/data/outputs/2025.03.26/05.29.02_train_diffusion_unet_image_real_image/checkpoints/epoch=0150-train_loss=0.001.ckpt', help='Path to checkpoint')
@click.option('--wrist_rgb_path', '-wr', required=True, default="/mnt/data-3/users/mengxinpan/code/diffusion_policy/data/real/bc_data_il/gello/3/images_1/00000.png", help='Path to wrist RGB image')
@click.option('--base_rgb_path', '-br', required=True, default="/mnt/data-3/users/mengxinpan/code/diffusion_policy/data/real/bc_data_il/gello/3/images_2/00000.png", help='Path to base RGB image')
@click.option('--joint_positions', '-jp', type=str, default='-0.74960115, -2.13055178, 2.03415718, -1.47440167, -1.57079626, -0.74960115, 0.', help='Comma-separated joint positions (7 values)')
def main(input, wrist_rgb_path, base_rgb_path, joint_positions):
    # Load checkpoint
    ckpt_path = input
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Policy setup
    if 'diffusion' in cfg.name:
        policy: BaseImagePolicy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)
        policy.num_inference_steps = 16
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    else:
        raise ValueError(f"Unsupported policy type: {cfg.name}. Only 'diffusion' is supported.")

    # Reset policy once at initialization (matching first code)
    # policy.reset()

    # Load images
    wrist_rgb = cv2.cvtColor(cv2.imread(wrist_rgb_path), cv2.COLOR_BGR2RGB)
    if wrist_rgb is None:
        raise FileNotFoundError(f"Failed to load wrist RGB image: {wrist_rgb_path}")
    base_rgb = cv2.cvtColor(cv2.imread(base_rgb_path), cv2.COLOR_BGR2RGB)
    if base_rgb is None:
        raise FileNotFoundError(f"Failed to load base RGB image: {base_rgb_path}")

    # Parse joint positions
    try:
        joint_positions = np.array([float(x) for x in joint_positions.split(',')])
        if len(joint_positions) != 7:
            raise ValueError("Joint positions must contain exactly 7 values.")
    except ValueError as e:
        raise ValueError(f"Invalid joint positions format: {e}")

    # Prepare observation (fixed resolution and 2 steps to match first code)
    # Resize images to (320, 240) as in first code
    image = cv2.resize(wrist_rgb, (320, 240)).transpose(2, 0, 1)  # (C, H, W)
    image = np.expand_dims(image, axis=(0, 1))  # (1, 1, C, H, W)
    image_tensor = torch.from_numpy(image).float().to(device) / 255.0

    image_2 = cv2.resize(base_rgb, (320, 240)).transpose(2, 0, 1)  # (C, H, W)
    image_2 = np.expand_dims(image_2, axis=(0, 1))  # (1, 1, C, H, W)
    image_2_tensor = torch.from_numpy(image_2).float().to(device) / 255.0

    # Prepare joint positions (fixed to 7 dimensions)
    joint_positions = joint_positions.reshape(1, 1, 7)
    joint_tensor = torch.from_numpy(joint_positions).float().to(device)

    # Simulate last observation (mimicking first code's behavior)
    last_image_obs = image_tensor.clone()  # Use current image as "last" for first call
    last_image_obs_1 = image_2_tensor.clone()
    last_state_obs = joint_tensor.clone()

    # Concatenate to form 2-step observation
    image_out = torch.cat((last_image_obs, image_tensor), dim=1)  # (1, 2, C, H, W)
    image_2_out = torch.cat((last_image_obs_1, image_2_tensor), dim=1)  # (1, 2, C, H, W)
    state_out = torch.cat((last_state_obs, joint_tensor), dim=1)  # (1, 2, 7)

    # Construct observation dictionary
    obs_dict = {
        'image_1': image_out,
        'image_2': image_2_out,
        'joint_positions': state_out
    }

    # Perform single-step inference (no warm-up, matching first code)
    print("Performing single-step inference...")
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
        action_pred = result['action'][0].detach().to('cpu').numpy()
        action = action_pred[:]  # Take only the first step, matching first code

    print(f"Predicted action: {action}")
    return action

if __name__ == '__main__':
    main()