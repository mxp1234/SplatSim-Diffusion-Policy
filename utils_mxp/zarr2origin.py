import zarr
import numpy as np
import os
import pandas as pd

# 指定数据路径
data_path = "/mnt/data-3/users/mengxinpan/mxp_exper/pusht_real/real_pusht_20230105/replay_buffer.zarr"
output_dir = "./restored_real_pusht_data"  # 输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 打开zarr文件
store = zarr.open(data_path, mode='r')

# 检查文件结构
print("Zarr文件结构:")
print(store.tree())

# 获取data和meta group
data_group = store['data']
meta_group = store['meta']

# 获取episode分割点
episode_ends = meta_group['episode_ends'][:]
n_episodes = len(episode_ends)
print(f"共 {n_episodes} 个episodes")

# 检查data group中的数组
print("\nData group 包含的数组:")
for array_name in data_group.array_keys():
    print(f"{array_name}: {data_group[array_name].shape}, {data_group[array_name].dtype}")

# 遍历每个episode并还原数据
for i in range(n_episodes):
    # 计算当前episode的起止索引
    start_idx = 0 if i == 0 else episode_ends[i-1]
    end_idx = episode_ends[i]
    
    # 创建episode子目录
    episode_dir = os.path.join(output_dir, f"episode_{i}")
    os.makedirs(episode_dir, exist_ok=True)
    
    # 1. 还原所有数据并保存为numpy数组
    action_data = data_group['action'][start_idx:end_idx]  # (n_frames, 6)
    robot_eef_pose_data = data_group['robot_eef_pose'][start_idx:end_idx]
    robot_eef_pose_vel_data = data_group['robot_eef_pose_vel'][start_idx:end_idx]
    robot_joint_data = data_group['robot_joint'][start_idx:end_idx]
    robot_joint_vel_data = data_group['robot_joint_vel'][start_idx:end_idx]
    stage_data = data_group['stage'][start_idx:end_idx]
    timestamp_data = data_group['timestamp'][start_idx:end_idx]
    
    np.save(os.path.join(episode_dir, "action.npy"), action_data)
    np.save(os.path.join(episode_dir, "robot_eef_pose.npy"), robot_eef_pose_data)
    np.save(os.path.join(episode_dir, "robot_eef_pose_vel.npy"), robot_eef_pose_vel_data)
    np.save(os.path.join(episode_dir, "robot_joint.npy"), robot_joint_data)
    np.save(os.path.join(episode_dir, "robot_joint_vel.npy"), robot_joint_vel_data)
    np.save(os.path.join(episode_dir, "stage.npy"), stage_data)
    np.save(os.path.join(episode_dir, "timestamp.npy"), timestamp_data)
    
    # 2. 保存为CSV文件（便于查看）
    action_df = pd.DataFrame(action_data, columns=[f'action_{k}' for k in range(6)])
    robot_eef_pose_df = pd.DataFrame(robot_eef_pose_data, columns=[f'eef_pose_{k}' for k in range(6)])
    robot_eef_pose_vel_df = pd.DataFrame(robot_eef_pose_vel_data, columns=[f'eef_pose_vel_{k}' for k in range(6)])
    robot_joint_df = pd.DataFrame(robot_joint_data, columns=[f'joint_{k}' for k in range(6)])
    robot_joint_vel_df = pd.DataFrame(robot_joint_vel_data, columns=[f'joint_vel_{k}' for k in range(6)])
    stage_df = pd.DataFrame(stage_data, columns=['stage'])
    timestamp_df = pd.DataFrame(timestamp_data, columns=['timestamp'])
    
    action_df.to_csv(os.path.join(episode_dir, "action.csv"), index=False)
    robot_eef_pose_df.to_csv(os.path.join(episode_dir, "robot_eef_pose.csv"), index=False)
    robot_eef_pose_vel_df.to_csv(os.path.join(episode_dir, "robot_eef_pose_vel.csv"), index=False)
    robot_joint_df.to_csv(os.path.join(episode_dir, "robot_joint.csv"), index=False)
    robot_joint_vel_df.to_csv(os.path.join(episode_dir, "robot_joint_vel.csv"), index=False)
    stage_df.to_csv(os.path.join(episode_dir, "stage.csv"), index=False)
    timestamp_df.to_csv(os.path.join(episode_dir, "timestamp.csv"), index=False)
    
    print(f"Episode {i} 已处理，包含 {end_idx - start_idx} 帧")

print(f"\n数据已还原并保存至: {output_dir}")