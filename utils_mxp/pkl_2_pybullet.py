import pickle
import os
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from pathlib import Path

# 全局参数
DATA_DIR = "/mnt/data-3/users/mengxinpan/code/diffusion_policy/data/real/bc_data_il/gello/6"
OUTPUT_IMAGE_DIR = "/mnt/data-3/users/mengxinpan/SplatSim/output_pybullet_images"
OUTPUT_VIDEO_PATH = "/mnt/data-3/users/mengxinpan/SplatSim/output_pybullet_video.mp4"
FPS = 30  # 视频帧率

def load_joint_positions(data_dir):
    """从指定路径加载所有 .pkl 文件中的 joint_positions"""
    data_dir = Path(data_dir)
    joint_positions_list = []

    # 获取所有 .pkl 文件并按文件名排序
    pkl_files = sorted([f for f in data_dir.glob("*.pkl") if f.is_file()])
    
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            joint_positions = data["joint_positions"]  # 假设每个 .pkl 文件包含 joint_positions
            joint_positions_list.append(joint_positions)
            print(f"Loaded {pkl_file.name}: {joint_positions}")
    
    return joint_positions_list

def setup_pybullet():
    """初始化 PyBullet 环境并加载机器人模型（无可视化）"""
    # 连接 PyBullet（DIRECT 模式，不显示 GUI）
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 添加 PyBullet 数据路径

    # 加载地面和机器人模型
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("/mnt/data-3/users/mengxinpan/SplatSim/pybullet-playground_2/urdf/sisbot.urdf", 
                         basePosition=[0, 0, -0.1], 
                         baseOrientation=[0, 0, 0, 1], 
                         useFixedBase=True)
    
    # 设置重力
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240)
    
    return robot_id

def save_pybullet_images(robot_id, joint_positions_list, output_dir):
    """根据 joint_positions 在 PyBullet 中渲染并保存图像（无可视化）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = []
    
    for step, joint_positions in enumerate(joint_positions_list):
        # 设置关节位置（假设 joint_positions 是 7 维，前 6 个是关节，第 7 个是夹爪）
        for i in range(min(6, len(joint_positions))):  # 只设置前 6 个关节
            p.resetJointState(robot_id, i + 1, joint_positions[i])  # PyBullet 关节索引从 1 开始
        
        # 模拟一步，确保物理更新
        p.stepSimulation()
        
        # 设置相机参数
        width, height = 640, 480
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0.5, 0.5],      # 相机位置
            cameraTargetPosition=[0, 0, 0],         # 目标位置（机器人基座附近）
            cameraUpVector=[0, 0, 1]                # 上方向
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(width)/height, nearVal=0.1, farVal=100.0
        )
        
        # 渲染图像（使用软件渲染器）
        _, _, rgb, _, _ = p.getCameraImage(
            width, height, view_matrix, projection_matrix, renderer=p.ER_TINY_RENDERER
        )
        rgb_array = np.reshape(rgb, (height, width, 4))[:, :, :3]  # 提取 RGB，丢弃 alpha
        
        # 保存图像
        image_file = output_dir / f"step_{step:05d}.png"
        cv2.imwrite(str(image_file), cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
        image_files.append(image_file)
        print(f"Saved image: {image_file}")
    
    return image_files

def images_to_video(image_files, output_video_path, fps):
    """将图像序列转换为视频"""
    if not image_files:
        raise ValueError("没有图像文件可转换为视频")
    
    # 读取第一张图像以确定尺寸
    first_frame = cv2.imread(str(image_files[0]))
    height, width, _ = first_frame.shape
    
    # 初始化视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 写入视频
    for img_file in image_files:
        frame = cv2.imread(str(img_file))
        video_writer.write(frame)
        print(f"Added frame to video: {img_file}")
    
    # 释放视频编写器
    video_writer.release()
    print(f"Video saved to {output_video_path}")

def main():
    # 加载所有 joint_positions
    joint_positions_list = load_joint_positions(DATA_DIR)
    
    # 设置 PyBullet 环境
    robot_id = setup_pybullet()
    
    # 渲染并保存图像
    image_files = save_pybullet_images(robot_id, joint_positions_list, OUTPUT_IMAGE_DIR)
    
    # 将图像转换为视频
    images_to_video(image_files, OUTPUT_VIDEO_PATH, FPS)
    
    # 断开 PyBullet 连接
    p.disconnect()

if __name__ == "__main__":
    main()