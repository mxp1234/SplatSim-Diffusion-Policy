import cv2
import os
from pathlib import Path

def images_to_video(image_dir, output_video_path, fps=30):
    """
    将指定文件夹中的图像序列转换为视频。
    
    参数：
        image_dir (str): 包含图像的文件夹路径
        output_video_path (str): 输出视频的保存路径（例如 'output.mp4'）
        fps (int): 视频帧率（默认 30）
    """
    # 检查文件夹是否存在
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"文件夹 {image_dir} 不存在")

    # 获取图像文件列表并按文件名排序
    image_files = sorted(image_dir.glob("step_*.png"), key=lambda x: int(x.stem.split('_')[1]))
    if not image_files:
        raise ValueError(f"文件夹 {image_dir} 中没有找到任何 step_*.png 文件")

    # 读取第一张图像以确定视频尺寸
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        raise ValueError(f"无法读取第一张图像 {image_files[0]}，可能是文件损坏")
    
    height, width, _ = first_frame.shape
    print(f"Video dimensions: {width}x{height}")

    # 初始化视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐帧写入视频
    for img_file in image_files:
        frame = cv2.imread(str(img_file))
        if frame is None:
            print(f"警告：无法读取图像 {img_file}，跳过")
            continue
        
        video_writer.write(frame)
        print(f"已添加帧：{img_file}")

    # 释放视频编写器
    video_writer.release()
    print(f"视频已保存至 {output_video_path}")

if __name__ == "__main__":
    # 配置参数
    image_dir = "/mnt/data-3/users/mengxinpan/SplatSim/output_pybullet_act_images"  # 图像保存路径
    output_video_path = "/mnt/data-3/users/mengxinpan/SplatSim/output_pybullet_act_video.mp4"  # 输出视频路径
    fps = 30  # 每秒帧数，可调整

    # 执行转换
    images_to_video(image_dir, output_video_path, fps)