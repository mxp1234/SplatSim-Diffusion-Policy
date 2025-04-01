import cv2
import os

# 设置路径和输出视频参数
image_folder = '/mnt/data-3/users/mengxinpan/SplatSim/output_images'
output_video = '/mnt/data-3/users/mengxinpan/SplatSim/output_video.mp4'
fps = 20  # 每秒帧数，可以调整

# 获取图片列表
images = [f"step{i}_view5.png" for i in range(0, 99)]  # 从 step0 到 step58
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 写入每一帧
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video_writer.write(frame)

# 释放资源
video_writer.release()
print(f"视频已保存到 {output_video}")