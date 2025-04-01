import copy
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils_fk import *
from e3nn import o3
from einops import einsum
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.transform.rotation import Rotation as R
import einops
import pickle
import numpy as np
import yaml
import pybullet as p

# 初始化 PyBullet
p.connect(p.DIRECT)
p.resetSimulation()
flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
pandaUid = p.loadURDF('pybullet-playground_2/urdf/sisbot.urdf', useFixedBase=True, basePosition=[0.0, 0.0, -0.1])

# 初始化关节状态
joint_poses = [0, 0.0, -1.5707963267948966, 1.5707963267948966, -1.5707963267948966, -1.5707963267948966, 0.0, 0.0, 0.0, 0.7999999999999996, 0.0, -0.8000070728762431, 0.0, 0.7999947291384548, 0.799996381456464, 0.0, -0.799988452159267, 0.0, 0.7999926186486127]
for i in range(len(joint_poses)):
    p.resetJointState(pandaUid, i, joint_poses[i])

initial_joints = []
for joint_index in range(19):
    joint_info = p.getJointInfo(pandaUid, joint_index)
    link_state = p.getLinkState(pandaUid, joint_index, computeForwardKinematics=True)
    initial_joints.append(link_state)

# 获取变换列表的函数（保持不变）
def get_transfomration_list(new_joint_poses):
    for i in range(len(joint_poses)):
        p.resetJointState(pandaUid, i, new_joint_poses[i])
    new_joints = []
    for joint_index in range(19):
        link_state = p.getLinkState(pandaUid, joint_index, computeForwardKinematics=True)
        new_joints.append(link_state)
    transformations_list = []
    for joint_index in range(19):
        # input格式（x，y，z，quad）
        input_1 = (initial_joints[joint_index][0][0], initial_joints[joint_index][0][1], initial_joints[joint_index][0][2], np.array(initial_joints[joint_index][1]))
        input_2 = (new_joints[joint_index][0][0], new_joints[joint_index][0][1], new_joints[joint_index][0][2], np.array(new_joints[joint_index][1]))
        r_rel, t = compute_transformation(input_1, input_2)
        r_rel = torch.from_numpy(r_rel).to(device='cuda').float()
        t = torch.from_numpy(t).to(device='cuda').float()
        transformations_list.append((r_rel, t))
    return transformations_list

# 原始渲染函数（保持不变）
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, object_list, robot_name, object_splat_folder):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    gaussians_backup = copy.deepcopy(gaussians)

    object_gaussians = [GaussianModel(3) for _ in range(len(object_list))]
    for i in range(len(object_list)):
        object_gaussians[i].load_ply(os.path.join(object_splat_folder, "{}/point_cloud/iteration_7000/point_cloud.ply".format(object_list[i])))
    object_gaussians_backup = copy.deepcopy(object_gaussians)

    with open('SplatSim/object_configs/objects.yaml', 'r') as file:
        object_config = yaml.safe_load(file)
        robot_transformation = object_config[robot_name]['transformation']['matrix']

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 255:
            break
        if idx not in [5, 254]:
            continue
        traj_folder = args.traj_folder
        for j, folder in enumerate(sorted(os.listdir(traj_folder))):
            if j > 3:
                break
            if idx == 254:
                image_folder = os.path.join(traj_folder, folder, 'images_2')
            if idx == 5:
                image_folder = os.path.join(traj_folder, folder, 'images_1')
            makedirs(image_folder, exist_ok=True)
            file_names = sorted([f for f in os.listdir(os.path.join(traj_folder, folder)) if f.endswith('.pkl')])
            iterator = tqdm(range(len(file_names)), desc="Rendering progress")
            for k in iterator:
                file_path = os.path.join(traj_folder, folder, file_names[k])
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    cur_joint = [0] + data['joint_positions_dummy'].tolist()
                    cur_joint = np.array(cur_joint)
                    cur_object_position_list = []
                    cur_object_rotation_list = []
                    for object_name in object_list:
                        cur_object_position = torch.from_numpy(np.array(data[object_name + '_position'])).to(device='cuda').float()
                        cur_object_rotation = torch.from_numpy(np.roll(np.array(data[object_name + '_orientation']), 1)).to(device='cuda').float()
                        cur_object_position_list.append(cur_object_position)
                        cur_object_rotation_list.append(cur_object_rotation)

                    transformations_list = get_transfomration_list(cur_joint)
                    segmented_list, xyz = get_segmented_indices(gaussians_backup, robot_transformation)
                    xyz, rot, opacity, shs_featrest, shs_dc = transform_means(gaussians_backup, xyz, segmented_list, transformations_list, robot_transformation)

                    xyz_obj_list, rot_obj_list, opacity_obj_list, scales_obj_list, features_dc_obj_list, features_rest_obj_list = [], [], [], [], [], []
                    for i in range(len(object_list)):
                        xyz_obj, rot_obj, opacity_obj, scales_obj, features_dc_obj, features_rest_obj = transform_object(
                            object_gaussians_backup[i], object_config[object_list[i]], cur_object_position_list[i], cur_object_rotation_list[i], robot_transformation)
                        xyz_obj_list.append(xyz_obj)
                        rot_obj_list.append(rot_obj)
                        opacity_obj_list.append(opacity_obj)
                        scales_obj_list.append(scales_obj)
                        features_dc_obj_list.append(features_dc_obj)
                        features_rest_obj_list.append(features_rest_obj)

                    with torch.no_grad():
                        gaussians._xyz = torch.cat([xyz] + xyz_obj_list, dim=0)
                        gaussians._rotation = torch.cat([rot] + rot_obj_list, dim=0)
                        gaussians._opacity = torch.cat([opacity] + opacity_obj_list, dim=0)
                        gaussians._features_rest = torch.cat([shs_featrest] + features_rest_obj_list, dim=0)
                        gaussians._features_dc = torch.cat([shs_dc] + features_dc_obj_list, dim=0)
                        gaussians._scaling = torch.cat([gaussians_backup._scaling] + scales_obj_list, dim=0)
                        rendering = render(view, gaussians, pipeline, background)["render"]
                        torchvision.utils.save_image(rendering,"/mnt/data-3/users/mengxinpan/SplatSim/output_images")
                    object_gaussians = copy.deepcopy(object_gaussians_backup)
            print("Done with folder: ", folder)

# 新增函数：基于自定义 cur_joint 渲染并保存图像
def render_with_custom_joint(cur_joint, model_path, view, gaussians, pipeline, background, object_list, robot_name, object_splat_folder, save_path="custom_renders"):
    """
    根据给定的 cur_joint 渲染高斯图像并保存。
    参数:
        cur_joint: 关节角度列表（长度为19，例如 [0, j1, j2, ..., j18]）
        model_path: 模型路径
        view: 渲染视角
        gaussians: 高斯模型
        pipeline: 渲染管线参数
        background: 背景颜色
        object_list: 物体列表
        robot_name: 机器人名称
        object_splat_folder: 物体高斯文件路径
        save_path: 保存渲染图像的文件夹（相对于 model_path）
    """
    # 设置保存路径
    custom_render_path = os.path.join(model_path, save_path)
    makedirs(custom_render_path, exist_ok=True)

    # 备份高斯模型
    gaussians_backup = copy.deepcopy(gaussians)
    object_gaussians = [GaussianModel(3) for _ in range(len(object_list))]
    print(len(object_list))
    for i in range(len(object_list)):
        object_gaussians[i].load_ply(os.path.join(object_splat_folder, "{}/point_cloud/iteration_7000/point_cloud.ply".format(object_list[i])))
    object_gaussians_backup = copy.deepcopy(object_gaussians)

    # 加载机器人变换配置
    with open('object_configs/objects.yaml', 'r') as file:
        object_config = yaml.safe_load(file)
        robot_transformation = object_config[robot_name]['transformation']['matrix']

    # 确保 cur_joint 是 numpy 数组
    cur_joint = np.array(cur_joint) # 20维
    
    # 假设物体位置和旋转是固定的（如果需要动态输入，可以扩展参数）
    cur_object_position_list = [torch.tensor([0.6, 0.4, -0.02], device="cuda").float() for _ in object_list]
    cur_object_rotation_list = [torch.tensor([1.0, 0.0, 0.0, 0.0], device='cuda', dtype=torch.float32) for _ in object_list]  # 默认四元数（无旋转）
    transformations_list = get_transfomration_list(cur_joint)
    segmented_list, xyz = get_segmented_indices(gaussians_backup, robot_transformation)
    xyz, rot, opacity, shs_featrest, shs_dc = transform_means(gaussians_backup, xyz, segmented_list, transformations_list, robot_transformation)

    # 变换物体
    xyz_obj_list, rot_obj_list, opacity_obj_list, scales_obj_list, features_dc_obj_list, features_rest_obj_list = [], [], [], [], [], []
    for i in range(len(object_list)):
        xyz_obj, rot_obj, opacity_obj, scales_obj, features_dc_obj, features_rest_obj = transform_object(
            object_gaussians_backup[i], object_config[object_list[i]], cur_object_position_list[i], cur_object_rotation_list[i], robot_transformation)
        xyz_obj_list.append(xyz_obj)
        rot_obj_list.append(rot_obj)
        opacity_obj_list.append(opacity_obj)
        scales_obj_list.append(scales_obj)
        features_dc_obj_list.append(features_dc_obj)
        features_rest_obj_list.append(features_rest_obj)

    # 渲染
    with torch.no_grad():
        gaussians._xyz = torch.cat([xyz] + xyz_obj_list, dim=0)
        gaussians._rotation = torch.cat([rot] + rot_obj_list, dim=0)
        gaussians._opacity = torch.cat([opacity] + opacity_obj_list, dim=0)
        gaussians._features_rest = torch.cat([shs_featrest] + features_rest_obj_list, dim=0)
        gaussians._features_dc = torch.cat([shs_dc] + features_dc_obj_list, dim=0)
        gaussians._scaling = torch.cat([gaussians_backup._scaling] + scales_obj_list, dim=0)
        rendering = render(view, gaussians, pipeline, background)["render"]
        
        # 保存渲染图像（使用关节配置的哈希作为文件名，避免覆盖）
        joint_hash = str(hash(str(cur_joint.tolist())))
        save_file = os.path.join(custom_render_path, f"render_{joint_hash}.png")
        torchvision.utils.save_image(rendering, "/mnt/data-3/users/mengxinpan/SplatSim/output_images/9.png")
        print(f"Rendered image saved to: {save_file}")

# 修改后的 render_sets，支持自定义渲染
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, object_list: list, robot_name: str, object_splat_folder: str, custom_joint=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if custom_joint is not None:
            # 如果提供了自定义关节，则渲染单张图像
            render_with_custom_joint(
                cur_joint=custom_joint,
                model_path=dataset.model_path,
                view=scene.getTrainCameras()[0],  # 使用第一个训练视角，你可以改为其他视角
                gaussians=gaussians,
                pipeline=pipeline,
                background=background,
                object_list=object_list,
                robot_name=robot_name,
                object_splat_folder=object_splat_folder
            )
        else:
            # 否则执行原始的批量渲染
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, object_list, robot_name, object_splat_folder)
            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, object_list, robot_name, object_splat_folder)

# 其余函数保持不变：transform_means, transform_object, transform_shs, get_segmented_indices
def transform_means(pc, xyz, segmented_list, transformations_list, robot_transformation):

    # Trans = torch.tensor([[0.091979, 0.040193, -0.202176, 0.771204],
    #                         [0.205922, -0.007912, 0.092110, -0.335536],
    #                         [0.009315, -0.221975, -0.039892, 0.520633],
    #                         [0, 0, 0, 1]]).to(device=xyz.device) # shape


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
        shs_feat = transform_shs(shs_feat, rot_rotation_matrix)
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
        shs_feat = transform_shs(shs_feat, rot_rotation_matrix)
        # print('shs_feat : ', shs_feat.shape)
        with torch.no_grad():
            shs_featrest[segment] = shs_feat
        # shs_dc[segment] = shs_dc_segment
        # shs_featrest[segment] = torch.zeros_like(shs_featrest[segment])
        cnt += 1
           
    #transform_back
    xyz = torch.matmul(inv_rotation_matrix, xyz.T).T + inv_translation
    
        
    return xyz, rot, opacity, shs_featrest, shs_dc


def transform_object(pc, object_config, pos, quat, robot_transformation):
            
    
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
    features_rest_obj= transform_shs( features_rest_obj, rot_rotation_matrix)
    # features_rest_obj = torch.zeros_like(features_rest_obj)
    
    return xyz_obj, rotation_obj, opacity_obj, scales_obj, features_dc_obj, features_rest_obj


def transform_shs(shs_feat, rotation_matrix):

    ## rotate shs
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


def get_segmented_indices(pc, robot_transformation):
    # empty torch cache
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
    labels = np.load('labels_iphone.npy')
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

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--objects", default='plastic_apple', type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--traj_folder", default='/shared_disk/datasets/public_datasets/SplatSim/bc_data/gello', type=str)
    parser.add_argument("--custom_joint", default="0,0.10400249, -1.27832248, 1.39666304, -1.57140399, -1.56695457, 0.23855038, 0. , 0. , 0.24190453, 0. , -0.29585218, 0. , 0.25283451, 0.21198826, 0. , -0.27606276, 0. , 0.25462059, 0. ", type=str, help="Custom joint positions as a comma-separated list (e.g., '0,0,-1.57,1.57,-1.57,-1.57,0,...')")
    args = get_combined_args(parser)
    robot_name = args.model_path.split('/')[-1]
    object_splat_folder = args.model_path.replace(robot_name, '')
    if args.objects == " ":
        object_list = []
    else:
        object_list = args.objects.split(' ')

    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    # 如果提供了自定义关节，则转换为列表
    custom_joint = None
    if args.custom_joint:
        custom_joint = [float(x) for x in args.custom_joint.split(',')]
        if len(custom_joint) != 20:
            raise ValueError("custom_joint must contain exactly 19 values matching the robot's joint configuration.")

    render_sets(
        model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
        object_list=object_list, robot_name=robot_name, object_splat_folder=object_splat_folder, custom_joint=custom_joint
    )