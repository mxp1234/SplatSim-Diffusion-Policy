import torch
import dill
import hydra
ckpt_path = '/mnt/data-3/users/mengxinpan/code/diffusion_policy/data/outputs/2025.03.31/00.48.37_train_diffusion_unet_image_real_image/checkpoints/epoch=0100-train_loss=0.002.ckpt'
payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
cfg = payload['cfg']

# 检查 cfg 中是否包含 device 字段
print("原始 cfg:", cfg)  # 调试：查看 cfg 的内容

# 假设 cfg 中有 device 字段，修改它
# if hasattr(cfg, 'device'):
#     cfg.device = 'cuda:1'  # 修改为 cuda:1
# elif 'device' in cfg:
cfg['training']['device'] = 'cuda:1'  # 如果是字典形式
print('done')
# else:
#     # 如果 cfg 中没有 device 字段，可以手动添加
#     from omegaconf import OmegaConf
#     cfg = OmegaConf.create(cfg)  # 确保是 OmegaConf 对象
#     cfg.device = 'cuda:1'

# print("修改后的 cfg:", cfg)  # 确认修改结果