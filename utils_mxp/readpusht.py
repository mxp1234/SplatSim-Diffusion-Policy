import zarr
import numpy as np

# 指定数据路径
data_path = "/mnt/data-3/users/mengxinpan/mxp_exper/pusht_real/real_pusht_20230105/replay_buffer.zarr"

# 打开zarr文件
store = zarr.open(data_path, mode='r')

# 查看zarr文件结构
print("Zarr文件结构:")
print(store.tree())

# 获取data group
data_group = store['data']
meta_group = store['meta']

# 处理data中的各个数组
print("\n处理 data group 中的数据:")
for array_name in data_group.array_keys():
    array_data = data_group[array_name]
    print(f"\n{array_name}:")
    print(f"数据类型: {array_data.dtype}")
    print(f"数据形状: {array_data.shape}")
    # 显示前几个样本作为示例
    sample_data = array_data[:5]
    print(f"前5个样本:\n{sample_data}")

# 处理meta中的数据
print("\n处理 meta group 中的数据:")
for array_name in meta_group.array_keys():
    array_data = meta_group[array_name]
    print(f"\n{array_name}:")
    print(f"数据类型: {array_data.dtype}")
    print(f"数据形状: {array_data.shape}")
    sample_data = array_data[:5]

