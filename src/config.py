# 导入PyTorch库，用于张量计算和设备管理
import torch
from torch import device
# 定义实验版本号，用于标识实验配置的版本
version = "2023/03/20-#02"

# 定义缓存文件路径，用于存储实验结果
cache_file = "cache/results.json"

# 定义HuggingFace缓存目录，None表示使用默认缓存位置
hf_cache_dir = None

# 定义我的设备配置 - 当前配置为单GPU设置
# 格式为 [(设备对象, 内存配置字典), ...]
# 内存配置字典的键为设备ID，值为内存大小
device_configs: list[tuple[device, dict[int | str, int | str]]] = [
    # CUDA设备0，分配16GB显存，CPU可用32GB内存
    (torch.device("cuda:0"), {0: "16GB", "cpu": "32GB"})
]

# # 8xV100 & Llama-2-7B - 多GPU配置示例（已注释）
# # 为8个V100 GPU配置设备，每个GPU分配32GB显存，CPU可用400GB内存
# device_configs = [
#     (torch.device("cuda:0"), {0: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:1"), {1: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:2"), {2: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:3"), {3: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:4"), {4: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:5"), {5: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:6"), {6: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:7"), {7: "32GB", "cpu": "400GB"}),
# ]

# 8xV100 & Llama-2-13B - 多GPU配置示例（已注释）
# # 配置用于Llama-2-13B模型的8个V100 GPU，每个GPU对分配10GB和30GB显存
# device_configs = [
#     (torch.device("cuda:0"), {0: "10GB", 1: "30GB", "cpu": "400GB"}),
#     (torch.device("cuda:2"), {2: "10GB", 3: "30GB", "cpu": "400GB"}),
#     (torch.device("cuda:4"), {4: "10GB", 5: "30GB", "cpu": "400GB"}),
#     (torch.device("cuda:6"), {6: "10GB", 7: "30GB", "cpu": "400GB"}),
# ]

# # 8xV100 & Llama-2-70B - 多GPU配置示例（已注释）
# # 为Llama-2-70B模型配置设备，单个GPU管理多个层
# device_configs = [
#     (torch.device("cuda:0"), {
#         0: "32GB",
#         1: "32GB",
#         2: "32GB",
#         3: "32GB",
#         4: "32GB",
#         "cpu": "400GB",
#     }),
# ]