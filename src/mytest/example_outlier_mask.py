import torch
import math

# 创建示例数据
# cache.shape: (n_batch=1, seq_len=3, n_layer=2, n_head=2, embed_size_per_head=4)
cache = torch.tensor([[
    [
        [
            [1.0, 2.0, 3.0, 10.0],   # layer=0, head=0
            [4.0, 5.0, 6.0, -10.0]   # layer=0, head=1
        ],
        [
            [7.0, 8.0, 9.0, 20.0],   # layer=1, head=0
            [11.0, 12.0, 13.0, -20.0] # layer=1, head=1
        ]
    ],
    [
        [
            [14.0, 15.0, 16.0, 30.0], # layer=0, head=0
            [17.0, 18.0, 19.0, -30.0] # layer=0, head=1
        ],
        [
            [21.0, 22.0, 23.0, 40.0], # layer=1, head=0
            [24.0, 25.0, 26.0, -40.0] # layer=1, head=1
        ]
    ],
    [
        [
            [27.0, 28.0, 29.0, 50.0], # layer=0, head=0
            [31.0, 32.0, 33.0, -50.0] # layer=0, head=1
        ],
        [
            [34.0, 35.0, 36.0, 60.0], # layer=1, head=0
            [37.0, 38.0, 39.0, -60.0] # layer=1, head=1
        ]
    ]
]])

print("原始缓存张量:")
print(cache)
print(f"缓存张量形状: {cache.shape}")

# 设置参数
outliers_ratio = 0.25
quantize_dims = (-1,)  # 在最后一个维度上进行量化
device = torch.device('cpu')

print(f"\n异常值比例: {outliers_ratio}")
print(f"量化维度: {quantize_dims}")

# 步骤1: 展平量化维度
cache_flat = cache.flatten(start_dim=-len(quantize_dims))
print(f"\n展平后的缓存张量形状: {cache_flat.shape}")
print("展平后的缓存张量:")
print(cache_flat)

# 步骤2: 计算阈值
# 对于每个展平维度，我们有4个值，25%的异常值意味着每个尾部有1个值
# 下界阈值：位于下半部分异常值范围的边界 (第1个值)
lower_k = int(math.ceil(outliers_ratio/2 * cache_flat.shape[-1]))  # 1
# 上界阈值：位于上半部分异常值范围的边界 (第4个值)
upper_k = int(math.floor((1.0 - outliers_ratio/2) * cache_flat.shape[-1]))  # 3

print(f"\n计算阈值:")
print(f"lower_k (下界位置): {lower_k}")
print(f"upper_k (上界位置): {upper_k}")

# 计算下界和上界阈值
lower_threshold = torch.kthvalue(cache_flat, k=lower_k, dim=-1).values
upper_threshold = torch.kthvalue(cache_flat, k=upper_k, dim=-1).values

print(f"\n下界阈值形状: {lower_threshold.shape}")
print("下界阈值:")
print(lower_threshold)
print(f"\n上界阈值形状: {upper_threshold.shape}")
print("上界阈值:")
print(upper_threshold)

# 步骤3: 调整阈值张量形状
lower_threshold = lower_threshold.view(*lower_threshold.shape, *((1,)*len(quantize_dims)))
upper_threshold = upper_threshold.view(*upper_threshold.shape, *((1,)*len(quantize_dims)))

print(f"\n调整后的下界阈值形状: {lower_threshold.shape}")
print(f"调整后的上界阈值形状: {upper_threshold.shape}")

# 步骤4: 创建异常值掩码
mask = (cache <= lower_threshold) | (cache > upper_threshold)

print(f"\n最终异常值掩码形状: {mask.shape}")
print("异常值掩码 (True表示异常值):")
print(mask)