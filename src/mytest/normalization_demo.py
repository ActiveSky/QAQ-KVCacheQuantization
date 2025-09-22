import torch
import numpy as np
import math

def normalize_demo():
    """演示QAQ中的归一化过程"""
    print("=== QAQ归一化过程演示 ===\n")

    # 创建示例KV缓存数据
    torch.manual_seed(42)  # 固定随机种子以确保结果可重现
    cache = torch.randn(2, 3, 4, 5) * 10 + 5  # 创建一个带有偏移的正态分布数据
    print(f"原始缓存形状: {cache.shape}")
    print(f"原始缓存范围: [{cache.min().item():.2f}, {cache.max().item():.2f}]")
    print(f"原始缓存均值: {cache.mean().item():.2f}")
    print(f"原始缓存标准差: {cache.std().item():.2f}\n")

    # 创建异常值掩码（模拟10%的异常值）
    outlier_ratio = 0.1
    cache_flat = cache.flatten()
    lower_threshold = torch.kthvalue(cache_flat, k=int(math.ceil(outlier_ratio/2 * cache_flat.shape[0])), dim=0).values
    upper_threshold = torch.kthvalue(cache_flat, k=int(math.floor((1.0 - outlier_ratio/2) * cache_flat.shape[0])), dim=0).values
    outlier_mask = (cache <= lower_threshold) | (cache > upper_threshold)
    print(f"异常值比例: {outlier_mask.sum().item() / outlier_mask.numel():.2f}")
    print(f"异常值数量: {outlier_mask.sum().item()}\n")

    # 参数设置
    n_bits = 8  # 量化比特数
    quantize_dims = (-2, -1)  # 在最后两个维度上进行量化

    print("=== 对称量化 (symmetric=True) ===")
    symmetric = True

    # 对称量化的归一化处理
    # 手动处理异常值（将异常值设置为0）
    masked_cache = cache.clone()
    masked_cache[outlier_mask] = 0

    # 均值为0（对称量化假设）
    mean_value = torch.zeros_like(masked_cache)
    print(f"均值: 0.00 (对称量化固定为0)")

    # 最小-最大归一化
    # 为了避免异常值影响，我们只在非异常值上计算最大值
    valid_mask = ~outlier_mask
    max_value = torch.zeros_like(cache)
    # 在指定维度上计算最大值（手动实现）
    for i in range(cache.shape[0]):
        for j in range(cache.shape[1]):
            for k in range(cache.shape[2]):
                valid_values = masked_cache[i, j, k, valid_mask[i, j, k, :]]
                if valid_values.numel() > 0:
                    max_value[i, j, k, :] = valid_values.abs().max().item()

    scale_value = 2 * max_value / (2 ** n_bits)
    print(f"缩放因子范围: [{scale_value.min().item():.2f}, {scale_value.max().item():.2f}]")

    # 执行归一化
    normalized_cache = (masked_cache - 0) / scale_value
    print(f"归一化后范围: [{normalized_cache.min().item():.2f}, {normalized_cache.max().item():.2f}]")
    print(f"归一化后均值: {normalized_cache.mean().item():.2f}")
    print(f"归一化后标准差: {normalized_cache.std().item():.2f}\n")

    print("=== 非对称量化 (symmetric=False) ===")
    symmetric = False

    # 非对称量化的归一化处理
    # 手动处理异常值
    masked_cache = cache.clone()
    masked_cache[outlier_mask] = 0

    # 计算实际均值（在非异常值上计算）
    mean_value = torch.zeros_like(cache)
    # 在指定维度上计算均值（手动实现）
    for i in range(cache.shape[0]):
        for j in range(cache.shape[1]):
            for k in range(cache.shape[2]):
                valid_values = masked_cache[i, j, k, valid_mask[i, j, k, :]]
                if valid_values.numel() > 0:
                    mean_value[i, j, k, :] = valid_values.mean().item()

    print(f"均值范围: [{mean_value.min().item():.2f}, {mean_value.max().item():.2f}]")

    # 最小-最大归一化
    max_value = torch.zeros_like(cache)
    min_value = torch.zeros_like(cache)
    # 在指定维度上计算最大值和最小值（手动实现）
    for i in range(cache.shape[0]):
        for j in range(cache.shape[1]):
            for k in range(cache.shape[2]):
                valid_values = masked_cache[i, j, k, valid_mask[i, j, k, :]]
                if valid_values.numel() > 0:
                    max_value[i, j, k, :] = valid_values.max().item()
                    min_value[i, j, k, :] = valid_values.min().item()

    scale_value = (max_value - min_value) / (2 ** n_bits)
    print(f"缩放因子范围: [{scale_value.min().item():.2f}, {scale_value.max().item():.2f}]")

    # 执行归一化
    normalized_cache = (masked_cache - mean_value) / scale_value
    print(f"归一化后范围: [{normalized_cache.min().item():.2f}, {normalized_cache.max().item():.2f}]")
    print(f"归一化后均值: {normalized_cache.mean().item():.2f}")
    print(f"归一化后标准差: {normalized_cache.std().item():.2f}\n")

    print("=== 标准差归一化方法对比 ===")

    # 对称+标准差归一化
    masked_cache = cache.clone()
    masked_cache[outlier_mask] = 0
    mean_value = torch.zeros_like(masked_cache)

    # 计算标准差
    scale_value = torch.zeros_like(cache)
    # 在指定维度上计算标准差（手动实现）
    for i in range(cache.shape[0]):
        for j in range(cache.shape[1]):
            for k in range(cache.shape[2]):
                valid_values = masked_cache[i, j, k, valid_mask[i, j, k, :]]
                if valid_values.numel() > 0:
                    scale_value[i, j, k, :] = valid_values.std().item()

    normalized_cache = (masked_cache - 0) / scale_value
    # 处理标准差为0的情况
    normalized_cache[torch.isnan(normalized_cache)] = 0
    print(f"对称+标准差归一化后范围: [{normalized_cache.min().item():.2f}, {normalized_cache.max().item():.2f}]")
    print(f"对称+标准差归一化后均值: {normalized_cache.mean().item():.2f}")
    print(f"对称+标准差归一化后标准差: {normalized_cache[valid_mask].std().item() if valid_mask.sum() > 0 else 0:.2f}\n")

    # 非对称+标准差归一化（高精度）
    masked_cache = cache.clone()
    masked_cache[outlier_mask] = 0

    # 计算均值
    mean_value = torch.zeros_like(cache)
    for i in range(cache.shape[0]):
        for j in range(cache.shape[1]):
            for k in range(cache.shape[2]):
                valid_values = masked_cache[i, j, k, valid_mask[i, j, k, :]]
                if valid_values.numel() > 0:
                    mean_value[i, j, k, :] = valid_values.mean().item()

    # 计算高精度标准差
    scale_value = torch.zeros_like(cache)
    for i in range(cache.shape[0]):
        for j in range(cache.shape[1]):
            for k in range(cache.shape[2]):
                valid_values = masked_cache[i, j, k, valid_mask[i, j, k, :]].to(torch.float64)
                if valid_values.numel() > 0:
                    scale_value[i, j, k, :] = valid_values.std().to(cache.dtype).item()

    normalized_cache = (masked_cache - mean_value) / scale_value
    # 处理标准差为0的情况
    normalized_cache[torch.isnan(normalized_cache)] = 0
    print(f"非对称+标准差归一化后范围: [{normalized_cache.min().item():.2f}, {normalized_cache.max().item():.2f}]")
    print(f"非对称+标准差归一化后均值: {normalized_cache.mean().item():.2f}")
    print(f"非对称+标准差归一化后标准差: {normalized_cache[valid_mask].std().item() if valid_mask.sum() > 0 else 0:.2f}")

if __name__ == "__main__":
    normalize_demo()