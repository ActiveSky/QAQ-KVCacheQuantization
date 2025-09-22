import torch

def simple_normalize_demo():
    """简化版QAQ归一化过程演示"""
    print("=== QAQ归一化过程核心概念演示 ===\n")

    # 创建简单的示例数据
    torch.manual_seed(42)
    cache = torch.tensor([[[[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]],
                           [[7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0]]]])  # 形状: (1, 2, 2, 3)

    print(f"原始缓存:\n{cache}")
    print(f"形状: {cache.shape}\n")

    # 设置参数
    n_bits = 4  # 4比特量化
    quantize_dims = (-1,)  # 在最后一个维度上量化

    # 创建简单的异常值掩码（将量化维度上的最大值视为异常值）
    max_values = cache.amax(dim=quantize_dims, keepdim=True)
    outlier_mask = (cache == max_values)
    print(f"异常值掩码:\n{outlier_mask}\n")

    print("=== 1. 对称+最小-最大归一化 ===")
    # 对称量化：均值为0
    mean_value = torch.tensor(0.0)
    print(f"均值: {mean_value}")

    # 排除异常值后计算最大绝对值
    masked_cache = cache.clone()
    masked_cache[outlier_mask] = 0  # 简化处理异常值

    # 在量化维度上计算最大绝对值
    max_abs_value = masked_cache.abs().amax(dim=quantize_dims, keepdim=True)
    print(f"最大绝对值: {max_abs_value}")

    # 计算缩放因子
    scale_value = 2 * max_abs_value / (2 ** n_bits)
    print(f"缩放因子: {scale_value}")

    # 归一化
    normalized = (cache - mean_value) / scale_value
    print(f"归一化结果:\n{normalized}\n")

    # 量化过程（均匀量化）
    print("--- 量化过程 ---")
    quantized = torch.clamp(torch.round(normalized).to(torch.int32), -(2 ** (n_bits-1)), 2 ** (n_bits-1) - 1)
    print(f"量化结果:\n{quantized}\n")

    # 反量化过程
    print("--- 反量化过程 ---")
    dequantized = quantized.to(cache.dtype)
    print(f"反量化结果:\n{dequantized}\n")

    # 反归一化过程
    print("--- 反归一化过程 ---")
    denormalized = dequantized * scale_value + mean_value
    print(f"反归一化结果:\n{denormalized}\n")

    # 最终结果（保留异常值）
    print("--- 最终量化结果（保留异常值） ---")
    final_result = torch.where(outlier_mask, cache, denormalized)
    print(f"最终结果:\n{final_result}\n")

    print("=== 2. 非对称+最小-最大归一化 ===")
    # 非对称量化：计算实际均值
    mean_value = masked_cache.mean(dim=quantize_dims, keepdim=True)
    print(f"均值: {mean_value}")

    # 在量化维度上计算最大值和最小值
    max_value = masked_cache.amax(dim=quantize_dims, keepdim=True)
    min_value = masked_cache.amin(dim=quantize_dims, keepdim=True)
    print(f"最大值: {max_value}")
    print(f"最小值: {min_value}")

    # 计算缩放因子
    scale_value = (max_value - min_value) / (2 ** n_bits)
    print(f"缩放因子: {scale_value}")

    # 归一化
    normalized = (cache - mean_value) / scale_value
    print(f"归一化结果:\n{normalized}\n")

    # 量化过程
    print("--- 量化过程 ---")
    quantized = torch.clamp(torch.round(normalized).to(torch.int32), -(2 ** (n_bits-1)), 2 ** (n_bits-1) - 1)
    print(f"量化结果:\n{quantized}\n")

    # 反量化过程
    print("--- 反量化过程 ---")
    dequantized = quantized.to(cache.dtype)
    print(f"反量化结果:\n{dequantized}\n")

    # 反归一化过程
    print("--- 反归一化过程 ---")
    denormalized = dequantized * scale_value + mean_value
    print(f"反归一化结果:\n{denormalized}\n")

    # 最终结果（保留异常值）
    print("--- 最终量化结果（保留异常值） ---")
    final_result = torch.where(outlier_mask, cache, denormalized)
    print(f"最终结果:\n{final_result}\n")

    print("=== 3. 对称+标准差归一化 ===")
    # 对称量化：均值为0
    mean_value = torch.tensor(0.0)
    print(f"均值: {mean_value}")

    # 计算标准差
    std_value = masked_cache.std(dim=quantize_dims, keepdim=True)
    print(f"标准差: {std_value}")

    # 归一化
    normalized = (cache - mean_value) / std_value
    print(f"归一化结果:\n{normalized}\n")

    # 量化过程
    print("--- 量化过程 ---")
    quantized = torch.clamp(torch.round(normalized).to(torch.int32), -(2 ** (n_bits-1)), 2 ** (n_bits-1) - 1)
    print(f"量化结果:\n{quantized}\n")

    # 反量化过程
    print("--- 反量化过程 ---")
    dequantized = quantized.to(cache.dtype)
    print(f"反量化结果:\n{dequantized}\n")

    # 反归一化过程
    print("--- 反归一化过程 ---")
    denormalized = dequantized * std_value + mean_value
    print(f"反归一化结果:\n{denormalized}\n")

    # 最终结果（保留异常值）
    print("--- 最终量化结果（保留异常值） ---")
    final_result = torch.where(outlier_mask, cache, denormalized)
    print(f"最终结果:\n{final_result}\n")

    print("=== 4. 非对称+标准差归一化 ===")
    # 非对称量化：计算实际均值
    mean_value = masked_cache.mean(dim=quantize_dims, keepdim=True)
    print(f"均值: {mean_value}")

    # 计算标准差（高精度）
    std_value = masked_cache.to(torch.float64).std(dim=quantize_dims, keepdim=True).to(cache.dtype)
    print(f"标准差: {std_value}")

    # 归一化
    normalized = (cache - mean_value) / std_value
    print(f"归一化结果:\n{normalized}\n")

    # 量化过程
    print("--- 量化过程 ---")
    quantized = torch.clamp(torch.round(normalized).to(torch.int32), -(2 ** (n_bits-1)), 2 ** (n_bits-1) - 1)
    print(f"量化结果:\n{quantized}\n")

    # 反量化过程
    print("--- 反量化过程 ---")
    dequantized = quantized.to(cache.dtype)
    print(f"反量化结果:\n{dequantized}\n")

    # 反归一化过程
    print("--- 反归一化过程 ---")
    denormalized = dequantized * std_value + mean_value
    print(f"反归一化结果:\n{denormalized}\n")

    # 最终结果（保留异常值）
    print("--- 最终量化结果（保留异常值） ---")
    final_result = torch.where(outlier_mask, cache, denormalized)
    print(f"最终结果:\n{final_result}\n")

    print("=== 5. 正态分布量化 ===")
    # 正态分布量化：使用预计算的分位数表进行量化
    print(f"均值: {mean_value}")
    print(f"标准差: {std_value}")

    # 归一化（标准正态分布）
    normalized = (cache - mean_value) / std_value
    print(f"归一化结果:\n{normalized}\n")

    # 创建简单的分位数表（模拟Quantizer中的预计算分位数）
    # 对于4比特量化，我们有2^4=16个区间
    # 使用PyTorch内置函数近似实现正态分布的分位数计算
    # 这里我们手动定义分位数表来避免依赖scipy

    # 定义上界分位数（用于量化）
    upper_bound_quantiles = torch.tensor([
        -2.0537, -1.6449, -1.3829, -1.1904, -1.0364, -0.9074, -0.7937, -0.6903,
        -0.5930, -0.5000, -0.4097, -0.3215, -0.2346, -0.1484, -0.0627,  0.0230
    ], dtype=cache.dtype)
    print(f"上界分位数:\n{upper_bound_quantiles}\n")

    # 定义中心分位数（用于反量化）
    center_quantiles = torch.tensor([
        -1.8508, -1.5139, -1.2866, -1.1134, -0.9619, -0.8306, -0.7171, -0.6132,
        -0.5165, -0.4234, -0.3332, -0.2451, -0.1589, -0.0733,  0.0118,  0.0973
    ], dtype=cache.dtype)
    print(f"中心分位数:\n{center_quantiles}\n")

    # 量化过程：使用searchsorted找到每个值在分位数表中的位置
    print("--- 量化过程 ---")
    # 将normalized值限制在[-3, 3]范围内（标准正态分布的99.7%区间）
    normalized_clipped = torch.clamp(normalized, -3, 3)
    # 映射到[-2.0537, 0.0973]区间（分位数表的范围）
    normalized_mapped = (normalized_clipped + 3) * (0.0973 - (-2.0537)) / 6 + (-2.0537)
    # 使用searchsorted找到每个值在分位数表中的位置
    quantized = torch.searchsorted(upper_bound_quantiles, normalized_mapped.contiguous())
    # 确保索引在有效范围内
    quantized = torch.clamp(quantized, 0, 15)
    print(f"量化结果:\n{quantized}\n")

    # 反量化过程：使用中心分位数值
    print("--- 反量化过程 ---")
    dequantized = center_quantiles[quantized]
    print(f"反量化结果:\n{dequantized}\n")

    # 反归一化过程
    print("--- 反归一化过程 ---")
    # 将分位数值映射回[-2.0537, 0.0973]区间
    denormalized_mapped = (dequantized - (-2.0537)) * 6 / (0.0973 - (-2.0537)) - 3
    # 反归一化
    denormalized = denormalized_mapped * std_value + mean_value
    print(f"反归一化结果:\n{denormalized}\n")

    # 最终结果（保留异常值）
    print("--- 最终量化结果（保留异常值） ---")
    final_result = torch.where(outlier_mask, cache, denormalized)
    print(f"最终结果:\n{final_result}\n")

if __name__ == "__main__":
    simple_normalize_demo()