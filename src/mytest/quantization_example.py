
import torch

# 创建示例数据
cache = torch.tensor([[[[1.2, 2.5, 3.8, 10.1],
                        [4.1, 5.9, 6.7, -8.2]]]], dtype=torch.float32)

print("原始缓存数据:")
print(cache)
print(f"数据类型: {cache.dtype}")
print(f"内存占用: {cache.element_size() * cache.nelement()} 字节")

# 模拟量化过程
# 假设我们使用2位量化 (4个级别: 0, 1, 2, 3)
scale = 5.0  # 量化缩放因子

# 量化：浮点 -> 整数
quantized_cache = (cache / scale).round().clamp(0, 3).to(torch.int8)
print("\n量化后的数据:")
print(quantized_cache)
print(f"数据类型: {quantized_cache.dtype}")
print(f"内存占用: {quantized_cache.element_size() * quantized_cache.nelement()} 字节")

# 反量化：整数 -> 浮点
dequantized_cache = quantized_cache.to(torch.float32) * scale
print("\n反量化后的数据:")
print(dequantized_cache)
print(f"数据类型: {dequantized_cache.dtype}")

# 比较原始数据和反量化数据
print("\n数据对比:")
print("原始数据:    ", cache.flatten())
print("反量化数据:  ", dequantized_cache.flatten())
print("误差:        ", (cache - dequantized_cache).flatten()) 