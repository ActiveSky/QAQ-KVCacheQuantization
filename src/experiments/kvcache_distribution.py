# =============================================================================
# KV CACHE DISTRIBUTION EXPERIMENT - KV缓存数值分布分析
# =============================================================================

"""
KVcacheDistribution实验说明：
=========================

1. 实验目的：
   - 分布分析：分析KV缓存的数值分布特征
   - 量化基础：为量化算法设计提供数据支持
   - 统计特性：理解KV缓存的统计规律

2. 设计特点：
   - 数据驱动：基于实际模型推理数据
   - 高精度统计：使用1000个bin的直方图
   - 大样本：使用question_count个问题统计
   - 原始数据：不进行任何量化处理

3. 技术要点：
   - 范围选择：[-5.0, 5.0] 覆盖大部分数值
   - 数据类型：转换为float32保证统计精度
   - 内存优化：使用CPU避免GPU内存不足
   - 进度显示：使用tqdm显示处理进度

4. 输出内容：
   - 分布图：Key和Value缓存的分布对比
   - 原始数据：保存为npy文件供后续分析
   - 统计特征：均值、方差、分布形状等

5. 对量化的指导意义：
   - 分布假设：验证uniform vs normal量化假设
   - 范围确定：指导量化范围的设置
   - 异常值处理：了解异常值的分布情况
   - 位数选择：为最优位数选择提供依据

6. 学习价值：
   - 理解KV缓存的统计特性
   - 掌握数据分布分析方法
   - 为量化算法设计提供数据支撑
   - 理解模型内部数据表示

依赖关系：
- 继承：Experiment基类
- 配置：device_configs[0][0]使用第一个设备
- 数据：self.questions和self.get_model()
- 工具：torch, numpy, matplotlib, tqdm
"""

import torch
import numpy as np
from tqdm import tqdm
from .base import Experiment
from config import device_configs
from matplotlib import pyplot as plt


class KVcacheDistribution(Experiment):
    def process_result(self, _):
        """
        KV缓存分布统计和可视化
        
        设计要点：
        - 大样本统计：使用所有问题收集足够的数据
        - 高精度直方图：1000个bin提供精细分布信息
        - 数值范围：[-5.0, 5.0]覆盖大多数数值
        - 内存优化：使用CPU避免GPU内存压力
        
        技术实现：
        =================================================================
        Phase 1: 初始化配置
            n_bins = 1000: 高精度直方图
            model = self.get_model(0): 加载模型到worker 0
            device = device_configs[0][0]: 使用第一个设备
            初始化统计张量：key_cache_hist, value_cache_hist
        
        Phase 2: 数据收集
            遍历问题：使用tqdm显示进度
            模型推理：forward(use_cache=True)获取KV缓存
            数据提取：从past_key_values提取key和value缓存
            形状处理：(n_layer, 1, n_head, seq_len, embed_size_per_head)
            数据扁平化：view(-1)转换为一维数组
            统计更新：使用torch.histc计算直方图
        
        Phase 3: 结果处理和保存
            数据转换：torch.Tensor → numpy.array
            可视化：绘制分布对比曲线
            数据保存：保存npy文件供后续分析
        =================================================================
        
        核心函数说明：
        - torch.histc(): 计算直方图统计
        - torch.view(): 张量形状重塑
        - torch.no_grad(): 禁用梯度计算节省内存
        - numpy.linspace(): 生成X轴坐标范围
        
        输出文件：
        - figs/cache_distribution.png: 分布对比图
        - data/key_cache_hist.npy: Key缓存直方图数据
        - data/value_cache_hist.npy: Value缓存直方图数据
        
        预期发现：
        - Key缓存：通常呈正态分布，均值接近0
        - Value缓存：可能有一定的偏态分布
        - 数值范围：大部分值在[-3, 3]之间
        - 异常值：少量值在[-5, 5]范围外
        
        量化算法指导：
        - uniform量化：适合均匀分布的数据
        - normal量化：适合正态分布的数据
        - 异常值比例：根据分布确定outliers_ratio
        - 量化范围：根据分布确定min/max值
        """
        # Phase 1: 初始化配置
        n_bins = 1000  # 高精度直方图
        model = self.get_model(0)  # 加载模型到worker 0
        device = device_configs[0][0]  # 使用第一个设备
        
        # 初始化统计张量（在CPU上避免GPU内存不足）
        key_cache_hist, value_cache_hist = torch.zeros(n_bins, dtype=torch.int64), torch.zeros(n_bins, dtype=torch.int64)
        
        # Phase 2: 数据收集
        with torch.no_grad():  # 禁用梯度计算，节省内存
            for question in tqdm(self.datasets.questions):  # 显示处理进度
                length = question.question_length
                input_ids = question.input_ids[:1,:length].to(device)  # 单个batch
                
                # 模型推理获取KV缓存
                kvcache = model.forward(input_ids, use_cache=True, return_dict=True).past_key_values
                
                # 提取并处理Key缓存
                key_cache = torch.stack([key.to(device) for key, _ in kvcache]).cpu()
                # key_cache.shape: (n_layer, 1, n_head, seq_len, embed_size_per_head)
                key_cache = key_cache.view(-1).to(dtype=torch.float32)  # 扁平化
                
                # 提取并处理Value缓存
                value_cache = torch.stack([value.to(device) for _, value in kvcache]).cpu()
                # value_cache.shape: (n_layer, 1, n_head, seq_len, embed_size_per_head)
                value_cache = value_cache.view(-1).to(dtype=torch.float32)  # 扁平化
                
                # 更新直方图统计
                key_cache_hist += torch.histc(key_cache, bins=n_bins, min=-5.0, max=5.0).to(dtype=torch.int64)
                value_cache_hist += torch.histc(value_cache, bins=n_bins, min=-5.0, max=5.0).to(dtype=torch.int64)
        
        # Phase 3: 结果处理和保存
        key_cache_hist = key_cache_hist.detach().numpy()  # 转换为numpy数组
        value_cache_hist = value_cache_hist.detach().numpy()
        x_range = np.linspace(-5.0, 5.0, 1000)  # X轴坐标
        
        # 绘制分布对比图
        plt.plot(x_range, key_cache_hist, label="Key cache distribution")
        plt.plot(x_range, value_cache_hist, label="Value cache distribution")
        plt.legend()
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.savefig("figs/cache_distribution.png")
        
        # 保存原始数据供后续分析
        np.save("data/key_cache_hist.npy", key_cache_hist)
        np.save("data/value_cache_hist.npy", value_cache_hist)
