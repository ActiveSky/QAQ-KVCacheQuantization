# =============================================================================
# KEY-VALUE DIFFERENCE EXPERIMENT - Key缓存 vs Value缓存量化效果对比
# =============================================================================

"""
KeyValueDifference实验说明：
========================

1. 实验目的：
   - 对比分析：比较Key缓存和Value缓存量化的不同影响
   - 敏感性测试：分析两种缓存对量化精度的敏感性差异
   - 位宽优化：找到最优的量化位数配置

2. 设计特点：
   - 分离测试：分别量化Key或Value，另一个保持原始精度
   - 多位宽测试：测试1-8位均匀量化的效果
   - 多级别对比：token/layer/head三个量化级别对比

3. 实验设计：
   Group 1: Key量化 + Value不量化
     - Key缓存：1-8位均匀量化，三个级别
     - Value缓存：no-quantization，保持原始精度
   
   Group 2: Key不量化 + Value量化  
     - Key缓存：no-quantization，保持原始精度
     - Value缓存：1-8位均匀量化，三个级别

4. 参数配置：
   - 量化方法：uniform（均匀量化）
   - 对称性：False（非对称量化）
   - 异常值比例：0（无异常值处理）
   - 注意力感知：False（基础量化）

5. 输出分析：
   - 准确率曲线：不同位数下的准确率变化
   - 对比分析：Key vs Value量化的性能差异
   - 级别对比：token/layer/head级别的效果差异

6. 学习价值：
   - 理解KV缓存的特性差异
   - 指导量化策略的优化方向
   - 为QAQ算法提供设计依据

依赖关系：
- 继承：Experiment基类
- 工具：itertools.chain, product（迭代器操作）
- 可视化：matplotlib.pyplot（曲线绘制）
- 数据：EvaluationResult（结果数据）
"""

from .base import Experiment
from itertools import chain, product
from matplotlib import pyplot as plt
from functools import cached_property
from evaluator import EvaluationResult
from quantizer import Quantizer, build_quantizers


class KeyValueDifference(Experiment):
    @cached_property
    def quantizer_list(self) -> list[tuple[Quantizer, Quantizer]]:
        """
        量化器配置：Key-Value对比实验设计
        
        设计核心：
        - 分离测试：分别量化Key或Value，分析各自的敏感性
        - 多位宽：测试1-8位均匀量化，找到最优位宽
        - 多级别：对比token/layer/head三个量化级别
        
        实验分组：
        =================================================================
        Group 1: Key量化 + Value不量化
            目的：测试Key缓存量化的敏感性
            Key配置：
                - level: ["token", "layer", "head"] (三个级别）
                - n_bits_uniform: [1,2,3,4,5,6,7,8] (8个位宽）
                - method: "uniform" (均匀量化）
                - symmetric: False (非对称）
                - outliers_ratio: 0 (无异常值处理）
                - use_attentions: False (基础量化）
            Value配置：
                - level: "no-quantization" (保持原始精度）
            数量：3级别 × 8位宽 = 24个Key量化器
        
        Group 2: Key不量化 + Value量化
            目的：测试Value缓存量化的敏感性
            Key配置：
                - level: "no-quantization" (保持原始精度）
            Value配置：
                - 与Group1的Key配置相同
            数量：3级别 × 8位宽 = 24个Value量化器
        
        总计：48个量化器组合
        =================================================================
        
        itertools.chain作用：
        - 连接两个独立的迭代器序列
        - 将Group1和Group2的结果合并为一个列表
        - 便于统一处理和结果分析
        
        itertools.product作用：
        - 计算两个列表的笛卡尔积
        - 在Group1中：24个Key量化器 × 1个Value量化器 = 24个组合
        - 在Group2中：1个Key量化器 × 24个Value量化器 = 24个组合
        
        依赖关系：
        - 输入：build_quantizers()工厂函数
        - 输出：[(key_quantizer, value_quantizer), ...]
        - 后续：process_result()根据配置分离结果
        """
        # Group 1: Key量化 + Value不量化
        key_quantizers_1 = build_quantizers([{
            "key_or_value_cache": ["key"],                    # 处理Key缓存
            "use_attentions": [False],                      # 非注意力感知
            "method": ["uniform"],                          # 均匀量化
            "level": ["token", "layer", "head"],          # 三个量化级别
            "symmetric": [False],                           # 非对称量化
            "outliers_ratio": [0],                          # 无异常值处理
            "n_bits_uniform": [1, 2, 3, 4, 5, 6, 7, 8],  # 8个位宽测试
        }])
        
        value_quantizers_1 = build_quantizers([{
            "key_or_value_cache": ["value"],                 # 处理Value缓存
            "level": ["no-quantization"],                   # 保持原始精度
        }])
        
        # Group 2: Key不量化 + Value量化
        key_quantizers_2 = build_quantizers([{
            "key_or_value_cache": ["key"],                   # 处理Key缓存
            "level": ["no-quantization"],                   # 保持原始精度
        }])
        
        value_quantizers_2 = build_quantizers([{
            "key_or_value_cache": ["value"],                # 处理Value缓存
            "use_attentions": [False],                      # 非注意力感知
            "method": ["uniform"],                          # 均匀量化
            "level": ["token", "layer", "head"],          # 三个量化级别
            "symmetric": [False],                           # 非对称量化
            "outliers_ratio": [0],                          # 无异常值处理
            "n_bits_uniform": [1, 2, 3, 4, 5, 6, 7, 8],  # 8个位宽测试
        }])
        
        # 合并两组实验：chain连接两个product的结果
        return list(chain(
            product(key_quantizers_1, value_quantizers_1),  # Group1: 24个组合
            product(key_quantizers_2, value_quantizers_2),  # Group2: 24个组合
        ))

    def process_result(self, results: list[EvaluationResult]):
        """
        结果处理：Key-Value对比曲线生成
        
        设计要点：
        - 数据分离：根据实验分组分离Key和Value的量化结果
        - 系列组织：按量化级别组织数据系列
        - 可视化：生成对比曲线图展示差异
        
        处理逻辑：
        =================================================================
        Phase 1: 数据分离和组织
            输入：48个结果，对应48个量化器组合
            分离条件：
                - Group1 (Key量化): key_quantizer.level != "no-quantization"
                - Group2 (Value量化): value_quantizer.level != "no-quantization"
            筛选：只处理token/layer/head三个级别
            命名：使用"Key (level)-level"和"Value (level)-level"格式
        
        Phase 2: 数据结构构建
            series字典：{系列名称: [1位, 2位, 3位, ..., 8位]}
            初始化：[None] * 8 预分配8个位置
            数据填充：根据n_bits_uniform-1索引填充对应位置
            效果：每个系列包含8个位宽的准确率数据
        
        Phase 3: 可视化生成
            曲线绘制：6条曲线 (Key:3级别 + Value:3级别)
            样式区分：
                - Key量化：实线 (linestyle="solid")
                - Value量化：虚线 (linestyle="dashed")
            图表要素：
                - X轴：# of bits (1-8位）
                - Y轴：Accuracy (准确率）
                - 图例：每条曲线的标签
                - 高分辨率：dpi=400确保出版质量
        =================================================================
        
        数据组织示例：
            series = {
                "Key (token-level)": [0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95],
                "Key (layer-level)": [0.82, 0.85, 0.87, 0.89, 0.90, 0.91, 0.92, 0.93],
                "Key (head-level)":  [0.80, 0.83, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91],
                "Value (token-level)": [0.78, 0.82, 0.84, 0.86, 0.87, 0.88, 0.89, 0.90],
                "Value (layer-level)": [0.75, 0.79, 0.81, 0.83, 0.84, 0.85, 0.86, 0.87],
                "Value (head-level)":  [0.72, 0.76, 0.78, 0.80, 0.81, 0.82, 0.83, 0.84]
            }
        
        分析价值：
        - 敏感性对比：Key vs Value量化对精度的不同影响
        - 级别对比：token/layer/head级别的量化效果差异
        - 位宽优化：找到每个级别的最优量化位数
        - 算法指导：为QAQ算法提供设计依据
        
        依赖关系：
        - 输入：EvaluationResult列表（与quantizer_list一一对应）
        - 量化器配置：通过zip(self.quantizer_list, results)关联
        - 可视化：matplotlib.pyplot库
        - 输出：figs/key_value_difference.png文件
        
        关键洞察预期：
        - Key缓存通常对量化更敏感（影响检索精度）
        - Value缓存相对更鲁棒（对精度影响较小）
        - token-level量化通常效果最好（粒度最细）
        - 随着位数增加，所有级别的准确率都趋向原始精度
        """
        # 定义启用的量化级别（排除"no-quantization"）
        enabled_series = ["token", "layer", "head"]
        series: dict[str, list[float]] = {}
        
        # Phase 1: 数据分离和组织
        for (key_quantizer, value_quantizer), result in zip(self.quantizer_list, results):
            # Group2处理：Key不量化 + Value量化
            if key_quantizer.level == "no-quantization":
                if value_quantizer.level not in enabled_series:
                    continue  # 跳过非目标级别
                name = f"Value ({value_quantizer.level}-level)"
                if name not in series:
                    series[name] = [None] * 8  # 预分配8个位置
                # 填充对应位宽的准确率（索引从0开始）
                series[name][value_quantizer.n_bits_uniform-1] = result.accuracy
            
            # Group1处理：Key量化 + Value不量化
            elif value_quantizer.level == "no-quantization":
                if key_quantizer.level not in enabled_series:
                    continue  # 跳过非目标级别
                name = f"Key ({key_quantizer.level}-level)"
                if name not in series:
                    series[name] = [None] * 8  # 预分配8个位置
                # 填充对应位宽的准确率（索引从0开始）
                series[name][key_quantizer.n_bits_uniform-1] = result.accuracy
        
        # Phase 3: 可视化生成
        for name, data in series.items():
            # 绘制曲线：Key用实线，Value用虚线
            plt.plot([1,2,3,4,5,6,7,8], data, 
                    label=name, 
                    linestyle="solid" if name.startswith("Key") else "dashed")
        
        # 图表配置和保存
        plt.legend()           # 显示图例
        plt.xlabel("# of bits")  # X轴标签
        plt.ylabel("Accuracy")  # Y轴标签
        plt.savefig("figs/key_value_difference.png", dpi=400)  # 高分辨率保存
        print(series)  # 打印数据供调试分析
