# 导入数学模块，用于数学计算
import math

# 导入PyTorch库，用于张量操作
import torch

# 导入NumPy库，用于数值计算
import numpy as np

# 导入自定义模型模块，用于类型注解
from models import CausalLM

# 导入SciPy统计模块，用于正态分布计算
from scipy.stats import norm

# 导入itertools模块，用于笛卡尔积计算
from itertools import product

# 导入functools模块的cached_property装饰器，用于缓存属性值
from functools import cached_property

# 导入类型提示模块，用于类型注解
from typing import Literal, Optional, Any


# 定义注意力类型别名，为张量列表
AttentionType = list[torch.Tensor]

# 定义量化级别类型字面量
QuantizationLevels = Literal["no-quantization", "token", "layer", "head"]

# 定义量化方法类型字面量
QuantizationMethods = Literal["uniform", "normal"]


# 量化器类，用于KV缓存的量化处理
class Quantizer:
    """量化器类，用于对大语言模型中的键值缓存进行量化处理以减少内存占用。

    该类实现了多种量化策略，包括不同级别的量化（token、layer、head）、对称/非对称量化、
    均匀分布量化和正态分布量化，以及基于注意力机制的自适应比特分配。

    Attributes:
        key_or_value_cache (Literal["key", "value"]): 指定量化键缓存还是值缓存
        level (QuantizationLevels): 量化级别 ("no-quantization", "token", "layer", "head")
        symmetric (bool): 是否使用对称量化
        method_name (QuantizationMethods): 量化方法 ("uniform", "normal")
        use_attentions (bool): 是否启用注意力感知量化
        quantize_dims (tuple): 根据量化级别确定的量化维度
        outliers_ratio (float): 异常值比例
        quantization_method (callable): 实际使用的量化方法函数
    """
    def __init__(self,
                 # 键缓存或值缓存
                 key_or_value_cache: Optional[Literal["key", "value"]] = None,
                 # 无量化、token级别、层级别或头级别量化
                 level: Optional[QuantizationLevels] = None,
                 # True: 假设缓存已经是零中心化的，只进行缩放
                 # False: 将缓存零中心化然后进行缩放
                 symmetric: Optional[bool] = None,
                 # uniform: 假设归一化的缓存值在最大值和最小值之间服从均匀分布
                 # normal: 假设归一化的缓存值服从标准正态分布
                 method: Optional[QuantizationMethods] = None,
                 # 异常值百分比（包括最低和最高值）
                 outliers_ratio: Optional[float] = None,
                 # 是否启用注意力感知量化
                 use_attentions: Optional[bool] = None,
                 # （仅适用于均匀量化）
                 # 均匀量化的比特数
                 n_bits_uniform: Optional[int] = None,
                 # （仅适用于注意力感知量化）
                 # 使用注意力的最后n行来计算量化比特数
                 last_n_attentions: Optional[int] = None,
                 # （仅适用于注意力感知量化）
                 # 目标量化误差
                 target_quantization_error: Optional[float] = None,
                 # （仅适用于注意力感知量化）
                 # 最小允许的量化比特数
                 n_bits_min: Optional[int] = None,
                 # （仅适用于注意力感知量化）
                 # 最大允许的量化比特数
                 n_bits_max: Optional[int] = None,
                 # （仅适用于键缓存的注意力感知量化）
                 # 公式中使用的查询张量的2-范数
                 q_norm: Optional[float] = None):
        """初始化量化器

        Args:
            key_or_value_cache: 指定量化键缓存还是值缓存
            level: 量化级别，可选值包括"no-quantization", "token", "layer", "head"
            symmetric: 是否使用对称量化 (True: 假设缓存已零中心化; False: 需要零中心化)
            method: 量化方法，可选"uniform"(均匀分布)或"normal"(正态分布)
            outliers_ratio: 异常值比例（包括最低和最高值）
            use_attentions: 是否启用注意力感知量化
            n_bits_uniform: 均匀量化的比特数（当use_attentions=False时使用）
            last_n_attentions: 使用注意力的最后n行来计算量化比特数（注意力感知量化时使用）
            target_quantization_error: 目标量化误差（注意力感知量化时使用）
            n_bits_min: 最小允许的量化比特数（注意力感知量化时使用）
            n_bits_max: 最大允许的量化比特数（注意力感知量化时使用）
            q_norm: 查询张量的2-范数（键缓存的注意力感知量化时使用）
        """
        # 设置key_or_value_cache，必须指定是量化键缓存还是值缓存
        assert key_or_value_cache is not None
        self.key_or_value_cache = key_or_value_cache

        # 无量化情况的早期退出，如果不需要量化则直接返回
        assert level is not None
        self.level = level
        if level == "no-quantization":
            return

        # 设置量化维度，根据不同的量化级别设置对应的维度
        # token级别：在所有维度(-3, -2, -1)上进行量化
        # layer级别：在层和头维度(-2, -1)上进行量化
        # head级别：仅在头维度(-1)上进行量化
        if level == "token":
            self.quantize_dims = (-3, -2, -1)
        elif level == "layer":
            self.quantize_dims = (-2, -1)
        elif level == "head":
            self.quantize_dims = (-1,)

        # 设置对称性，确定是否使用对称量化
        assert symmetric is not None
        self.symmetric = symmetric

        # 设置异常值比率，用于处理分布中的极端值
        assert outliers_ratio is not None
        self.outliers_ratio = outliers_ratio

        # 设置use_attentions，确定是否使用注意力感知量化
        assert use_attentions is not None
        self.use_attentions = use_attentions
        if use_attentions:
            # 注意力感知量化参数设置
            # 设置last_n_attentions，使用注意力的最后n行来计算量化比特数
            assert last_n_attentions is not None
            assert last_n_attentions > 0
            self.last_n_attentions = last_n_attentions

            # 设置target_quantization_error，目标量化误差
            assert target_quantization_error is not None
            assert target_quantization_error > 0.0
            self.target_quantization_error = target_quantization_error

            # 设置n_bits_min，最小允许的量化比特数
            assert n_bits_min is not None
            assert 0 <= n_bits_min <= 16
            self.n_bits_min = n_bits_min

            # 设置n_bits_max，最大允许的量化比特数
            assert n_bits_max is not None
            assert n_bits_min <= n_bits_max <= 16
            self.n_bits_max = n_bits_max

            # 如果是键缓存的注意力感知量化，需要设置q_norm
            if self.key_or_value_cache == "key":
                # 设置q_norm，公式中使用的查询张量的2-范数
                assert q_norm is not None
                assert q_norm > 0
                self.q_norm = q_norm
        else:
            # 非注意力感知量化参数设置
            # 设置n_bits_uniform，均匀量化的比特数
            assert n_bits_uniform is not None
            assert 0 <= n_bits_uniform <= 16
            self.n_bits_uniform = n_bits_uniform

        # 设置方法，确定使用均匀量化还是正态量化
        assert method is not None
        self.method_name = method
        if method == "uniform":
            self.quantization_method = self._uniform_quantize
        elif method == "normal":
            self.quantization_method = self._normal_quantize

    def set_dtype_and_device(self, dtype: torch.dtype, device: torch.device):
        """设置数据类型和设备

        Args:
            dtype: PyTorch数据类型
            device: PyTorch设备（CPU或GPU）
        """
        self.dtype = dtype
        self.device = device
        # 如果使用正态分布量化，需要预先计算分位数
        if self.level != "no-quantization" and self.method_name == "normal":
            # 根据是否使用注意力感知量化确定比特数范围
            if self.use_attentions:
                n_bits_range = range(self.n_bits_min, self.n_bits_max+1)
            else:
                n_bits_range = range(self.n_bits_uniform, self.n_bits_uniform+1)
            # 计算正态分布的上界分位数，用于量化时的searchsorted操作
            self.normal_quantiles_upper_bound = {
                n: torch.tensor(norm.ppf(np.arange(0, 1, 1/(2**n)) + 1/(2**n)), dtype=dtype, device=device)
                for n in n_bits_range
            }
            # 计算正态分布的中心分位数，用于反量化
            self.normal_quantiles_center = {
                n: torch.tensor(norm.ppf(np.arange(0, 1, 1/(2**n)) + 0.5/(2**n)), dtype=dtype, device=device)
                for n in n_bits_range
            }

    @cached_property
    def params(self) -> dict[str, Any]:
        """缓存属性：参数字典，包含量化器配置信息

        Returns:
            dict: 包含量化器所有配置参数的字典
        """
        res: dict[str, Any] = {}
        res["key_or_value_cache"] = self.key_or_value_cache
        res["level"] = self.level
        # 如果不进行量化，直接返回基本参数
        if self.level == "no-quantization":
            return res
        res["symmetric"] = self.symmetric
        res["method_name"] = self.method_name
        res["outliers_ratio"] = self.outliers_ratio
        res["use_attentions"] = self.use_attentions
        # 如果使用注意力感知量化，添加相关参数
        if self.use_attentions:
            res["n_bits_min"] = self.n_bits_min
            res["n_bits_max"] = self.n_bits_max
            res["last_n_attentions"] = self.last_n_attentions
            res["target_quantization_error"] = self.target_quantization_error
            # 如果是键缓存，还需要添加q_norm参数
            if self.key_or_value_cache == "key":
                res["q_norm"] = self.q_norm
        else:
            # 如果不使用注意力感知量化，添加均匀量化比特数参数
            res["n_bits_uniform"] = self.n_bits_uniform
        return res

    def _calc_quantization_bits(self, attentions: AttentionType, cache: torch.Tensor, outlier_mask: torch.Tensor) -> torch.Tensor:
        """计算量化比特数

        根据注意力感知量化算法计算每个量化单元应该使用的比特数。
        对于非注意力感知量化，直接返回统一的比特数。

        Args:
            attentions: 注意力张量列表
            cache: KV缓存张量
            outlier_mask: 异常值掩码

        Returns:
            torch.Tensor: 每个量化单元的比特数张量
        """
        # cache/outlier_mask.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        n_batch, seq_len, n_layer, n_head, _ = cache.shape

        # 如果不使用注意力感知量化，直接返回统一的比特数
        if not self.use_attentions:
            if self.level == "token":
                shape = (n_batch, seq_len)
            elif self.level == "layer":
                shape = (n_batch, seq_len, n_layer)
            elif self.level == "head":
                shape = (n_batch, seq_len, n_layer, n_head)
            return torch.ones(shape, dtype=torch.int64, device=self.device) * self.n_bits_uniform

        # 根据键缓存或值缓存计算最大允许误差
        if self.key_or_value_cache == "key":
            # 键缓存的最大误差计算公式
            max_error = math.sqrt(12.0 / self.q_norm * math.log(seq_len**3/(seq_len-1) * self.target_quantization_error**2 + 1))
            # max_error.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
        elif self.key_or_value_cache == "value":
            # 值缓存的最大误差计算，基于注意力权重
            attentions = torch.stack(attentions)
            # attentions.shape: (n_layer, n_batch, n_head, seq_len, seq_len)
            attentions = attentions[:, :, :, -self.last_n_attentions:, :]
            # attentions.shape: (n_layer, n_batch, n_head, last_n_attentions, seq_len)
            attentions = attentions.permute(1, 4, 0, 2, 3)
            # attentions.shape: (n_batch, seq_len, n_layer, n_head, last_n_attentions)
            # 计算注意力权重的最大值
            attentions = attentions.amax(dim=self.quantize_dims)
            # attentions.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
            # 值缓存的最大误差计算公式
            max_error = math.sqrt(12.0 / seq_len) * self.target_quantization_error / attentions
            # max_error.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)

        # 创建掩码张量，排除异常值
        cache = torch.masked.masked_tensor(cache, torch.logical_not(outlier_mask))
        # NOTE: PyTorch's bug: https://github.com/pytorch/pytorch/issues/115624
        # 转换为正索引
        quantize_dims = [x + len(cache.shape) for x in self.quantize_dims]

        # 根据量化方法计算比特数
        if self.method_name == "uniform":
            # 均匀量化比特数计算
            if self.symmetric:
                # 对称量化，使用绝对值的最大值
                max_value = cache.abs().amax(dim=quantize_dims)
                scale_value = 2 * max_value
            else:
                # 非对称量化，使用最大值和最小值
                max_value = cache.amax(dim=quantize_dims)
                min_value = cache.amin(dim=quantize_dims)
                scale_value = max_value - min_value
            assert scale_value.get_mask().all()
            scale_value = scale_value.get_data()
            # scale_value.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
            # 根据量化误差公式计算所需比特数
            n_bits = torch.log2(scale_value / (2 * max_error) + 1)
            # n_bits.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
        elif self.method_name == "normal":
            # 正态分布量化暂未实现
            raise NotImplementedError()

        # 将计算得到的比特数限制在允许的范围内
        n_bits = torch.clamp(torch.ceil(n_bits).to(torch.int64), self.n_bits_min, self.n_bits_max)
        # n_bits.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)

        # 最后的(last_n_attentions-1)个token没有足够的历史注意力信息，所以不进行量化
        if self.last_n_attentions > 1:
            n_bits[:, -self.last_n_attentions+1:] = self.n_bits_max

        return n_bits

    def _calc_outlier_mask(self, cache: torch.Tensor) -> torch.Tensor:
        """计算异常值掩码

        根据设定的异常值比例，计算缓存中哪些值应该被视为异常值并避免量化。

        Args:
            cache: KV缓存张量

        Returns:
            torch.Tensor: 异常值掩码，异常值位置为True，其他位置为False
        """
        # 如果异常值比例为0，则不标记任何异常值
        if self.outliers_ratio == 0.0:
            return torch.zeros_like(cache, dtype=torch.bool, device=self.device)
        # cache.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)

        # 将缓存在量化维度上展平，便于计算分位数
        cache_flat = cache.flatten(start_dim=-len(self.quantize_dims))
        # cache_flat.shape: (n_batch, seq_len, n_layer*n_head*embed_size_per_head) or (n_batch, seq_len, n_layer, n_head*embed_size_per_head) or (n_batch, seq_len, n_layer, n_head, embed_size_per_head)

        # 计算下界和上界阈值
        # 下界阈值：位于下半部分异常值范围的边界
        lower_threshold = torch.kthvalue(cache_flat, k=int(math.ceil(self.outliers_ratio/2 * cache_flat.shape[-1])), dim=-1).values
        # 上界阈值：位于上半部分异常值范围的边界
        upper_threshold = torch.kthvalue(cache_flat, k=int(math.floor((1.0 - self.outliers_ratio/2) * cache_flat.shape[-1])), dim=-1).values
        # lower_threshold/upper_threshold.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)

        # 调整阈值张量的形状以匹配原始缓存张量
        lower_threshold = lower_threshold.view(*lower_threshold.shape, *((1,)*len(self.quantize_dims)))
        upper_threshold = upper_threshold.view(*upper_threshold.shape, *((1,)*len(self.quantize_dims)))
        # lower_threshold/upper_threshold.shape: (n_batch, seq_len, 1, 1, 1) or (n_batch, seq_len, n_layer, 1, 1) or (n_batch, seq_len, n_layer, n_head, 1)

        # 创建异常值掩码：小于等于下界阈值或大于上界阈值的值被视为异常值
        mask = (cache <= lower_threshold) | (cache > upper_threshold)
        # mask.shape = (n_batch, seq_len, n_layer, n_head, embed_size_per_head)

        return mask

    def _normalize(self, cache: torch.Tensor, method: Literal["minmax", "std"], n_bits: int, outlier_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """归一化函数，返回（归一化缓存，均值，缩放值）

        将缓存值归一化到[-1, 1]或标准正态分布范围，便于量化处理。

        Args:
            cache: KV缓存张量
            method: 归一化方法 ("minmax" 或 "std")
            n_bits: 量化比特数
            outlier_mask: 异常值掩码

        Returns:
            tuple: (归一化缓存, 均值, 缩放值)
        """
        # cache/outlier_mask.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        # 使用掩码张量排除异常值
        cache = torch.masked.masked_tensor(cache, torch.logical_not(outlier_mask))

        # 根据是否对称量化计算均值
        if self.symmetric:
            # 对称量化：均值为0
            mean_value = torch.zeros((1,)*cache.dim(), dtype=self.dtype, device=self.device)
            # 根据方法计算缩放值
            if method == "minmax":
                # 最小-最大归一化：使用绝对值的最大值
                max_value = cache.abs().amax(dim=self.quantize_dims, keepdim=True)
                scale_value = 2 * max_value / (2 ** n_bits)
            elif method == "std":
                # 标准差归一化：使用标准差
                scale_value = cache.std(dim=self.quantize_dims, keepdim=True)
        else:
            # 非对称量化：计算实际均值
            mean_value = cache.mean(dim=self.quantize_dims, keepdim=True)
            # 根据方法计算缩放值
            if method == "minmax":
                # 最小-最大归一化：使用最大值和最小值
                max_value = cache.amax(dim=self.quantize_dims, keepdim=True)
                min_value = cache.amin(dim=self.quantize_dims, keepdim=True)
                scale_value = (max_value - min_value) / (2 ** n_bits)
            elif method == "std":
                # 标准差归一化：使用标准差（转换为float64以提高精度后再转回原类型）
                scale_value = cache.to(torch.float64).std(dim=self.quantize_dims, keepdim=True).to(self.dtype)

        # 确保均值和缩放值的掩码都是有效的
        assert mean_value.get_mask().all()
        assert scale_value.get_mask().all()

        # 提取掩码张量的数据部分
        cache = cache.get_data()
        mean_value = mean_value.get_data()
        scale_value = scale_value.get_data()
        # mean_value/scale_value.shape: (n_count, n_layer/1, n_head/1, embed_size_per_head/1) or (n_count, n_head/1, embed_size_per_head/1), (n_count, embed_size_per_head/1)

        # 执行归一化操作
        normalized_cache = (cache - mean_value) / scale_value
        # normalized_cache.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)

        return normalized_cache, mean_value, scale_value

    def _denormalize(self, normalized_cache: torch.Tensor, mean_value: torch.Tensor, scale_value: torch.Tensor) -> torch.Tensor:
        """反归一化函数

        将归一化的缓存值还原为原始范围。

        Args:
            normalized_cache: 归一化的缓存张量
            mean_value: 均值张量
            scale_value: 缩放值张量

        Returns:
            torch.Tensor: 反归一化后的缓存张量
        """
        # 执行反归一化操作：缩放值 * 归一化缓存 + 均值
        return normalized_cache * scale_value + mean_value

    def _uniform_quantize(self, cache: torch.Tensor, n_bits: int, outlier_mask: torch.Tensor) -> torch.Tensor:
        """均匀量化函数

        使用均匀分布假设对缓存进行量化处理。

        Args:
            cache: KV缓存张量
            n_bits: 量化比特数
            outlier_mask: 异常值掩码

        Returns:
            torch.Tensor: 量化后的缓存张量
        """
        # cache/outlier_mask.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        # 首先对缓存进行归一化处理，使用最小-最大归一化方法
        normalized_cache, mean_value, scale_value = self._normalize(cache, "minmax", n_bits, outlier_mask)

        # 执行均匀量化：四舍五入到最近的整数，并限制在量化范围內
        quantized_cache = torch.clamp(torch.round(normalized_cache).to(torch.int32), -(2 ** (n_bits-1)), 2 ** (n_bits-1) - 1)

        # 反量化：将整数量化值转换回原始数据类型
        dequantized_cache = quantized_cache.to(self.dtype)

        # 反归一化：将量化值还原到原始范围
        denormalized_cache = self._denormalize(dequantized_cache, mean_value, scale_value)
        # denormalized_cache.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)

        # 对于异常值位置，保留原始缓存值；对于非异常值位置，使用量化后的值
        return torch.where(outlier_mask, cache, denormalized_cache)

    def _normal_quantize(self, cache: torch.Tensor, n_bits: int, outlier_mask: torch.Tensor) -> torch.Tensor:
        """正态量化函数

        使用正态分布假设对缓存进行量化处理。

        Args:
            cache: KV缓存张量
            n_bits: 量化比特数
            outlier_mask: 异常值掩码

        Returns:
            torch.Tensor: 量化后的缓存张量
        """
        # cache/outlier_mask.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        # 首先对缓存进行归一化处理，使用标准差归一化方法
        normalized_cache, mean_value, scale_value = self._normalize(cache, "std", n_bits, outlier_mask)

        # 执行正态量化：使用预计算的分位数表进行量化
        # 通过searchsorted找到每个值在分位数表中的位置
        quantized_cache = torch.searchsorted(self.normal_quantiles_upper_bound[n_bits], normalized_cache.contiguous())

        # 反量化：使用预计算的中心分位数值
        dequantized_cache = self.normal_quantiles_center[n_bits][quantized_cache]

        # 反归一化：将量化值还原到原始范围
        denormalized_cache = self._denormalize(dequantized_cache, mean_value, scale_value)
        # denormalized_cache.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)

        # 对于异常值位置，保留原始缓存值；对于非异常值位置，使用量化后的值
        return torch.where(outlier_mask, cache, denormalized_cache)

    def quantize(self, cache: torch.Tensor, attentions: AttentionType) -> tuple[torch.Tensor, float]:
        """量化函数，返回（量化后的kvcache，平均比特数）

        主量化函数，执行完整的量化流程：计算异常值掩码、计算量化比特数、执行量化操作。

        Args:
            cache: KV缓存张量，形状为(n_layer, n_batch, n_head, seq_len, embed_size_per_head)
            attentions: 注意力张量列表

        Returns:
            tuple: (量化后的KV缓存张量, 平均比特数)
        """
        # 如果不进行量化，直接返回原始缓存和数据类型的比特数
        if self.level == "no-quantization":
            return cache, torch.finfo(self.dtype).bits
        # cache.shape: (n_layer, n_batch, n_head, seq_len, embed_size_per_head)

        # 调整缓存张量的维度顺序，便于处理
        cache = cache.permute(1, 3, 0, 2, 4)
        # cache.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)

        # 计算异常值掩码，标识哪些值应该避免量化
        outlier_mask = self._calc_outlier_mask(cache)
        # outlier_mask.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)

        # 计算每个量化单元应该使用的比特数
        n_bits = self._calc_quantization_bits(attentions, cache, outlier_mask)
        # n_bits.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)

        # 计算平均比特数，考虑异常值的影响
        average_n_bits = n_bits.mean(dtype=self.dtype).item()
        # 平均比特数 = 普通值的平均比特数 * (1 - 异常值比例) + 异常值的比特数 * 异常值比例
        average_n_bits = average_n_bits * (1 - self.outliers_ratio) + torch.finfo(self.dtype).bits * self.outliers_ratio

        # 获取比特数的最小值和最大值，用于循环处理不同比特数的量化单元
        n_bits_min, n_bits_max = n_bits.min().item(), n_bits.max().item()

        # 克隆缓存张量，避免修改原始数据
        cache = cache.clone()

        # 对每个比特数级别的量化单元分别进行量化处理
        for n in range(n_bits_min, n_bits_max+1):
            # 找到需要使用n比特量化的单元索引
            indices = torch.where(n_bits == n)
            # 对这些单元执行量化操作
            cache[indices] = self.quantization_method(cache[indices], n_bits=n, outlier_mask=outlier_mask[indices])

        # 恢复缓存张量的原始维度顺序
        cache = cache.permute(2, 0, 3, 1, 4)
        # cache.shape: (n_layer, n_batch, n_head, seq_len, embed_size_per_head)

        return cache, average_n_bits

    def calc_quantized_cache_size_per_token(self, average_n_bits: float, model: CausalLM) -> float:
        """计算量化缓存每令牌的大小

        根据平均比特数和模型配置计算每个令牌的量化缓存大小。

        Args:
            average_n_bits: 平均比特数
            model: 大语言模型实例

        Returns:
            float: 每个令牌的量化缓存大小（比特）
        """
        # 计算基本缓存大小：平均比特数 × 层数 × 隐藏层大小
        cache_size = average_n_bits * model.config.num_hidden_layers * model.config.hidden_size

        # 获取数据类型的默认比特数（如float16为16比特）
        default_n_bits = torch.finfo(self.dtype).bits

        # 计算额外信息大小因子
        # 无量化：0
        # 对称量化：1（只需要存储缩放因子）
        # 非对称量化：2（需要存储均值和缩放因子）
        n_extra = 0 if self.level == "no-quantization" else 1 if self.symmetric else 2

        # 根据量化级别计算额外信息大小
        if self.level == "no-quantization":
            # 无量化不需要额外信息
            extra_size = 0
        elif self.level == "token":
            # token级别：每个令牌需要存储额外信息
            extra_size = n_extra * default_n_bits
        elif self.level == "layer":
            # layer级别：每层需要存储额外信息
            extra_size = n_extra * default_n_bits * model.config.num_hidden_layers
        elif self.level == "head":
            # head级别：每个注意力头需要存储额外信息
            extra_size = n_extra * default_n_bits * model.config.num_hidden_layers * model.config.num_attention_heads

        # 返回总大小（基本缓存大小 + 额外信息大小）
        return cache_size + extra_size


def build_quantizers(config_grid_list: list[dict[str, list]]) -> list[Quantizer]:
    """构建量化器列表的函数

    根据配置网格列表生成所有可能的量化器组合。

    Args:
        config_grid_list: 配置网格列表，每个元素是一个字典，键为参数名，值为参数值列表

    Returns:
        list: 量化器实例列表
    """
    quantizer_list: list[Quantizer] = []
    # 遍历每个配置网格
    for config_grid in config_grid_list:
        # 使用笛卡尔积生成所有可能的参数组合
        for args in product(*config_grid.values()):
            # 将参数名和参数值组合成关键字参数字典
            kwargs = {k: v for k, v in zip(config_grid.keys(), args)}
            # 创建量化器实例并添加到列表中
            quantizer_list.append(Quantizer(**kwargs))
    return quantizer_list