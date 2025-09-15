# 导入操作系统接口模块，用于文件路径操作
import os

# 导入垃圾回收模块，用于手动释放内存
import gc

# 导入JSON模块，用于结果缓存的序列化和反序列化
import json

# 导入数学模块，用于数学计算
import math

# 导入PyTorch库，用于张量计算
import torch

# 导入tqdm模块，用于显示进度条
from tqdm import tqdm

# 导入类型提示模块，用于类型注解
from typing import Optional, Any

# 导入PyTorch神经网络函数模块
from torch.nn import functional as F

# 导入functools模块的cached_property装饰器，用于缓存属性值
from functools import cached_property

# 导入dataclasses模块，用于创建数据类
from dataclasses import dataclass, asdict

# User code packages
# 导入QA数据集模块，用于加载问题数据
from qa_dataset import QADataset, Question

# 导入量化器模块，用于KV缓存量化
from quantizer import Quantizer, AttentionType

# 导入模型模块，用于定义模型类型
from models import CausalLM

# 定义评估结果数据类，包含各种评估指标
@dataclass
class EvaluationResult:
    # 准确率
    accuracy: float = 0.0
    
    # 准确率置信度（95%置信区间）
    accuracy_confidence: float = 0.0
    
    # 答案对数概率
    answer_log_probability: float = 0.0
    
    # 量化误差（键和值的平均误差）
    quantization_error: float = 0.0
    
    # 键量化误差
    key_quantization_error: float = 0.0
    
    # 值量化误差
    value_quantization_error: float = 0.0
    
    # 注意力误差
    attention_error: float = 0.0
    
    # 逻辑误差
    logit_error: float = 0.0
    
    # 平均比特数（键和值的平均）
    average_n_bits: float = 0.0
    
    # 键平均比特数
    key_average_n_bits: float = 0.0
    
    # 值平均比特数
    value_average_n_bits: float = 0.0
    
    # 平均大小（键和值的平均）
    average_size: float = 0.0
    
    # 键平均大小
    key_average_size: float = 0.0
    
    # 值平均大小
    value_average_size: float = 0.0


# 评估器类，用于执行模型评估
class Evaluator:
    # 初始化评估器
    def __init__(self, device: torch.device,
                 version: str,
                 model_name: str,
                 datasets: QADataset,
                 key_quantizer: Quantizer,
                 value_quantizer: Quantizer):
        # 设备对象，指定计算设备（CPU/GPU）
        self.device = device
        
        # 版本号，用于标识实验配置
        self.version = version
        
        # 模型名称，指定要评估的模型
        self.model_name = model_name
        
        # 数据集对象，包含评估问题
        self.datasets = datasets
        
        # 键量化器，用于量化键缓存
        self.key_quantizer = key_quantizer
        
        # 值量化器，用于量化值缓存
        self.value_quantizer = value_quantizer

    # 缓存属性：参数字典，包含评估配置信息
    @cached_property
    def params(self) -> dict[str, Any]:
        # 创建结果字典
        res: dict[str, Any] = {}
        
        # 添加版本号
        res["version"] = self.version
        
        # 添加模型名称
        res["model_name"] = self.model_name
        
        # 添加数据集名称
        res["dataset_name"] = self.datasets.dataset_name
        
        # 添加问题数量
        res["question_count"] = self.datasets.question_count
        
        # 添加键量化器参数
        res["key_quantizer"] = self.key_quantizer.params
        
        # 添加值量化器参数
        res["value_quantizer"] = self.value_quantizer.params
        
        # 返回参数字典
        return res

    # 计算两个张量之间的误差
    def _calc_tensor_error(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        # 计算均方误差并返回标量值
        return ((tensor1.to(self.device) - tensor2.to(self.device)) ** 2).mean().item()

    # 计算注意力张量之间的误差
    def _calc_attention_error(self, attention1: AttentionType, attention2: AttentionType) -> float:
        # 计算每对注意力张量的误差并求平均
        return sum(self._calc_tensor_error(attn1, attn2) for attn1, attn2 in zip(attention1, attention2)) / len(attention1)

    # 评估单个问题；核心代码
    def _evaluate_single(self, model: CausalLM, question: Question) -> EvaluationResult:
        # 获取问题长度
        question_len = question.question_length
        
        # 量化前前向传播
        # 将输入ID移到指定设备
        input_ids = question.input_ids.to(self.device)
        
        # 执行前向传播，返回注意力权重和KV缓存
        result = model.forward(input_ids, use_cache=True, output_attentions=True, return_dict=True)
        
        # 量化键/值缓存
        # 提取问题部分的注意力权重
        question_attentions = [attn[:,:,:question_len,:question_len].to(self.device) for attn in result.attentions]
        
        # 提取问题部分的键缓存
        key_cache = torch.stack([key[:,:,:question_len,:].to(self.device) for key, _ in result.past_key_values])
        
        # 提取问题部分的值缓存
        value_cache = torch.stack([value[:,:,:question_len,:].to(self.device) for _, value in result.past_key_values])
        
        # 对键缓存进行量化
        quantized_key_cache, key_average_n_bits = self.key_quantizer.quantize(key_cache, question_attentions)
        
        # 对值缓存进行量化
        quantized_value_cache, value_average_n_bits = self.value_quantizer.quantize(value_cache, question_attentions)
        
        # 构建量化后的KV缓存
        quantized_kvcache = [
            (key.to(result.past_key_values[idx][0].device), value.to(result.past_key_values[idx][0].device))
            for idx, (key, value) in enumerate(zip(quantized_key_cache, quantized_value_cache))
        ]
        
        # 量化后前向传播
        # 对问题答案部分进行前向传播，使用量化后的KV缓存
        quantized_result = model.forward(input_ids[:,question_len:], past_key_values=quantized_kvcache, use_cache=True, output_attentions=True, return_dict=True)
        
        # 计算对数概率
        # 计算第一个词的对数softmax
        first_word_log_softmax = F.log_softmax(result.logits[:,question_len-1], dim=-1)
        
        # 计算量化结果的对数softmax
        quantized_log_softmax = F.log_softmax(quantized_result.logits, dim=-1)
        
        # 初始化变量
        max_log_probability, max_choice_idx, answer_log_probability = None, None, None
        
        # 遍历所有选项
        for choice_idx, choice_len in enumerate(question.choice_length):
            # 计算量化后的对数概率
            quantized_log_probability = first_word_log_softmax[choice_idx, input_ids[choice_idx, question_len]].item()
            quantized_log_probability += quantized_log_softmax[choice_idx, torch.arange(choice_len-1), input_ids[choice_idx,question_len+1:question_len+choice_len]].sum().item()
            quantized_log_probability /= choice_len
            
            # 如果是正确答案，保存答案对数概率
            if choice_idx == question.answer_idx:
                answer_log_probability = quantized_log_probability
                
            # 更新最大对数概率和对应的选项索引
            if max_log_probability is None or quantized_log_probability > max_log_probability:
                max_log_probability = quantized_log_probability
                max_choice_idx = choice_idx
                
        # 计算量化指标
        # 计算键-量化误差
        key_quantization_error = self._calc_tensor_error(key_cache, quantized_key_cache)
        
        # 计算值-量化误差
        value_quantization_error = self._calc_tensor_error(value_cache, quantized_value_cache)
        
        # 计算注意力-误差
        attention_error = self._calc_attention_error(
            [attn[:,:,question_len:,:question_len].to(self.device) for attn in result.attentions],
            [attn[:,:,:,:question_len].to(self.device) for attn in quantized_result.attentions],
        )
        
        # 计算逻辑-误差
        logit_error = self._calc_tensor_error(result.logits[:,question_len:,:], quantized_result.logits)
        
        # 计算键平均大小
        key_average_size = self.key_quantizer.calc_quantized_cache_size_per_token(key_average_n_bits, model)
        
        # 计算值平均大小
        value_average_size = self.value_quantizer.calc_quantized_cache_size_per_token(value_average_n_bits, model)
        
        # 返回评估结果
        return EvaluationResult(
            accuracy=1.0 if max_choice_idx == question.answer_idx else 0.0,
            answer_log_probability=answer_log_probability,
            quantization_error=(key_quantization_error + value_quantization_error) / 2,
            key_quantization_error=key_quantization_error,
            value_quantization_error=value_quantization_error,
            attention_error=attention_error,
            logit_error=logit_error,
            average_size=(key_average_size + value_average_size) / 2,
            key_average_size=key_average_size,
            value_average_size=value_average_size,
            average_n_bits=(key_average_n_bits + value_average_n_bits) / 2,
            key_average_n_bits=key_average_n_bits,
            value_average_n_bits=value_average_n_bits,
        )

    # 执行评估
    def evaluate(self, model: CausalLM, use_tqdm: bool) -> EvaluationResult:
        # 验证模型名称匹配
        assert model.name_or_path == self.model_name
        
        # 初始化结果对象
        result = EvaluationResult()
        
        # 初始化总令牌数
        total_tokens = 0
        
        # 禁用梯度计算
        with torch.no_grad():
            # 遍历所有问题
            for idx, question in enumerate(tqdm(self.datasets.questions) if use_tqdm else self.datasets.questions):
                # 评估单个问题
                single_result = self._evaluate_single(model, question)
                
                # 获取问题长度
                n_tokens = question.question_length
                
                # 累加总令牌数
                total_tokens += n_tokens
                
                # 累加各项指标
                result.accuracy += single_result.accuracy
                result.answer_log_probability += single_result.answer_log_probability
                result.quantization_error += single_result.quantization_error
                result.key_quantization_error += single_result.key_quantization_error
                result.value_quantization_error += single_result.value_quantization_error
                result.attention_error += single_result.attention_error
                result.logit_error += single_result.logit_error
                result.average_size += single_result.average_size * n_tokens
                result.key_average_size += single_result.key_average_size * n_tokens
                result.value_average_size += single_result.value_average_size * n_tokens
                result.average_n_bits += single_result.average_n_bits * n_tokens
                result.key_average_n_bits += single_result.key_average_n_bits * n_tokens
                result.value_average_n_bits += single_result.value_average_n_bits * n_tokens
                
                # 每100个问题执行一次垃圾回收
                if (idx + 1) % 100 == 0:
                    gc.collect()
                    
        # 计算平均准确率
        result.accuracy /= self.datasets.question_count
        
        # 计算95%置信区间
        result.accuracy_confidence = 1.96 * math.sqrt(result.accuracy * (1.0 - result.accuracy) / self.datasets.question_count)
        
        # 计算各项指标的平均值
        result.answer_log_probability /= self.datasets.question_count
        result.quantization_error /= self.datasets.question_count
        result.key_quantization_error /= self.datasets.question_count
        result.value_quantization_error /= self.datasets.question_count
        result.attention_error /= self.datasets.question_count
        result.logit_error /= self.datasets.question_count
        result.average_size /= total_tokens
        result.key_average_size /= total_tokens
        result.value_average_size /= total_tokens
        result.average_n_bits /= total_tokens
        result.key_average_n_bits /= total_tokens
        result.value_average_n_bits /= total_tokens
        
        # 返回最终结果
        return result
    
    # 从缓存获取结果
    def get_cached_result(self, cache_file: Optional[str]) -> Optional[EvaluationResult]:
        # 如果缓存文件不存在，返回None
        if cache_file is None or not os.path.exists(cache_file):
            return None
            
        # 读取缓存文件
        with open(cache_file, "r") as f:
            cached_results = json.load(f)
            
            # 查找匹配的参数
            for entry in cached_results:
                if entry["params"] == self.params:
                    # 返回缓存的评估结果
                    return EvaluationResult(**entry["results"])
                    
        # 未找到匹配结果，返回None
        return None

    # 缓存结果
    def cache_result(self, cache_file: Optional[str], result: EvaluationResult):
        # 如果缓存文件为None，直接返回
        if cache_file is None:
            return
            
        # 如果缓存文件存在，读取现有内容
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached_results = json.load(f)
        else:
            # 否则创建空列表
            cached_results = []
            
        # 添加新结果
        cached_results.append({
            "params": self.params,
            "results": asdict(result),
        })
        
        # 写入缓存文件
        with open(cache_file, "w") as f:
            json.dump(cached_results, f, indent=4, separators=(", ", ": "))