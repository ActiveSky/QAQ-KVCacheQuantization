# 导入正则表达式模块，用于文本预处理
import re

# 导入PyTorch库，用于张量操作
import torch

# 导入随机模块，用于数据采样
import random

# 导入自定义模型模块，用于类型注解
from models import Tokenizer

# 导入配置模块，获取HuggingFace缓存目录
from config import hf_cache_dir

# 导入HuggingFace数据集库，用于加载数据集
from datasets import load_dataset

# 导入dataclasses模块，用于创建数据类
from dataclasses import dataclass

# 导入functools模块的cached_property装饰器，用于缓存属性值
from functools import cached_property


# 定义问题数据类，包含问题的相关信息
@dataclass
class Question:
    # 输入ID张量，包含问题和选项的tokenized形式
    input_ids: torch.Tensor
    
    # 问题长度，即问题部分的token数量
    question_length: int
    
    # 选项长度列表，每个选项的token数量
    choice_length: list[int]
    
    # 原始问题文本
    question: str
    
    # 选项文本列表
    choices: list[str]
    
    # 正确答案索引
    answer_idx: int


# QA数据集类，用于加载和处理问答数据集
class QADataset:
    # 初始化数据集
    def __init__(self, dataset_name: str, tokenizer: Tokenizer, question_count: int):
        # 数据集名称
        self.dataset_name = dataset_name
        
        # 分词器对象
        self.tokenizer = tokenizer
        
        # 问题数量，控制数据集大小
        self.question_count = question_count

    # 缓存属性：问题列表，根据数据集名称加载相应的问题
    @cached_property
    def questions(self) -> list[Question]:
        # 根据数据集名称加载验证集数据
        if self.dataset_name in ["Rowan/hellaswag", "math_qa", "piqa"]:
            # 加载HellaSwag、MathQA、PIQA数据集
            raw_dataset: list[dict[str, str]] = list(load_dataset(self.dataset_name, split="validation", cache_dir=hf_cache_dir))
        elif self.dataset_name == "truthful_qa":
            # 加载TruthfulQA数据集的多项选择部分
            raw_dataset: list[dict[str, str]] = list(load_dataset("truthful_qa", "multiple_choice", split="validation", cache_dir=hf_cache_dir))
            
        # 设置随机种子以确保结果可重现
        random.seed(42)
        
        # 随机采样指定数量的问题
        raw_dataset = random.sample(raw_dataset, k=self.question_count)
        
        # 根据数据集名称调用相应的加载方法
        if self.dataset_name == "Rowan/hellaswag":
            return self._load_hellaswag(raw_dataset)
        if self.dataset_name == "math_qa":
            return self._load_mathqa(raw_dataset)
        if self.dataset_name == "piqa":
            return self._load_piqa(raw_dataset)
        if self.dataset_name == "truthful_qa":
            return self._load_truthfulqa(raw_dataset)

    # 构建问题对象
    def _build_question(self, question: str, choices: list[str], answer_idx: int) -> Question:
        # 计算问题长度（不包括选项）
        question_len = self.tokenizer(question, return_tensors="pt", add_special_tokens=False, return_attention_mask=False).input_ids.shape[1]
        
        # 构建问题+选项的组合文本
        question_choices = [question + " " + choice for choice in choices]
        
        # 对组合文本进行tokenize和padding
        results = self.tokenizer(question_choices, return_tensors="pt", padding=True, add_special_tokens=False, return_attention_mask=True)
        
        # 计算每个选项的长度（包括问题部分）
        choices_len = (results.attention_mask.sum(dim=1) - question_len).tolist()
        
        # 返回问题对象
        return Question(
            input_ids=results.input_ids,
            question_length=question_len,
            choice_length=choices_len,
            question=question,
            choices=choices,
            answer_idx=answer_idx,
        )

    # 加载HellaSwag数据集
    def _load_hellaswag(self, raw_dataset: list[dict[str, str]]) -> list[Question]:
        # 编译正则表达式，用于移除方括号内容
        pattern = re.compile(r"\[.*?\]")
        
        # 文本预处理函数
        def preprocess_text(text: str) -> str:
            # 去除首尾空格
            text = text.strip()
            
            # 替换标题标记
            text = text.replace(" [title]", ". ")
            
            # 移除方括号内容
            text = re.sub(pattern, "", text)
            
            # 替换双重空格为单空格
            text = text.replace("  ", " ")
            
            return text
            
        # 初始化问题列表
        questions: list[Question] = []
        
        # 遍历原始数据集
        for data in raw_dataset:
            # 构建问题文本
            question = preprocess_text(data["activity_label"] + ": " + data["ctx_a"] + " " + data["ctx_b"].capitalize())
            
            # 预处理选项文本
            choices = [preprocess_text(choice) for choice in data["endings"]]
            
            # 获取正确答案索引
            answer_idx = int(data["label"])
            
            # 构建问题对象并添加到列表
            questions.append(self._build_question(question, choices, answer_idx))
            
        # 返回问题列表
        return questions

    # 加载MathQA数据集
    def _load_mathqa(self, raw_dataset: list[dict[str, str]]) -> list[Question]:
        # 编译正则表达式，用于提取选项
        pattern = re.compile(r"[abcd] \) .*?, |e \) .*?$")
        
        # 初始化问题列表
        questions: list[Question] = []
        
        # 遍历原始数据集
        for data in raw_dataset:
            # 构建问题文本
            question = f"Question: {data['Problem']}\nAnswer:"
            
            # 提取选项文本
            choices = [c[4:].rstrip(" ,") for c in re.findall(pattern, data["options"])]
            
            # 获取正确答案索引
            answer_idx = ["a", "b", "c", "d", "e"].index(data["correct"])
            
            # 构建问题对象并添加到列表
            questions.append(self._build_question(question, choices, answer_idx))
            
        # 返回问题列表
        return questions

    # 加载PIQA数据集
    def _load_piqa(self, raw_dataset: list[dict[str, str]]) -> list[Question]:
        # 初始化问题列表
        questions: list[Question] = []
        
        # 遍历原始数据集
        for data in raw_dataset:
            # 构建问题文本
            question = f"Question: {data['goal']}\nAnswer:"
            
            # 获取选项文本
            choices = [data["sol1"], data["sol2"]]
            
            # 获取正确答案索引
            answer_idx = int(data["label"])
            
            # 构建问题对象并添加到列表
            questions.append(self._build_question(question, choices, answer_idx))
            
        # 返回问题列表
        return questions

    # 加载TruthfulQA数据集
    def _load_truthfulqa(self, raw_dataset: list[dict[str, str]]) -> list[Question]:
        # 初始化问题列表
        questions: list[Question] = []
        
        # 遍历原始数据集
        for data in raw_dataset:
            # 构建问题文本
            question = f"Q: {data['question']}\nA:"
            
            # 获取选项文本
            choices = data["mc1_targets"]["choices"]
            
            # 设置正确答案索引为0（TruthfulQA的格式）
            answer_idx = 0
            
            # 构建问题对象并添加到列表
            questions.append(self._build_question(question, choices, answer_idx))
            
        # 返回问题列表
        return questions