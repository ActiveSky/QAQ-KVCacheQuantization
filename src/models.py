# 导入类型提示模块，用于类型注解
from typing import Union

# 导入HuggingFace Transformers库中的模型和分词器类
# LlamaForCausalLM: Llama模型的因果语言建模实现
# OPTForCausalLM: OPT模型的因果语言建模实现
# LlamaTokenizerFast: Llama模型的快速分词器
# GPT2TokenizerFast: GPT-2模型的快速分词器
from transformers import LlamaForCausalLM, OPTForCausalLM, LlamaTokenizerFast, GPT2TokenizerFast

# 定义因果语言模型类型别名
# 支持Llama和OPT两种因果语言模型
CausalLM = Union[LlamaForCausalLM, OPTForCausalLM]

# 定义分词器类型别名
# 支持Llama和GPT-2两种快速分词器
Tokenizer = Union[LlamaTokenizerFast, GPT2TokenizerFast]