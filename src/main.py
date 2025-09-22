# 导入操作系统模块，用于环境变量设置
import os

# 导入PyTorch库，用于张量计算和数据类型定义
import torch

# 导入实验模块，包含各种实验类
import experiments as exp

# 设置环境变量，启用tokenizers的并行处理
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # CUDA调试模式（已注释）
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 定义模型名称，指定要使用的预训练模型路径
# 当前使用本地缓存的Llama-2-7b-hf模型
model_name="/root/.cache/modelscope/hub/models/shakechen/Llama-2-7b-hf"

# 其他可选的模型名称（已注释）
# model_name="shakechen/Llama-2-7b-hf"  # HuggingFace模型标识符
# model_name = "meta-llama/Llama-2-7b-hf"  # Meta官方Llama-2-7b模型
# model_name = "meta-llama/Llama-2-13b-hf"  # Meta官方Llama-2-13b模型
# model_name = "meta-llama/Llama-2-70b-hf"  # Meta官方Llama-2-70b模型
# model_name = "facebook/opt-125m"  # Facebook OPT-125M模型
# model_name = "facebook/opt-350m"  # Facebook OPT-350M模型
# model_name = "facebook/opt-2.7b"  # Facebook OPT-2.7B模型
# model_name = "facebook/opt-6.7b"  # Facebook OPT-6.7B模型
# model_name = "facebook/opt-13b"  # Facebook OPT-13B模型
# model_name = "facebook/opt-30b"  # Facebook OPT-30B模型
# model_name = "facebook/opt-66b"  # Facebook OPT-66B模型

# 定义数据集名称，指定要使用的问答数据集
# 当前使用HellaSwag数据集
dataset_name = "Rowan/hellaswag"

# 其他可选的数据集名称（已注释）
# dataset_name = "math_qa"  # MathQA数据集
# dataset_name = "piqa"  # PIQA数据集
# dataset_name = "truthful_qa"  # TruthfulQA数据集

# 定义数据类型，指定模型计算时使用的精度
# 使用半精度浮点数（float16）以节省内存
dtype = torch.float16

# 定义问题数量，控制实验规模
# 设置为1000个问题进行评估
question_count = 10


# 程序入口点
if __name__ == "__main__":
    # 运行网格搜索实验（已注释）
    # exp.GridSearch(model_name, dataset_name, dtype, question_count, parallel=True, verbose=True).run()

    # 运行键值差异实验（已注释）
    exp.KeyValueDifference(model_name, dataset_name, dtype, question_count, parallel=True, verbose=True).run()

    # 运行KV缓存分布实验（已注释）
    # exp.KVcacheDistribution(model_name, dataset_name, dtype, question_count, parallel=True, verbose=True).run()

    # 运行注意力洞察实验（已注释）
    # exp.AttentionInsight(model_name, dataset_name, dtype, question_count, parallel=True, verbose=True).run()

    # 运行测试实验（当前启用）
    # exp.Test(model_name, dataset_name, dtype, question_count, parallel=True, verbose=True).run()