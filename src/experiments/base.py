# =============================================================================
# EXPERIMENT FRAMEWORK - 实验框架基类
# =============================================================================

"""
实验框架架构说明：
===================

1. 设计模式：
   - 抽象基类模式：定义实验的统一接口
   - 模板方法模式：提供完整的实验执行流程
   - 策略模式：子类可自定义量化器列表和结果处理

2. 核心组件：
   - 资源管理：tokenizer、model、datasets的缓存管理
   - 执行引擎：支持串行和并行执行
   - 结果处理：抽象的结果处理接口

3. 执行流程：
   run() → 并行/串行执行 → 单次评估 → 结果收集 → process_result()

4. 依赖关系：
   - 强依赖：quantizer_list → run() → _run_single_evaluation()
   - 资源链：datasets → tokenizer → AutoTokenizer
   - 模型链：get_model() → device_configs → infer_auto_device_map()

5. 并发机制：
   - 多进程：基于multiprocessing.Process
   - 任务分配：基于Queue的任务队列
   - 线程安全：基于Lock的文件访问保护
"""

import abc
import torch
from dataclasses import asdict
from quantizer import Quantizer
from qa_dataset import QADataset
from models import CausalLM, Tokenizer
from functools import cached_property, cache
from evaluator import Evaluator, EvaluationResult
from multiprocessing import queues, Queue, Lock, Process
from accelerate import init_empty_weights, infer_auto_device_map
# from config import version, cache_file, hf_cache_dir, device_configs
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
#加载量化配置
from transformers import BitsAndBytesConfig

import queue
import config as cfg

class Experiment(abc.ABC):
    def __init__(self, model_name: str, dataset_name: str, dtype: torch.dtype, question_count: int, parallel: bool, verbose: bool):
        """
        初始化实验配置
        
        参数说明：
        - model_name: 模型名称（如 "meta-llama/Llama-2-7b-hf"）
        - dataset_name: 数据集名称（如 "Rowan/hellaswag"）
        - dtype: 数据类型（如 torch.float16）
        - question_count: 问题数量（控制实验规模）
        - parallel: 是否启用并行执行
        - verbose: 是否显示详细信息
        
        并行判断逻辑：
        - 必须同时满足：parallel=True、quantizer_list>1、device_configs>1
        - 确保在多设备环境下才启用并行执行
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dtype = dtype
        self.question_count = question_count
        self.verbose = verbose
        # 动态判断是否启用并行执行
        self.parallel = parallel and len(self.quantizer_list) > 1 and len(cfg.device_configs) > 1

    @cached_property
    def tokenizer(self) -> Tokenizer:
        """
        分词器资源管理
        
        设计要点：
        - @cached_property: 确保只加载一次，避免重复I/O
        - 从HuggingFace Hub加载预训练分词器
        - 设置pad_token_id=0，统一填充符号
        - 依赖hf_cache_dir进行本地缓存
        
        依赖关系：
        - 外部：AutoTokenizer.from_pretrained(), hf_cache_dir
        - 内部：被self.datasets()依赖
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cfg.hf_cache_dir)
        tokenizer.pad_token_id = 0
        return tokenizer

    @cache
    def get_model(self, worker_id: int) -> CausalLM:
        """
        模型加载和设备映射
        
        设计要点：
        - @cache装饰器：按worker_id缓存，每个工作进程独立模型实例
        - init_empty_weights(): 空权重初始化，减少内存占用
        - infer_auto_device_map(): 自动设备映射，支持多GPU和CPU卸载
        - tie_weights(): 绑定权重，减少参数数量
        
        内存优化：
        - 支持CPU卸载（当device_map包含cpu或disk时发出警告）
        - 根据device_configs[worker_id]进行设备分配
        
        依赖关系：
        - 外部：accelerate库的设备映射功能
        - 配置：device_configs[worker_id]定义设备和内存限制
        
        worker_id作用：
        - 在并行执行中标识不同的工作进程
        - 每个worker_id对应不同的设备配置
        """
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name, cache_dir=cfg.hf_cache_dir))
        _, max_memory = cfg.device_configs[worker_id]
        model.tie_weights()
        device_map = infer_auto_device_map(model, max_memory=max_memory, dtype=self.dtype, no_split_module_classes=model._no_split_modules)
        if any(x == "cpu" or x == "disk" for x in device_map.values()):
            print("Warning: CPU offloading enabled!")
            
        # 设置4位量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 启用4位量化
            bnb_4bit_compute_dtype=torch.float16,  # 计算时使用float16
            # bnb_4bit_quant_type="nf4",  # 使用正态浮点数4位量化
            bnb_4bit_use_double_quant=True,  # 启用双重量化以进一步减少内存使用
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map=device_map, 
            torch_dtype=self.dtype, 
            cache_dir=cfg.hf_cache_dir,
            # quantization_config=quantization_config
            ).eval()
        return model

    @cached_property
    def datasets(self) -> QADataset:
        """
        数据集创建和管理
        
        设计要点：
        - @cached_property: 确保数据集只创建一次
        - 依赖self.tokenizer，确保分词器已初始化
        - 传递问题数量，控制实验规模
        
        依赖关系：
        - 强依赖：self.tokenizer（内部依赖链）
        - 外部：QADataset类负责具体的数据加载
        
        数据流：
        datasets → tokenizer → AutoTokenizer.from_pretrained()
        """
        return QADataset(self.dataset_name, self.tokenizer, self.question_count)

    @cached_property
    def quantizer_list(self) -> list[tuple[Quantizer, Quantizer]]:
        """
        量化器组合列表（抽象属性）
        
        设计要点：
        - 抽象属性：子类必须重写此方法
        - 返回格式：[(key_quantizer, value_quantizer), ...]
        - 定义实验要测试的量化器组合
        
        子类实现示例：
        - Test: 返回no-quantization的量化器对
        - GridSearch: 返回大量参数组合的量化器对
        - KeyValueDifference: 返回对比测试的量化器对
        
        在实验执行中的作用：
        - run()遍历此列表执行所有量化器组合
        - 并行执行时，列表元素作为任务分配给不同worker
        - 结果处理时，与结果一一对应
        
        依赖关系：
        - 被run()和__init__()的并行判断依赖
        - 是整个实验的核心输入
        """
        return []

    @abc.abstractmethod
    def process_result(self, results: list[EvaluationResult]):
        """
        结果处理方法（抽象方法）
        
        设计要点：
        - 抽象方法：子类必须实现具体的结果处理逻辑
        - 输入：完整的评估结果列表，与quantizer_list一一对应
        - 用途：可视化、分析、保存结果等
        
        子类实现示例：
        - Test: 空实现（仅验证系统）
        - GridSearch: 生成大量可视化图表
        - KeyValueDifference: 生成对比曲线图
        - KVcacheDistribution: 分析KV缓存分布
        - AttentionInsight: 生成注意力热力图
        
        执行时机：
        - 在所有评估任务完成后调用
        - 所有结果已从缓存中收集完毕
        
        依赖关系：
        - 输入：EvaluationResult列表
        - 内部：被run()在最后阶段调用
        
        设计模式：
        - 模板方法模式：定义算法骨架，子类实现具体步骤
        - 策略模式：不同的结果处理策略
        """
        pass

    def _run_single_evaluation(self, worker_id: int, task_queue: Queue, file_lock: Lock):
        """
        单次评估执行（核心执行单元）
        
        设计要点：
        - 工作单元：处理一个量化器组合的完整评估
        - 任务获取：从队列获取任务，支持timeout异常处理
        - 设备配置：为量化器设置数据类型和设备
        - 缓存机制：优先使用缓存，避免重复计算
        
        执行流程：
        1. 任务获取 → 从队列获取(idx, key_quantizer, value_quantizer)
        2. 设备配置 → 调用quantizer.set_dtype_and_device()
        3. 评估器创建 → 创建Evaluator实例
        4. 缓存检查 → 调用evaluator.get_cached_result()
        5. 评估执行 → 如无缓存，执行evaluator.evaluate()
        6. 结果缓存 → 保存结果到缓存文件
        7. 信息输出 → verbose模式下显示详细信息
        
        线程安全：
        - 使用file_lock保护缓存文件读写操作
        - 确保多进程并发时缓存数据一致性
        
        依赖关系：
        - 内部：self.get_model()（模型加载）
        - 外部：Evaluator类（评估执行）、Quantizer类（量化处理）
        
        worker_id作用：
        - 标识工作进程，用于设备分配
        - 传入get_model()获取对应设备的模型实例
        """
        
        
        task_result: tuple[int, Quantizer, Quantizer] = task_queue.get(timeout=1)
        idx, key_quantizer, value_quantizer = task_result

        print(f"Running evaluation #{idx+1} on worker #{worker_id+1}...")
        device, _ = cfg.device_configs[worker_id]
        key_quantizer.set_dtype_and_device(self.dtype, device)
        value_quantizer.set_dtype_and_device(self.dtype, device)
        evaluator = Evaluator(device, cfg.version, self.model_name, self.datasets, key_quantizer, value_quantizer)
        with file_lock:
            result = evaluator.get_cached_result(cfg.cache_file)
        if result is None:
            model = self.get_model(worker_id)
            result = evaluator.evaluate(model, use_tqdm=True)
            with file_lock:
                evaluator.cache_result(cfg.cache_file, result)
        if self.verbose:
            print(f"  Params: {evaluator.params}")
            print(f"  Results: {asdict(result)}")
            print("======================================")

    def run(self):
        """
        实验主执行入口（核心控制函数）
        
        设计要点：
        - 统一入口：所有实验的执行都从此开始
        - 双模式执行：支持并行和串行两种执行模式
        - 资源预热：并行执行前确保关键资源已加载
        - 结果收集：从缓存中统一收集所有结果
        
        执行流程：
        =================================================================
        Phase 1: 任务准备
            - 创建文件锁（保护缓存文件）
            - 创建任务队列（分配评估任务）
            - 将quantizer_list中的所有量化器组合放入队列
        
        Phase 2: 执行模式选择
            并行执行（多进程）:
            - 资源预热：确保tokenizer和datasets已加载
            - 进程创建：为每个device_config创建一个worker进程
            - 进程启动：所有worker并发执行任务
            - 进程等待：等待所有worker完成
            
            串行执行（单进程）:
            - 直接调用worker(0)在主进程中执行
        
        Phase 3: 结果收集和处理
            - 创建空的results列表
            - 遍历quantizer_list，从缓存中读取结果
            - 调用process_result()进行结果处理（子类实现）
        =================================================================
        
        并发机制详解：
        - 进程模型：使用multiprocessing.Process创建独立进程
        - 任务分配：基于Queue的生产者-消费者模式
        - 异常处理：queues.Empty异常标识任务完成
        - 线程安全：Lock保护缓存文件的并发访问
        
        关键优化：
        - 资源预热：并行执行前预先加载共享资源
        - 缓存机制：避免重复计算，提高执行效率
        - 动态判断：根据实际情况选择最优执行模式
        
        依赖关系：
        - 核心依赖：self.quantizer_list（任务源）
        - 内部依赖：self._run_single_evaluation()（执行逻辑）
        - 资源依赖：self.datasets, self.tokenizer（预热资源）
        - 外部依赖：multiprocessing, Queue, Lock（并发原语）
        
        worker函数设计：
        - 内嵌函数：访问外部作用域的task_queue和file_lock
        - 循环执行：持续从队列获取任务直到队列为空
        - 异常处理：捕获queues.Empty异常退出循环
        """
        # Phase 1: 任务准备
        file_lock = Lock()  # 文件锁，保护缓存文件并发访问
        task_queue = Queue()  # 任务队列，分配评估任务
        
        # 将所有量化器组合放入队列
        for idx, (key_quantizer, value_quantizer) in enumerate(self.quantizer_list):
            task_queue.put((idx, key_quantizer, value_quantizer))
        
        # Worker函数定义（内嵌函数，共享作用域）
        def worker(worker_id: int):
            """
            Worker进程的执行函数
            
            设计：
            - 循环执行：持续从队列获取任务
            - 异常退出：捕获queues.Empty异常表示任务完成
            - 任务执行：调用_run_single_evaluation执行具体评估
            """
            while True:
                try:
                    self._run_single_evaluation(worker_id, task_queue, file_lock)
                except queue.Empty:
                    break  # 队列为空，任务完成

        # Phase 2: 执行模式选择
        if self.parallel:
            # 并行执行模式
            # 资源预热：确保关键资源已加载，避免并行时重复加载
            _, _ = self.datasets.questions, self.tokenizer
            
            # 创建多进程worker
            process_list: list[Process] = []
            for worker_id in range(len(cfg.device_configs)):
                process = Process(target=worker, args=(worker_id,))
                process_list.append(process)
                process.start()
            
            # 等待所有进程完成
            for process in process_list:
                process.join()
        else:
            # 串行执行模式
            worker(0)
        
        # Phase 3: 结果收集和处理
        results: list[EvaluationResult] = []
        for key_quantizer, value_quantizer in self.quantizer_list:
            # 创建评估器（在CPU上，仅用于缓存读取）
            evaluator = Evaluator(torch.device("cpu"), cfg.version, self.model_name, self.datasets, key_quantizer, value_quantizer)
            # 从缓存中读取结果
            result = evaluator.get_cached_result(cfg.cache_file)
            assert result is not None, f"Result not found for {evaluator.params}"
            results.append(result)
        
        # 调用子类实现的结果处理方法
        self.process_result(results)
