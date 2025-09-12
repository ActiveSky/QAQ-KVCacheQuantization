# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the official implementation of QAQ (Quality Adaptive Quantization) for LLM KV Cache, a method for compressing the Key-Value cache in large language models with minimal accuracy loss. The project implements various quantization strategies and evaluation frameworks for analyzing their impact on model performance.

## Repository Structure

- `src/`: Main source code
  - `config.py`: Configuration settings including device configurations and cache paths
  - `quantizer.py`: Core quantization implementation with various quantization strategies
  - `evaluator.py`: Evaluation framework for measuring quantization impact
  - `qa_dataset.py`: Dataset handling for question-answering tasks
  - `models.py`: Model type definitions
  - `experiments/`: Experiment implementations including various quantization studies
  - `main.py`: Entry point for running experiments

## Key Components

### 1. Quantizer (`src/quantizer.py`)
The core quantization component that supports:
- Different quantization levels (token, layer, head)
- Multiple quantization methods (uniform, normal)
- Attention-aware quantization with adaptive bit allocation
- Outlier handling

### 2. Evaluator (`src/evaluator.py`)
Evaluates quantization performance on question-answering datasets, measuring:
- Accuracy impact
- Quantization error metrics
- Cache size reduction
- Attention and logit errors

### 3. Experiment Framework (`src/experiments/base.py`)
Abstract base class for running quantization experiments with:
- Parallel execution support
- Result caching
- Flexible quantizer configuration

## Development Commands

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running Experiments
```bash
# Run the main entry point
python src/main.py
```

The main entry point runs a Test experiment by default. Other experiments can be enabled by uncommenting the relevant lines in `src/main.py`.

### Configuration
- Modify `src/config.py` to change device configurations for multi-GPU setups
- Adjust model paths and dataset settings in `src/main.py`
- Experiment-specific parameters are defined in the respective experiment files under `src/experiments/`

## Architecture Overview

The system follows a modular architecture:
1. **Configuration Layer**: Defines model, dataset, and device settings
2. **Data Layer**: Handles dataset loading and preprocessing (`qa_dataset.py`)
3. **Quantization Layer**: Implements various quantization strategies (`quantizer.py`)
4. **Evaluation Layer**: Measures quantization impact (`evaluator.py`)
5. **Experiment Layer**: Orchestrates experiments and result processing (`experiments/`)

Experiments are designed to be run in parallel across multiple GPUs, with results automatically cached to avoid recomputation.