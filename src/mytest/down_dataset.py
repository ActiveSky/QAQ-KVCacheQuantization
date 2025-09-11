import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset


hf_cache_dir=None
dataset_name="Rowan/hellaswag"
dataset=load_dataset(dataset_name, split="validation")


print(dataset[0:10])