from transformers import  AutoModelForCausalLM

model_dir="/root/.cache/modelscope/hub/models/shakechen/Llama-2-7b-hf"
model_name="Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             trust_remote_code=True,
                                             cache_dir=model_dir)
print("model loaded successfully")