import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "/home/huawei/huawei/Qwen3-0.6B-GPTQ-Int8"

logging.getLogger("httpx").setLevel(logging.WARNING)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

prompt = "介绍一下量化模型的原理。请用中文详细说明，并给出示例、使用方法和评估方法。尽量不要使用英文字母。"

messages = [{"role": "user", "content": prompt}]
try:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
except Exception:
    text = prompt

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())