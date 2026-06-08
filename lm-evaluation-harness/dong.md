# humaneval
## vllm
cd /home/huawei/huawei/lm-evaluation-harness
HF_ALLOW_CODE_EVAL=1 python -m lm_eval \
  --model vllm \
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-W8A8-Dynamic-Per-Token,dtype=auto,gpu_memory_utilization=0.35,max_model_len=1024 \
  --tasks humaneval \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --output_path outputs/Qwen3-0.6B-humaneval_vllm \
  --log_samples
## onnx
HF_ALLOW_CODE_EVAL=1 python -m lm_eval \
  --model onnx \
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0522-w8a8-llm/result/qwen3_0_6b_fake_quant_model_merged.onnx,provider=CUDAExecutionProvider,use_amct=true \
  --tasks humaneval \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --output_path outputs/Qwen3-0.6B_onnx_humaneval \
  --log_samples

HF_ALLOW_CODE_EVAL=1 python -m lm_eval \
  --model onnx \
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-onnx-ai-onnx-opset16-cache/model.onnx,tokenizer=/home/huawei/huawei/Qwen3-0.6B,provider=CUDAExecutionProvider \
  --tasks humaneval \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --output_path outputs/Qwen3-0.6B_amct_fakequant_humaneval \
  --log_samples
## hf
```bash
cd /home/huawei/huawei/lm-evaluation-harness

HF_ALLOW_CODE_EVAL=1 python -m lm_eval \
  --model hf \
  --model_args pretrained=pretrained=/home/zilai/yuheng/huawei/Qwen3-0.6B \
  --tasks humaneval \
  --device cuda:2 \
  --batch_size 1 \
  --gen_kwargs "do_sample=False,temperature=0,top_p=1" \
  --confirm_run_unsafe_code \
  --output_path outputs/Qwen3-0.6B_humaneval \
  --log_samples

HF_ALLOW_CODE_EVAL=1 python -m lm_eval \
  --model hf \
    --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-MXFP8,tokenizer=/home/huawei/huawei/Qwen3-0.6B \
  --tasks humaneval \
  --device cuda \
  --batch_size 1 \
  --gen_kwargs "do_sample=False,temperature=0,top_p=1" \
  --confirm_run_unsafe_code \
  --output_path outputs/Qwen3-0.6B-MXFP8\
  --log_samples
```



# mbpp
## vllm 
HF_ALLOW_CODE_EVAL=1 python -m lm_eval \
  --model vllm \
  --model_args pretrained=/home/huawei/huawei/Qwen3-0.6B-GPTQ-Int8,gpu_memory_utilization=0.35,max_model_len=2048 \
  --tasks mbpp \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --output_path outputs/Qwen3-0.6B-mbpp_vllm \
  --log_samples
  

## onnx
```bash
HF_ALLOW_CODE_EVAL=1 python -m lm_eval \
  --model onnx \
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0522-w8a8-llm/result/qwen3_0_6b_fake_quant_model_merged.onnx,provider=CUDAExecutionProvider,use_amct=true \
  --tasks mbpp \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --output_path outputs/Qwen3-0.6B_onnx_mbpp \
  --log_samples \
  --use_cache /home/huawei/huawei/lm-evaluation-harness/cache/mbpp-amct-0524


cd /home/huawei/huawei/lm-evaluation-harness

HF_ALLOW_CODE_EVAL=1 python -m lm_eval \
  --model onnx \
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0509/result/qwen3_0_6b_fake_quant_model_merged.onnx,tokenizer=/home/huawei/huawei/Qwen3-0.6B,provider=CUDAExecutionProvider,use_amct=true \
  --tasks mbpp \
  --confirm_run_unsafe_code \
  --batch_size 1 \
  --output_path outputs/Qwen3-0.6B_amct_fakequant_mbpp \
  --log_samples
```
,fix_mistral_regex=true 
1. qwen系列 mbpp加--apply_chat_template 参数后会影响自动会自动加角色和特殊 token
例如 user/assistant 结构、<|im_start|> 这类控制符（具体取决于模型模板）。

few-shot 示例会更像多轮对话
通常会配合多轮消息格式，而不是直接拼接文本。

对指令模型通常更“像它训练时的输入”
在一些聊天类任务上可能更好。

在代码基准（如 MBPP）上可能带来副作用
模型更容易输出聊天风格内容（解释、代码块围栏、思维痕迹等），而不是“只给可执行函数定义”，从而影响 pass@1。
```bash
HF_ALLOW_CODE_EVAL=1 lm-eval run \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/gemma-3-1b-it\
  --tasks mbpp \
  --confirm_run_unsafe_code \
  --output_path /home/huawei/huawei/lm-evaluation-harness/outputs \
  --log_samples

HF_ALLOW_CODE_EVAL=1 lm-eval run \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/Qwen3-0.6B enable_thinking=false \
  --tasks mbpp \
  --gen_kwargs do_sample=False,temperature=0.6,top_p=0.95,top_k=20 \
  --confirm_run_unsafe_code \
  --output_path /home/huawei/huawei/lm-evaluation-harness/outputs/ \
  --log_samples 
```
# ifeval

  --apply_chat_template \ qwen系列加上这个参数分数更高
## vllm

HF_ALLOW_CODE_EVAL=1 python -m lm_eval \
  --model vllm \
  --model_args pretrained=/home/huawei/huawei/Qwen3-0.6B-GPTQ-Int8,gpu_memory_utilization=0.35,max_model_len=2048 \
  --tasks ifeval \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --output_path outputs/Qwen3-0.6B-ifeval_vllm \
  --log_samples
## onnx
```bash
HF_ALLOW_CODE_EVAL=1 python -m lm_eval \
  --model onnx \
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0522-w8a8-llm/result/qwen3_0_6b_fake_quant_model_merged.onnx,use_amct=true,provider=CUDAExecutionProvider \
  --tasks ifeval \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --apply_chat_template \
  --output_path outputs/Qwen3-0.6B_onnx_ifeval \
  --log_samples \
  --use_cache /home/huawei/huawei/lm-evaluation-harness/cache/ifeval-amct-0526


cd /home/huawei/huawei/lm-evaluation-harness

python -m lm_eval \
  --model onnx \
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0509/result/qwen3_0_6b_fake_quant_model_merged.onnx,tokenizer=/home/huawei/huawei/Qwen3-0.6B,provider=CUDAExecutionProvider,use_amct=true \
  --tasks ifeval \
  --batch_size 1 \
  --output_path outputs/Qwen3-0.6B_amct_fakequant_ifeval \
  --log_samples
```
```bash


cd /home/huawei/huawei/lm-evaluation-harness
python -m lm_eval \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/gemma-3-1b-it \
  --tasks ifeval \
  --device cuda \
  --batch_size 4 \
  --output_path outputs/gemma3-1-ifeval \
  --log_samples

python -m lm_eval \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/Qwen3-0.6B-MXFP8 \
  --tasks ifeval \
  --device cuda \
  --batch_size 4 \
  --output_path outputs/Qwen3-0.6B_ifeval \
  --gen_kwargs "repetition_penalty=1" \
  --log_samples

cd /home/huawei/huawei/lm-evaluation-harness
python -m lm_eval \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/Qwen3-0.6B-GPTQ-Int8 \
  --tasks ifeval \
  --apply_chat_template \
  --device cuda \
  --batch_size 4 \
  --gen_kwargs "repetition_penalty=1.5" \
  --output_path outputs/Qwen3-0.6B-gptq-int8_ifeval \
  --log_samples

```
# gsm-8k
## onnx
```bash
cd /home/huawei/huawei/lm-evaluation-harness
python -m lm_eval \
  --model onnx \
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0522-w8a8-llm/result/qwen3_0_6b_fake_quant_model_merged.onnx,provider=CUDAExecutionProvider,use_amct=true,trim_cpu_mem=true \
  --tasks gsm8k \
  --batch_size 1 \
  --output_path outputs/Qwen3-0.6B_onnx_gsm8k \
  --log_samples \
  --use_cache /home/huawei/huawei/lm-evaluation-harness/cache/gsm8k-amct-0526

python -m lm_eval \
  --model onnx \
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-onnx-ai-onnx-opset16-cache/model.onnx,tokenizer=/home/huawei/huawei/Qwen3-0.6B,provider=CUDAExecutionProvider \
  --tasks gsm8k \
  --batch_size 1 \
  --output_path outputs/Qwen3-0.6B_amct_fakequant_gsm8k \
  --log_samples \
  --num_fewshot 5
```


## hf

cd /home/huawei/huawei/lm-evaluation-harness
python -m lm_eval \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-MXFP8 \
  --tasks gsm8k \
  --device cuda \
  --batch_size 4 \
  --output_path outputs/Qwen3-0.6B-mxfp8_gsm-8k \
  --log_samples \
  --num_fewshot 5


cd /home/huawei/huawei/lm-evaluation-harness
python -m lm_eval \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/gemma-3-1b-it\
  --tasks gsm8k \
  --device cuda \
  --batch_size 4 \
  --output_path outputs/gemma3-1b \
  --log_samples \
  --apply_chat_template \
  --num_fewshot 5 \
  --limit 150

# humaneval+ / mbpp+

题目数（test split）:
- humaneval+: 164
- mbpp+: 378

```bash
cd /home/huawei/huawei/lm-evaluation-harness

# HumanEval+ (164)
HF_ALLOW_CODE_EVAL=1 python -m lm_eval \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/Qwen3-0.6B \
  --tasks humaneval_plus \
  --device cuda \
  --batch_size 1 \
  --gen_kwargs "do_sample=False,temperature=0,top_p=1" \
  --confirm_run_unsafe_code \
  --output_path outputs/Qwen3-0.6B_humaneval_plus \
  --log_samples

# MBPP+ (378)
HF_ALLOW_CODE_EVAL=1 python -m lm_eval \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/Qwen3-0.6B \
  --tasks mbpp_plus \
  --device cuda \
  --batch_size 1 \
  --gen_kwargs "do_sample=False,temperature=0,top_p=1" \
  --confirm_run_unsafe_code \
  --output_path outputs/Qwen3-0.6B_mbpp_plus \
  --log_samples
```