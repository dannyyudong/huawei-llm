# mbpp
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
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-W8A8-Dynamic-Per-Token,fix_mistral_regex=true \
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
  --output_path /home/huawei/huawei/lm-evaluation-harness/outputs \
  --log_samples 
```
# ifeval

  --apply_chat_template \ qwen系列加上这个参数分数更高

```bash


cd /home/huawei/huawei/lm-evaluation-harness
python -m lm_eval \
  --model hf \
  --model_args pretrained=/home/shuawei/huawei/quantization/Qwen3-0.6B-W8A8-Dynamic-Per-Token,fix_mistral_regex=true \
  --tasks ifeval \
  --device cuda \
  --batch_size 4 \
  --output_path outputs/Qwen3-0.6B-w8a8_ifeval \
  --gen_kwargs "temperature=0.7,top_p=0.8,top_k=20,repetition_penalty=1" \
  --log_samples

python -m lm_eval \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/Qwen3-0.6B \
  --tasks ifeval \
  --device cuda \
  --batch_size 4 \
  --output_path outputs/Qwen3-0.6B_ifeval \
  --gen_kwargs "temperature=0.7, top_p=0.8, top_k=20,repetition_penalty=1" \
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
cd /home/huawei/huawei/lm-evaluation-harness
python -m lm_eval \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-W8A8-Dynamic-Per-Token,add_bos_token=true,fix_mistral_regex=true \
  --tasks gsm8k \
  --device cuda \
  --batch_size 4 \
  --output_path outputs/Qwen3-0.6B-w8a8_gsm-8k \
  --log_samples \
  --num_fewshot 5


cd /home/huawei/huawei/lm-evaluation-harness
python -m lm_eval \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/Qwen3-0.6B-GPTQ-Int8 \
  --tasks gsm8k \
  --device cuda \
  --batch_size 4 \
  --output_path outputs/Qwen3-0.6B-gptq_gsm-8k \
  --log_samples \
  --num_fewshot 5 \
