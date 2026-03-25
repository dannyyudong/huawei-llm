# mbpp
HF_ALLOW_CODE_EVAL=1 lm-eval run \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/Qwen3-0.6B-GPTQ-Int8 \
  --tasks mbpp \
  --confirm_run_unsafe_code \
  --output_path /home/huawei/huawei/lm-evaluation-harness/outputs


# ifeval
```bash
cd /home/huawei/huawei/lm-evaluation-harness
python -m lm_eval \
  --model hf \
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-W8A8-Dynamic-Per-Token \
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
  --model_args pretrained=/home/huawei/huawei/quantization/Qwen3-0.6B-W8A8-Dynamic-Per-Token,add_bos_token=true \
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
