# ceval
```bash
cd  /home/huawei/huawei/ceval
python eval_json.py --model-path /home/huawei/huawei/Qwen3-0.6B-MXFP8   --model-name qwen3-0.6b-mxfp8
```
# MMLU-Redux
```bash
cd /home/huawei/huawei/mmlu-redux
python scripts/eval_mmlu_redux_qwen_local.py   --model_path /home/huawei/huawei/DeepSeek-R1-Distill-Qwen-1.5B   --dataset_root /home/huawei/huawei/mmlu-redux/data/mmlu-redux   --subjects all   --output_dir /home/huawei/huawei/mmlu-redux/outputs/deepseek-1.5b  --trust_remote_code   --show_progress  
```
# Followbench
```bash
cd /home/huawei/huawei/FollowBench
python code_zh/eval_rule_json.py --save_dir /home/huawei/huawei/FollowBench/gemma3-1b --model_path /home/huawei/huawei/gemma-3-1b-it  --model_name gemma3-1b
```
# IFBench
```bash
cd /home/huawei/huawei/IFBench
python generate_responses_hf.py \
	--model-path /home/huawei/huawei/gemma-3-1b-it \
	--input-file data/IFBench_test.jsonl \
	--output-file data/gemma3-1b.jsonl \
	--max-new-tokens 1024 \
	--disable-thinking 
	--strip-think

python -m run_eval --input_data=data/IFBench_test.jsonl --input_response_data=data/DeepSeek-R1-Distill-Qwen-1.5B.jsonl --output_dir=eval
```

# BFCLv4

```bash 
```bash
bfcl generate \
  --model Qwen/Qwen3-1.7B \
  --test-category non_live \
  --backend transformers \
  --local-model-path /home/huawei/huawei/DeepSeek-R1-Distill-Qwen-1.5B \
  --result-dir result_qwen3-1.7b \
  --num-threads 1 
  --enable-think=false \
  --fix-mistral-regex

bfcl generate \
  --model google/gemma-3-1b-it \
  --backend transformers \
  --local-model-path /home/huawei/huawei/gemma-3-1b-it \
  --test-category non_live \
  --result-dir result_gemma3_1b \
  --num-threads 2 
bfcl evaluate --model google/gemma-3-1b-it --test-category non_live --result-dir result_gemma3_1b  --score-dir score_qwen3.5-0.8b

bfcl generate \
  --model deepseek-ai/DeepSeek-R1 \
  --backend transformers \
  --local-model-path /home/huawei/huawei/DeepSeek-R1-Distill-Qwen-1.5B \
  --test-category non_live \
  --result-dir result_deepseek_1.5b \
  --num-threads 2 
bfcl evaluate --model deepseek-ai/DeepSeek-R1 --test-category non_live --result-dir result_deepseek_1.5b  --score-dir score_deepseek_1.5b

bfcl evaluate --model Qwen/Qwen3-1.7B --test-category non_live --result-dir result_qwen3.5-0.8b  --score-dir score_qwen3.5-0.8b

`--enable-think` 是可选参数；不传就不会覆盖默认行为。
```
```