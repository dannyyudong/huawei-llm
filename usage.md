# llm环境跑onnx格式的时候要加export LD_LIBRARY_PATH=/home/huawei/miniconda3/envs/llm/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# ceval
```bash
cd  /home/huawei/huawei/ceval
python eval_json.py --model-path /home/huawei/huawei/Qwen3-0.6B-GPTQ-Int8   --model-name qwen3-0.6b-gptq
```
## vllm
```bash
cd /home/huawei/huawei/ceval
conda run -n llm python eval_json_vllm.py \
  --backend vllm \
  --model-path /home/huawei/huawei/Qwen3-0.6B-GPTQ-Int8 \
  --model-name Qwen3-0.6B-gptq-vllm \
  --quantization compressed-tensors \
  --max-model-len 4096 \
  --eval-subjects high_school_physics \
  cd /home/huawei/huawei/ceval
python eval_json_vllm.py \
  --backend vllm \
  --model-path /home/huawei/huawei/Qwen3-0.6B \
  --model-name Qwen3-0.6B-vllm \
  --max-model-len 4096


  conda run -n llm python eval_json_vllm.py \
  --backend vllm \
  --model-path /home/huawei/huawei/Qwen3-0.6B-GPTQ-Int8 \
  --model-name Qwen3-0.6B-gptq-vllm \
  --max-model-len 4096
```
## onnx
```bash 
python /home/huawei/huawei/ceval/eval_json_onnx.py \
  --backend onnx \
  --onnx-model-path /home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0522-w8a8-llm/result/qwen3_0_6b_fake_quant_model_merged.onnx \
  --model-name Qwen3-0.6B-onnx-amct-0522-w8a8-llm \
  --onnx-provider CUDAExecutionProvider
```
# MMLU-Redux
## onnx
```bash
python /home/huawei/huawei/mmlu-redux/scripts/eval_mmlu_redux_qwen_onnx.py \
  --backend onnx \
  --onnx-model-path /home/huawei/huawei/quantization/Qwen3-0.6B-onnx-ai-onnx-opset16-cache/model.onnx \
  --onnx-model-type onnx \
  --onnx-provider CUDAExecutionProvider \
  --dataset_root /home/huawei/huawei/mmlu-redux/data/mmlu-redux \
  --output_dir /home/huawei/huawei/mmlu-redux/outputs/qwen3-0.6b-onnx-normal-mmlu \
  --trust_remote_code \
  --fix-mistral-regex
python /home/huawei/huawei/mmlu-redux/scripts/eval_mmlu_redux_qwen_onnx.py \
  --backend onnx \
  --onnx-model-path /home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0522-w8a8-llm/result/qwen3_0_6b_fake_quant_model_merged.onnx \
  --onnx-model-type amct \
  --onnx-provider CUDAExecutionProvider \
  --dataset_root /home/huawei/huawei/mmlu-redux/data/mmlu-redux \
  --output_dir /home/huawei/huawei/mmlu-redux/outputs/qwen3-0.6b-onnx-amct0525-mmlu \
  --trust_remote_code \
  --fix-mistral-regex \
  --resume
  ```
## hf
```bash
cd /home/huawei/huawei/mmlu-redux
python scripts/eval_mmlu_redux_qwen_local.py   --model_path /home/huawei/huawei/DeepSeek-R1-Distill-Qwen-1.5B   --dataset_root /home/huawei/huawei/mmlu-redux/data/mmlu-redux   --subjects all   --output_dir /home/huawei/huawei/mmlu-redux/outputs/deepseek-1.5b  --trust_remote_code   --show_progress  
```
# Followbench
## onnx
cd /home/huawei/huawei/FollowBench/
python code_zh/eval_rule_json_onnx.py \
  --backend onnx \
  --onnx-model-path /home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0522-w8a8-llm/result/qwen3_0_6b_fake_quant_model_merged.onnx\
  --model-name Qwen3-0.6B-ONNX-amct \
  --resume

## hf
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

python /home/huawei/huawei/IFBench/generate_responses_onnx.py \
  --model-path /home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0509 \
  --output-file /home/huawei/huawei/IFBench/data/amct-fakequant-responses.jsonl
python /home/huawei/huawei/IFBench/generate_responses_onnx.py \
  --model-path /home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0522-w8a8-llm \
  --output-file /home/huawei/huawei/IFBench/data/amct-onnx-0526-responses.jsonl
  
python -m run_eval --input_data=data/IFBench_test.jsonl --input_response_data=/home/huawei/huawei/IFBench/data/amct-onnx-0526-responses.jsonl --output_dir=eval
```

# BFCLv4
## onnx
```bash
cd /home/huawei/huawei/gorilla/berkeley-function-call-leaderboard
bfcl generate \
  --model Qwen/Qwen3-0.6B \
  --test-category non_live \
  --backend onnx-amct \
  --onnx-model-path /home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0522-w8a8-llm/result/qwen3_0_6b_fake_quant_model_merged.data \
  --result-dir result_qwen3-0.6b-amct-w8a8 \
  --num-threads 1 \
  --enable-think=false \
  --fix-mistral-regex
  ```

  bfcl evaluate --model Qwen/Qwen3-0.6B --test-category non_live --result-dir result_qwen3-0.6b-amct-w8a8  --score-dir score_qwen3-0.6b-amct-w8a8
  
## hf
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


# LiveCodeBench (Qwen Local)
```bash
export HF_DATASETS_CACHE=/home/huawei/huawei/LiveCodeBench/local_datasets/hf_datasets_cache
export HF_MODULES_CACHE=/home/huawei/huawei/LiveCodeBench/local_datasets/hf_modules_cache
```
## onnx
```bash
cd /home/huawei/huawei/LiveCodeBench
export HF_DATASETS_CACHE=/home/huawei/huawei/LiveCodeBench/local_datasets/hf_datasets_cache
export HF_MODULES_CACHE=/home/huawei/huawei/LiveCodeBench/local_datasets/hf_modules_cache
python -m lcb_runner.runner.main \
  --backend onnx \
  --model Qwen/Qwen2.5-7B-Instruct \
  --onnx-model-path /home/huawei/huawei/quantization/Qwen3-0.6B-onnx-amct-0522-w8a8-llm/result/qwen3_0_6b_fake_quant_model_merged.onnx \
  --enable_thinking false \
  --scenario codegeneration \
  --release_version release_v6 \
  --n 1 \
  --max_tokens 1024 \
  --temperature 0.001 \
  --trust_remote_code \
  --evaluate \
  --continue_existing \
  --run_id 20260528
  ```
## hf

/home/huawei/huawei/LiveCodeBench/lcb_runner/lm_styles.py
--model仅代表提示词风格
```bash
source /home/huawei/miniconda3/etc/profile.d/conda.sh
conda activate llm
cd /home/huawei/huawei/LiveCodeBench

# debug 先跑小样本
python -m lcb_runner.runner.main \
  --backend transformers \
  --model Qwen/Qwen2.5-7B-Instruct \
  --local_model_path /home/huawei/huawei/Qwen3-0.6B \
  --enable_thinking false \
  --scenario codegeneration \
  --release_version release_v6 \
  --debug \
  --n 1 \
  --max_tokens 64 \
  --temperature 0.2 \
  --top_p 0.95 \
  --stop "###" \
  --trust_remote_code



# 抽 50 题并评测
python -m lcb_runner.runner.main \
  --backend transformers \
  --model Qwen/Qwen2.5-7B-Instruct \
  --local_model_path /home/huawei/huawei/Qwen3-0.6B \
  --enable_thinking false \
  --scenario codegeneration \
  --release_version release_v6 \
  --max_samples 50 \
  --n 1 \
  --max_tokens 1024 \
  --temperature 0.001 \
  --top_p 0.95 \
  --stop "###" \
  --trust_remote_code \
  --evaluate 

# 全量跑（去掉 --debug）
cd /home/huawei/huawei/LiveCodeBench
python -m lcb_runner.runner.main \
  --backend transformers \
  --model Qwen/Qwen2.5-7B-Instruct \
  --local_model_path /home/huawei/huawei/Qwen3-0.6B \
  --enable_thinking false \
  --scenario codegeneration \
  --release_version release_v6 \
  --n 1 \
  --max_tokens 1024 \
  --temperature 0.001 \
  --trust_remote_code \
  --evaluate

cd /home/huawei/huawei/LiveCodeBench
python -m lcb_runner.runner.main \
  --backend transformers \
  --model Qwen/Qwen2.5-7B-Instruct \
  --local_model_path /home/huawei/huawei/Qwen3.5-0.8B \
  --enable_thinking false \
  --scenario codegeneration \
  --release_version release_v6 \
  --n 1 \
  --max_tokens 1024 \
  --temperature 0.001 \
  --trust_remote_code \
  --evaluate \
  --continue_existing

# DeepSeek-R1-Distill-Qwen-1.5B（本地）
cd /home/huawei/huawei/LiveCodeBench
python -m lcb_runner.runner.main \
  --backend transformers \
  --model deepseek-ai/DeepSeek-R1 \
  --local_model_path /home/huawei/huawei/DeepSeek-R1-Distill-Qwen-1.5B \
  --scenario codegeneration \
  --release_version release_v6 \
  --n 1 \
  --max_tokens 2048 \
  --temperature 0.001 \
  --trust_remote_code \
  --evaluate

cd /home/huawei/huawei/LiveCodeBench
python -m lcb_runner.runner.main \
  --backend transformers \
  --model Qwen/Qwen2.5-7B-Instruct \
  --local_model_path /home/huawei/huawei/Qwen3.5-0.8B \
  --enable_thinking false \
  --scenario codegeneration \
  --release_version release_v6 \
  --n 1 \
  --max_tokens 1024 \
  --temperature 0.001 \
  --trust_remote_code \
  --evaluate \
  --continue_existing \
  --run_id 20260428_130228

  cd /home/huawei/huawei/LiveCodeBench
python -m lcb_runner.runner.main \
  --backend transformers \
  --model Qwen/Qwen2.5-7B-Instruct \
  --local_model_path /home/huawei/huawei/Qwen3.5-0.8B \
  --enable_thinking false \
  --scenario codegeneration \
  --release_version release_v6 \
  --n 1 \
  --max_tokens 1024 \
  --temperature 0.001 \
  --trust_remote_code \
  --evaluate \
  --continue_existing_with_eval \
  --run_id 20260428_130228 \
  --max_samples 50
# 如果要跑 Qwen3-1.7B，把路径替换成：
# --local_model_path /home/huawei/huawei/Qwen3-1.7B
```
