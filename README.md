# huawei-llm
git add -u
git commit -m "your commit message"
git push
2026/3/23
和远程仓库对齐
![alt text](1775016280050.png)
## (ii) CVALUES multi-choice responsibility prompts (Bias &Safety Evaluation)
```bash
python generate_qwen3_responses.py \
  --model_path /home/huawei/huawei/Qwen3-0.6B \
  --output_file ./data/cvalues_qwen3-0.6b.jsonl \
  --torch_dtype bfloat16 \
  --resume \
  --eval_after_generate
  

```
## (iii) SafetyBench (Safety Evaluation)
```bash
cd /home/huawei/huawei/SafetyBench
python3 code/evaluate_hf.py \
  --model /home/huawei/huawei/Qwen3-0.6B \
  --split zh \
  --shots 0 \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 4
```

## (iv) Political Compass Test (Politic Evaluation)
```bash
cd /home/huawei/huawei/llm-values-pct
conda activate llm

python src/eval_pct_semantic.py \
  --model_name_or_path /home/huawei/huawei/Qwen3-0.6B \
  --questions_path data/templates/pct_propositions.csv \
  --output_dir data/completions/Qwen3-0.6B-semantic-eval \
  --batch_size 16 \
  --require_cuda
  cat data/completions/Qwen3-0.6B-semantic-eval/summary.json
```
## amct安装

pip intall /home/huawei/huawei/amct_onnx/amct_onnx-0.23.2-py3-none-linux_x86_64.whl

cd /home/huawei/huawei/amct_onnx/amct_onnx_op && python3 setup.py build
对于onnx环境来说，
export LD_PRELOAD=$CONDA_PREFIX/lib/libcudart.so
ls -l $CONDA_PREFIX/lib/libcudart.so
后就能正常运行。

## 26/5/4
1. onnx 后是否和原模型对齐
2. no kvcache的问题




## 26/3/29

部分测评集需要api,及需要大模型辅助测评或是需要链接服务器进行测试，因而对测试集进行替换，将mcp-bench换为humaneval
## todolist 26/3/20
1. qwen3-0.6b-w8a8量化尝试
- [x] 1.1 量化方法环境构建(gptq+smoothquant)
- [x] 1.2 校准数据集构建(开源数据集)
- [x] 1.3 量化尝试
- [ ] 1.4 量化结果测试(与qwen3-0.6b/qwen3-0.6b-gptq-int8横向对比)
2. benchmark 构建
- [x] Berkeley Function Calling Leaderboard V4 (agent)
- [ ] MCP-Bench(agent)
- [x] humaneval(coding) 
- [x] gsm-8k(knowledge)
- [x] ifbench(instruction following)
- [x] MMLU-Redux(knowledge)
- [x] FollowBench(instruction following)
3. benchmark 测试(qwen3-0.6b)
- [x] Berkeley Function Calling Leaderboard V4 (agent)
- [ ] MCP-Bench(agent)
- [x] gsm-8k(knowledge)
- [x] ifbench(instruction following)
- [x] MMLU-Redux(knowledge)
- [ ] FollowBench(instruction following)
- [ ] humaneval(coding) 
## timeline 26/3/25-26/3/29

### 3/25 (周三)
- 目标: 完成 1.4 量化结果测试第二轮
- 任务:
	- 构建 Berkeley Function Calling Leaderboard V4
	- 跑 qwen3-0.6b-w8a8 的 ceval
	- 完成 smoke test（小样本可跑通，包括qwen量化格式）
- 交付:
	- 填充w8a8量化横向对比表（3 个模型 x 3 个任务）

### 3/26 (周四)
- 目标: 完成 benchmark 构建第一批
- 任务:
	- 完整测试 Berkeley Function Calling Leaderboard V4
	- 构建 MCP-Bench
	- 完成 smoke test（小样本可跑通，包括qwen量化格式）
- 交付:
	- 两个 benchmark 的可复现运行命令与依赖说明

### 3/27 (周五)
- 目标: 完成 benchmark 构建第二批
- 任务:
    - 完整测试MCP-Bench
	- 构建 gsm-8k、ifbench
	- 完成 smoke test（小样本可跑通，包括qwen量化格式）
- 交付:
	- 两个 benchmark 的可复现运行命令与依赖说明

### 3/28 (周六)
- 目标: 完成 benchmark 构建第三批
- 任务:
    - 完整测试 gsm-8k、ifbench
	- 构建MMLU-Redux、FollowBench
	- 完成 smoke test（小样本可跑通，包括qwen量化格式）
- 交付:
	- 两个 benchmark 的可复现运行命令与依赖说明

### 3/29 (周日)
- 目标: 完成周度收口与结论
- 任务:
	- 补跑未完成任务
	- 汇总新增benchmark结果

- 交付:
	- 部分结果汇总ppt

## todo list 26/3/31
1. 测试不同量化方法之间的差异
2. 测试不同校准数据集之间的差异
3. 构建数据集(如有)

## future work
微调