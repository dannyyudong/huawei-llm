# huawei-llm
git commit -m "your commit message"
git 
2026/3/23
和远程仓库对齐

## todolist 26/3/20
1. qwen3-0.6b-w8a8量化尝试
- [x] 1.1 量化方法环境构建(gptq+smoothquant)
- [x] 1.2 校准数据集构建(开源数据集)
- [x] 1.3 量化尝试
- [ ] 1.4 量化结果测试(与qwen3-0.6b/qwen3-0.6b-gptq-int8横向对比)
2. benchmark 构建
- [ ] Berkeley Function Calling Leaderboard V4 (agent)
- [ ] MCP-Bench(agent)
- [x] gsm-8k(knowledge)
- [x] ifbench(instruction following)
- [ ] MMLU-Redux(knowledge)
- [ ] FollowBench(instruction following)
3. benchmark 测试(qwen3-0.6b)
- [ ] Berkeley Function Calling Leaderboard V4 (agent)
- [ ] MCP-Bench(agent)
- [ ] gsm-8k(knowledge)
- [ ] ifbench(instruction following)
- [ ] MMLU-Redux(knowledge)
- [ ] FollowBench(instruction following)

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