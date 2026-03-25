import os
import re
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ================= 配置区域 =================
DATA_DIR = "/home/huawei/huawei/ceval/ceval_data"
MODEL_PATH = "/home/huawei/huawei/quantization/Qwen3-0.6B-W8A8-Dynamic-Per-Token"
MODEL_NAME = "Qwen3-0.6B-w8a8"
EVAL_SPLIT = "test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 4096  # Thinking 需要更多空间
SAVE_PATH = f"ceval_eval_results_{MODEL_NAME}_npthink0310.csv"
THINKING_SAVE_DIR = f"nothinking_logs_{MODEL_NAME}_0310"

# 🎯 指定要评测的科目（留空或设为 None 则评测所有科目）
# 示例：EVAL_SUBJECTS = ["high_school_mathematics", "high_school_physics"]
# 留空评测所有：EVAL_SUBJECTS = None
EVAL_SUBJECTS = None  # 设置为 None 评测所有科目，或指定科目列表
#EVAL_SUBJECTS = ["high_school_mathematics"]
os.makedirs(THINKING_SAVE_DIR, exist_ok=True)

# 📚 科目列表
JUNIOR_SUBJECTS = [
    "middle_school_biology", "middle_school_chemistry", "middle_school_geography",
    "middle_school_history", "middle_school_mathematics", "middle_school_physics",
    "middle_school_politics"
]

SENIOR_SUBJECTS = [
    "high_school_biology", "high_school_chemistry", "high_school_chinese",
    "high_school_geography", "high_school_history", "high_school_mathematics",
    "high_school_physics", "high_school_politics"
]

SUBJECT_LEVEL_MAP = {sub: "Junior" for sub in JUNIOR_SUBJECTS}
SUBJECT_LEVEL_MAP.update({sub: "Senior" for sub in SENIOR_SUBJECTS})
ALL_SUBJECTS = JUNIOR_SUBJECTS + SENIOR_SUBJECTS

SUBJECT_CN_MAP = {
    "middle_school_biology": "初中生物", "middle_school_chemistry": "初中化学",
    "middle_school_geography": "初中地理", "middle_school_history": "初中历史",
    "middle_school_mathematics": "初中数学", "middle_school_physics": "初中物理",
    "middle_school_politics": "初中政治", "high_school_biology": "高中生物",
    "high_school_chemistry": "高中化学", "high_school_chinese": "高中语文",
    "high_school_geography": "高中地理", "high_school_history": "高中历史",
    "high_school_mathematics": "高中数学", "high_school_physics": "高中物理",
    "high_school_politics": "高中政治"
}

# 🎯 标准化输出配置
# GENERATION_CONFIG = {
#     "do_sample": False,
#     "temperature": 0.7,
#     "top_p": 0.8,
#     "top_k": 20,
#     "min_p": 0.0,
# }
GENERATION_CONFIG = {
    "do_sample": False,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.3,       # 防止模型陷入重复循环
    "no_repeat_ngram_size": 6,        # 禁止连续重复的 6-gram
}

# ================= 模型加载 =================
print(f"🚀 正在加载模型：{MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="left")
model_kwargs = {"device_map": "auto", 
                "trust_remote_code":True}
model_kwargs["torch_dtype"] = torch.float16 if DEVICE == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"✅ 模型加载完成，运行设备：{DEVICE}")

# ================= 核心功能函数 =================

def load_local_data(subject, split=EVAL_SPLIT):
    file_path = os.path.join(DATA_DIR, f"{subject}_{split}.csv")
    if not os.path.exists(file_path):
        for alt_split in ["val", "test", "dev"]:
            alt_file = os.path.join(DATA_DIR, f"{subject}_{alt_split}.csv")
            if os.path.exists(alt_file):
                print(f"⚠️ 未找到 {subject}_{split}.csv，改用 {subject}_{alt_split}.csv")
                return pd.read_csv(alt_file, encoding='utf-8')
        raise FileNotFoundError(f"❌ 未找到科目 {subject} 的任何数据文件")
    return pd.read_csv(file_path, encoding='utf-8')

def build_prompt(question, options, subject_name):
    """
    ✅ C-Eval 官方标准化 Prompt（JSON 输出格式 + Thinking）
    """
    subject_cn = SUBJECT_CN_MAP.get(subject_name, subject_name)
    
    # 官方标准化 Prompt 格式
    prompt = f"以下是中国关于{subject_cn}考试的单项选择题，请选出其中的正确答案。\n\n"
    prompt += f"{question}\n"
    for key in ['A', 'B', 'C', 'D']:
        if key in options and pd.notna(options[key]):
            prompt += f"{key}. {options[key]}\n"
    
    # ✅ 标准化输出格式要求（C-Eval 官方建议）
    prompt += '\n请在答案栏中仅显示你的选择，例如："answer": "C"。\n'
   # prompt += '请逐步推理，并将最终答案放在 JSON 格式中。\n'
    prompt += '将最终答案放在 JSON 格式中。\n'
    prompt += '\n输出格式示例：\n{"answer": "A"}\n\n'
    prompt += '你的回答：\n'
    
    return prompt
def model_predict_with_thinking(prompt, question_id, subject_name):
    """
    模型推理（Thinking 模式），返回 (预测答案，thinking 内容，完整输出)
    ✅ 修改：无论是否有 thinking 都保存日志
    """
    try:
        # 构建消息格式（用于 apply_chat_template）
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # ✅ 修改 1：移除 enable_thinking 参数（可能不兼容）
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 已移除
           # enable_thinking=True
        )
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048, padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=GENERATION_CONFIG["do_sample"],
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"],
                top_k=GENERATION_CONFIG["top_k"],
                repetition_penalty=GENERATION_CONFIG.get("repetition_penalty", 1.0),
                no_repeat_ngram_size=GENERATION_CONFIG.get("no_repeat_ngram_size", 0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # ✅ 提取 Thinking 过程（支持多种标签格式）
        thinking_content = ""
        thinking_patterns = [
            r'<think>(.*?)</think>',
            r'<think>(.*?)</think>',
            r'思考过程：(.*?)(?=\n\n|答案|{"answer")',
            r'推理：(.*?)(?=\n\n|答案|{"answer")',
        ]
        
        for pattern in thinking_patterns:
            think_match = re.search(pattern, generated, flags=re.DOTALL | re.IGNORECASE)
            if think_match:
                thinking_content = think_match.group(1).strip()
                break
        
        # ✅ 增强答案提取逻辑（支持更多格式）
        prediction = None
        
        # 方法 1：解析 JSON
        json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', generated, re.IGNORECASE)
        if json_match:
            try:
                json_str = json_match.group(0)
                json_str = json_str.replace("'", '"').replace('""', '"')
                result = json.loads(json_str)
                answer = result.get('answer', '')
                if answer:
                    match = re.search(r'\b([A-D])\b', str(answer), re.IGNORECASE)
                    if match:
                        prediction = match.group(1).upper()
            except:
                pass
        
        # 方法 2：直接提取 "answer": "X"
        if not prediction:
            answer_match = re.search(r'"answer"\s*:\s*"([A-D])"', generated, re.IGNORECASE)
            if answer_match:
                prediction = answer_match.group(1).upper()
        
        # 方法 3：提取中文格式 答案：X
        if not prediction:
            cn_match = re.search(r'答案 [：:]\s*([A-D])', generated, re.IGNORECASE)
            if cn_match:
                prediction = cn_match.group(1).upper()
        
        # 方法 4：从整个输出中提取最后一个 A/B/C/D
        if not prediction:
            matches = re.findall(r'\b([A-D])\b', generated, re.IGNORECASE)
            if matches:
                prediction = matches[-1].upper()
        
        # ✅ 修改 2：无论是否有 thinking 都保存日志
        thinking_file = os.path.join(
            THINKING_SAVE_DIR,
            f"{subject_name}_q{question_id}_thinking.txt"
        )
        
        with open(thinking_file, 'w', encoding='utf-8') as f:
            f.write(f"题目 ID: {question_id}\n")
            f.write(f"科目：{subject_name}\n")
            f.write(f"Prompt:\n{prompt}\n\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Thinking 过程:\n{thinking_content if thinking_content else '(无 thinking 内容)'}\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"完整输出:\n{generated}\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"最终答案：{prediction if prediction else '(未提取到答案)'}\n")
            f.write(f"生成长度：{len(generated)} 字符\n")
            f.write(f"是否包含<think>标签：{'是' if '<think>' in generated else '否'}\n")
        
        # ✅ 修改 3：添加控制台调试信息
        print(f"  [Q{question_id}] 答案={prediction if prediction else 'None'}, "
              f"thinking={len(thinking_content)}字符，输出长度={len(generated)}")
        
        return prediction, thinking_content, generated
        
    except Exception as e:
        print(f"⚠️ 推理出错 [Q{question_id}]：{e}")
        import traceback
        traceback.print_exc()
        
        # ✅ 即使出错也保存错误日志
        thinking_file = os.path.join(
            THINKING_SAVE_DIR,
            f"{subject_name}_q{question_id}_thinking.txt"
        )
        with open(thinking_file, 'w', encoding='utf-8') as f:
            f.write(f"题目 ID: {question_id}\n")
            f.write(f"科目：{subject_name}\n")
            f.write(f"错误信息：{e}\n")
            f.write(f"Prompt:\n{prompt}\n")
        
        return None, "", ""
# def model_predict_with_thinking(prompt, question_id, subject_name):
#     """
#     模型推理（Thinking 模式），返回 (预测答案，thinking 内容)
#     """
#     try:
#         # 构建消息格式（用于 apply_chat_template）
#         messages = [
#             {"role": "user", "content": prompt}
#         ]
        
#         # 使用 apply_chat_template（支持 enable_thinking）
#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#             enable_thinking=True
#         )
        
#         inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096, padding=True).to(model.device)
        
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=MAX_NEW_TOKENS,
#                 do_sample=GENERATION_CONFIG["do_sample"],
#                 temperature=GENERATION_CONFIG["temperature"],
#                 top_p=GENERATION_CONFIG["top_p"],
#                 top_k=GENERATION_CONFIG["top_k"],
#                 pad_token_id=tokenizer.pad_token_id,
#                 eos_token_id=tokenizer.eos_token_id
#             )
        
#         generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
#         # ✅ 提取 Thinking 过程（<think> 标签内）
#         thinking_content = ""
#         think_match = re.search(r'<think>(.*?)</think>', generated, flags=re.DOTALL)
#         if think_match:
#             thinking_content = think_match.group(1).strip()
        
#         # ✅ 提取 JSON 中的 answer 字段
#         prediction = None
        
#         # 方法 1：解析 JSON
#         json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', generated, re.IGNORECASE)
#         if json_match:
#             try:
#                 json_str = json_match.group(0)
#                 # 修复可能的 JSON 格式问题
#                 json_str = json_str.replace("'", '"').replace('""', '"')
#                 result = json.loads(json_str)
#                 answer = result.get('answer', '')
#                 if answer:
#                     match = re.search(r'\b([A-D])\b', str(answer), re.IGNORECASE)
#                     if match:
#                         prediction = match.group(1).upper()
#             except:
#                 pass
        
#         # 方法 2：如果 JSON 解析失败，直接提取 A/B/C/D
#         if not prediction:
#             answer_match = re.search(r'"answer"\s*:\s*"([A-D])"', generated, re.IGNORECASE)
#             if answer_match:
#                 prediction = answer_match.group(1).upper()
#             else:
#                 # 从整个输出中提取最后一个 A/B/C/D
#                 matches = re.findall(r'\b([A-D])\b', generated, re.IGNORECASE)
#                 if matches:
#                     prediction = matches[-1].upper()
        
#         # ✅ 保存 Thinking 过程到文件
#         if thinking_content:
#             thinking_file = os.path.join(
#                 THINKING_SAVE_DIR,
#                 f"{subject_name}_q{question_id}_thinking.txt"
#             )
#             with open(thinking_file, 'w', encoding='utf-8') as f:
#                 f.write(f"题目 ID: {question_id}\n")
#                 f.write(f"科目：{subject_name}\n")
#                 f.write(f"Prompt:\n{prompt}\n\n")
#                 f.write(f"=" * 60 + "\n")
#                 f.write(f"Thinking 过程:\n{thinking_content}\n")
#                 f.write(f"=" * 60 + "\n")
#                 f.write(f"完整输出:\n{generated}\n")
#                 f.write(f"=" * 60 + "\n")
#                 f.write(f"最终答案：{prediction}\n")
        
#         return prediction, thinking_content, generated
        
#     except Exception as e:
#         print(f"⚠️ 推理出错：{e}")
#         return None, "", ""

def evaluate_subject(subject_name):
    """评测单个科目"""
    try:
        df = load_local_data(subject_name, EVAL_SPLIT)
        
        if 'answer' not in df.columns:
            print(f"⚠️ {subject_name} 数据不含 'answer' 列")
            return len(df), 0, None, []
        
        total = len(df)
        correct = 0
        details = []
        
        for idx, item in tqdm(df.iterrows(), total=total, desc=f"Evaluating {subject_name}", leave=False):
            question = str(item['question'])
            options = {
                'A': str(item.get('A', '')),
                'B': str(item.get('B', '')),
                'C': str(item.get('C', '')),
                'D': str(item.get('D', ''))
            }
            ground_truth = str(item['answer']).strip().upper()
            question_id = item.get('id', idx)
            
            prompt = build_prompt(question, options, subject_name)
            prediction, thinking_content, full_output = model_predict_with_thinking(prompt, question_id, subject_name)
            
            is_correct = (prediction == ground_truth)
            if is_correct:
                correct += 1
            
            details.append({
                'id': question_id,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'correct': is_correct,
                'has_thinking': len(thinking_content) > 0
            })
        
        acc = correct / total if total > 0 else 0.0
        return total, correct, acc, details
        
    except Exception as e:
        print(f"❌ 评测 {subject_name} 时出错：{e}")
        import traceback
        traceback.print_exc()
        return 0, 0, None, []

# ================= 主执行流程 =================

# 根据配置决定要评测的科目
if EVAL_SUBJECTS is None or len(EVAL_SUBJECTS) == 0:
    subjects_to_eval = ALL_SUBJECTS
    print(f"\n🔍 开始本地评测所有科目（共 {len(ALL_SUBJECTS)} 个）")
else:
    # 验证指定的科目是否有效
    subjects_to_eval = [s for s in EVAL_SUBJECTS if s in ALL_SUBJECTS]
    invalid_subjects = [s for s in EVAL_SUBJECTS if s not in ALL_SUBJECTS]
    if invalid_subjects:
        print(f"⚠️ 以下科目无效，将被忽略：{invalid_subjects}")
    print(f"\n🔍 开始评测指定的 {len(subjects_to_eval)} 个科目：{subjects_to_eval}")

print(f"📁 数据目录：{DATA_DIR}")
print(f"📋 使用评测集：{EVAL_SPLIT}")
print(f"📝 输出格式：JSON 标准化（C-Eval 官方建议）")
print(f"🧠 Thinking 模式：启用")
print(f"💾 Thinking 日志目录：{os.path.abspath(THINKING_SAVE_DIR)}")
print("=" * 60)

results = []
subject_accuracies = {}
all_details = {}

for subject in subjects_to_eval:
    total, correct, acc, details = evaluate_subject(subject)
    if acc is not None:
        subject_accuracies[subject] = acc
        all_details[subject] = details
        print(f"📊 {subject}: {correct}/{total} = {acc:.2%}")
    else:
        subject_accuracies[subject] = None
        print(f"⚠️ {subject}: 无法计算准确率")

# ================= 计算平均分（按科目数）=================
# 只计算实际评测过的科目
evaluated_junior = [s for s in subjects_to_eval if s in JUNIOR_SUBJECTS]
evaluated_senior = [s for s in subjects_to_eval if s in SENIOR_SUBJECTS]

valid_junior = [subject_accuracies[s] for s in evaluated_junior if subject_accuracies.get(s) is not None]
valid_senior = [subject_accuracies[s] for s in evaluated_senior if subject_accuracies.get(s) is not None]
valid_all = [acc for acc in subject_accuracies.values() if acc is not None]

junior_avg = sum(valid_junior) / len(valid_junior) if valid_junior else None
senior_avg = sum(valid_senior) / len(valid_senior) if valid_senior else None
overall_avg = sum(valid_all) / len(valid_all) if valid_all else None

# ================= 构建结果 DataFrame =================
for subject in subjects_to_eval:
    level = SUBJECT_LEVEL_MAP[subject]
    acc = subject_accuracies[subject]
    results.append({
        "model_name": MODEL_NAME,
        "subject_name": subject,
        "level": level,
        "accuracy": f"{acc:.2%}" if acc is not None else "N/A",
        "junior_avg_acc": f"{junior_avg:.2%}" if junior_avg is not None else "N/A",
        "senior_avg_acc": f"{senior_avg:.2%}" if senior_avg is not None else "N/A",
        "overall_avg_acc": f"{overall_avg:.2%}" if overall_avg is not None else "N/A"
    })

df_results = pd.DataFrame(results)
df_results.to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')

details_path = SAVE_PATH.replace('.csv', '_details.csv')
all_details_flat = []
for subject, details in all_details.items():
    for d in details:
        d['subject'] = subject
        all_details_flat.append(d)

if all_details_flat:
    pd.DataFrame(all_details_flat).to_csv(details_path, index=False, encoding='utf-8-sig')

# ================= 统计 Thinking 文件 =================
thinking_files = [f for f in os.listdir(THINKING_SAVE_DIR) if f.endswith('_thinking.txt')]

print("=" * 60)
print(f"🎉 评测完成！")
print(f"📁 主结果：{os.path.abspath(SAVE_PATH)}")
print(f"📁 详细结果：{os.path.abspath(details_path)}")
print(f"🧠 Thinking 日志：{os.path.abspath(THINKING_SAVE_DIR)} ({len(thinking_files)} 个文件)")
print("\n📈 汇总统计:")
print(f"   初中科目数：{len(valid_junior)}")
print(f"   高中科目数：{len(valid_senior)}")
print(f"   初中平均准确率：{junior_avg:.2%}" if junior_avg else "   初中平均准确率：N/A")
print(f"   高中平均准确率：{senior_avg:.2%}" if senior_avg else "   高中平均准确率：N/A")
print(f"   总体平均准确率：{overall_avg:.2%}" if overall_avg else "   总体平均准确率：N/A")
print("=" * 60)
print("\n📋 详细结果预览:")
print(df_results.to_string())