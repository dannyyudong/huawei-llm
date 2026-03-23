import os
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ================= 配置区域 =================
DATA_DIR = "/home/huawei/huawei/ceval/ceval_data"
MODEL_PATH = "/home/huawei/huawei/Qwen3-0.6B"
MODEL_NAME = "Qwen3-0.6B-Thinking"
EVAL_SPLIT = "test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 2048  # 思考模式需要更多 token 用于推理过程
SAVE_PATH = "ceval_eval_results_qwen3_thinking.csv"

# 🧠 Qwen3 思考模式解码参数（官方推荐）
GENERATION_CONFIG = {
    "do_sample": True,      # ✅ 必须为 True，不能用贪婪解码
    "temperature": 0.6,     # ✅ 官方推荐
    "top_p": 0.95,          # ✅ 官方推荐
    "top_k": 20,            # ✅ 官方推荐
    "repetition_penalty": 1.1,  # 防止重复
    "pad_token_id": None,   # 稍后设置
    "eos_token_id": None    # 稍后设置
}

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

# ================= 模型加载 =================
print(f"🚀 正在加载模型：{MODEL_PATH} ...")
print(f"🧠 使用思考模式：enable_thinking=True")
print(f"📋 解码参数：temperature=0.6, top_p=0.95, top_k=20")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model_kwargs = {"device_map": "auto", "trust_remote_code": True}
model_kwargs["torch_dtype"] = torch.float16 if DEVICE == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
model.eval()

# 设置 pad_token 和 eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
GENERATION_CONFIG["pad_token_id"] = tokenizer.pad_token_id
GENERATION_CONFIG["eos_token_id"] = tokenizer.eos_token_id

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

def build_prompt_thinking(question, options, subject_name):
    """
    ✅ Qwen3 思考模式 Prompt（使用 apply_chat_template）
    """
    subject_cn = SUBJECT_CN_MAP.get(subject_name, subject_name)
    
    system_prompt = "你是一个专业的考试答题助手，请仔细思考后给出正确答案。"
    
    user_content = f"以下是中国关于{subject_cn}考试的单项选择题，请选出其中的正确答案。\n\n"
    user_content += f"{question}\n"
    for key in ['A', 'B', 'C', 'D']:
        if key in options and pd.notna(options[key]):
            user_content += f"{key}. {options[key]}\n"
    user_content += "答案："
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # ✅ 使用 apply_chat_template 并启用思考模式
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # ✅ 启用思考模式
    )
    
    return text

def model_predict_thinking(prompt):
    """
    ✅ Qwen3 思考模式推理（使用官方推荐参数）
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                **GENERATION_CONFIG  # ✅ 使用思考模式参数
            )
        
        generated = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # 提取答案（思考模式可能输出 <think>...</think> 答案）
        # 先尝试提取</think> 后的内容
        think_match = re.search(r'</think>\s*([A-D])', generated, re.IGNORECASE)
        if think_match:
            return think_match.group(1).upper()
        
        # 再尝试提取普通的 A/B/C/D
        match = re.search(r'\b([A-D])\b', generated, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        return None
        
    except Exception as e:
        print(f"⚠️ 推理出错：{e}")
        return None

def evaluate_subject(subject_name):
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
            options = {'A': str(item.get('A', '')), 'B': str(item.get('B', '')), 
                       'C': str(item.get('C', '')), 'D': str(item.get('D', ''))}
            ground_truth = str(item['answer']).strip().upper()
            question_id = item.get('id', idx)
            
            # ✅ 使用思考模式 Prompt
            prompt = build_prompt_thinking(question, options, subject_name)
            prediction = model_predict_thinking(prompt)
            
            is_correct = (prediction == ground_truth)
            if is_correct:
                correct += 1
            
            details.append({'id': question_id, 'prediction': prediction, 
                           'ground_truth': ground_truth, 'correct': is_correct})
        
        acc = correct / total if total > 0 else 0.0
        return total, correct, acc, details
    except Exception as e:
        print(f"❌ 评测 {subject_name} 时出错：{e}")
        import traceback
        traceback.print_exc()
        return 0, 0, None, []

# ================= 主执行流程 =================

print(f"🔍 开始本地评测 {len(ALL_SUBJECTS)} 个科目")
print(f"📁 数据目录：{DATA_DIR}")
print(f"📋 使用评测集：{EVAL_SPLIT}")
print(f"🧠 模式：Qwen3 Thinking Mode")
print("=" * 60)

results = []
subject_accuracies = {}
all_details = {}

for subject in ALL_SUBJECTS:
    total, correct, acc, details = evaluate_subject(subject)
    if acc is not None:
        subject_accuracies[subject] = acc
        all_details[subject] = details
        print(f"📊 {subject}: {correct}/{total} = {acc:.2%}")
    else:
        subject_accuracies[subject] = None
        print(f"⚠️ {subject}: 无法计算准确率")

# ================= 计算平均分（按科目数）=================
valid_junior = [subject_accuracies[s] for s in JUNIOR_SUBJECTS if subject_accuracies[s] is not None]
valid_senior = [subject_accuracies[s] for s in SENIOR_SUBJECTS if subject_accuracies[s] is not None]
valid_all = [acc for acc in subject_accuracies.values() if acc is not None]

junior_avg = sum(valid_junior) / len(valid_junior) if valid_junior else None
senior_avg = sum(valid_senior) / len(valid_senior) if valid_senior else None
overall_avg = sum(valid_all) / len(valid_all) if valid_all else None

# ================= 构建结果 DataFrame =================
for subject in ALL_SUBJECTS:
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

print("=" * 60)
print(f"🎉 评测完成！")
print(f"📁 主结果：{os.path.abspath(SAVE_PATH)}")
if all_details_flat:
    print(f"📁 详细结果：{os.path.abspath(details_path)}")
print("\n📈 汇总统计:")
print(f"   初中科目数：{len(valid_junior)}")
print(f"   高中科目数：{len(valid_senior)}")
print(f"   初中平均准确率：{junior_avg:.2%}" if junior_avg else "   初中平均准确率：N/A")
print(f"   高中平均准确率：{senior_avg:.2%}" if senior_avg else "   高中平均准确率：N/A")
print(f"   总体平均准确率：{overall_avg:.2%}" if overall_avg else "   总体平均准确率：N/A")
print("=" * 60)
print("\n📋 详细结果预览:")
print(df_results.to_string())