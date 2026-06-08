import os
import re
import json
import argparse
import pandas as pd
import torch
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ================= 配置区域 =================
DATA_DIR = "/home/huawei/huawei/ceval/ceval_data"
DEFAULT_MODEL_PATH = "/home/huawei/huawei/Qwen3-1.7B"
DEFAULT_MODEL_NAME = "Qwen3-1.7b"
EVAL_SPLIT = "test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 4096  # Thinking 需要更多空间


def parse_args():
    parser = argparse.ArgumentParser(description="C-Eval 本地评测脚本")
    parser.add_argument(
        "--model-path",
        dest="model_path",
        default=DEFAULT_MODEL_PATH,
        help="模型路径，默认使用脚本内置路径",
    )
    parser.add_argument(
        "--model-name",
        dest="model_name",
        default=DEFAULT_MODEL_NAME,
        help="模型名称，用于结果文件命名",
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "onnx"],
        default="hf",
        help="推理后端：hf(默认) 或 onnx",
    )
    parser.add_argument(
        "--onnx-model-path",
        dest="onnx_model_path",
        default="/home/huawei/huawei/Qwen3-0.6B-ONNX/onnx/model.onnx",
        help="ONNX 模型文件路径（backend=onnx 时使用）",
    )
    parser.add_argument(
        "--onnx-provider",
        dest="onnx_provider",
        default="CUDAExecutionProvider",
        help="ONNX Runtime provider，默认 CUDAExecutionProvider",
    )
    return parser.parse_args()


args = parse_args()
MODEL_PATH = args.model_path
MODEL_NAME = args.model_name
BACKEND = args.backend
ONNX_MODEL_PATH = args.onnx_model_path
ONNX_PROVIDER = args.onnx_provider
MODEL_NAME_TAG = re.sub(r"[^0-9A-Za-z._-]+", "_", MODEL_NAME)
ONNX_TOKENIZER_PATH = None
if BACKEND == "onnx":
    if os.path.isdir(ONNX_MODEL_PATH):
        ONNX_TOKENIZER_PATH = ONNX_MODEL_PATH
        ONNX_MODEL_PATH = os.path.join(ONNX_MODEL_PATH, "model.onnx")
    else:
        ONNX_TOKENIZER_PATH = os.path.dirname(ONNX_MODEL_PATH)

SAVE_PATH = f"ceval_eval_results_{MODEL_NAME_TAG}_npthinkdosample0508.csv"
THINKING_SAVE_DIR = f"nothinking_logs_{MODEL_NAME_TAG}_0508"

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
  #  "repetition_penalty": 1,       # 防止模型陷入重复循环
    "no_repeat_ngram_size": 6,        # 禁止连续重复的 6-gram
}

# ================= 模型加载 =================
tokenizer_source = ONNX_TOKENIZER_PATH if BACKEND == "onnx" else MODEL_PATH
print(f"🚀 正在加载 tokenizer：{tokenizer_source} ...")
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_source,
    trust_remote_code=True,
    padding_side="left",
    fix_mistral_regex=True,##只有使用w8a8量化模型时才需要这个参数，其他模型可能不兼容
)
model = None
if BACKEND == "hf":
    print(f"🚀 正在加载 HF 模型：{MODEL_PATH} ...")
    model_kwargs = {"device_map": "auto", 
                    "trust_remote_code":True}
    model_kwargs["torch_dtype"] = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ✅ 兼容 Gemma 等模型的多终止 token（如 [1, 106]）
STOP_TOKEN_IDS = getattr(model.generation_config, "eos_token_id", None) if model is not None else None
if STOP_TOKEN_IDS is None:
    STOP_TOKEN_IDS = tokenizer.eos_token_id
if isinstance(STOP_TOKEN_IDS, int):
    STOP_TOKEN_IDS = [STOP_TOKEN_IDS]
elif isinstance(STOP_TOKEN_IDS, tuple):
    STOP_TOKEN_IDS = list(STOP_TOKEN_IDS)

if BACKEND == "hf":
    print(f"✅ HF 模型加载完成，运行设备：{DEVICE}")
else:
    print("✅ ONNX 模式：已跳过 HF 模型加载")
print(f"✅ 停止 token ids: {STOP_TOKEN_IDS}")

# ================= 核心功能函数 =================
class OnnxQwenRunner:
    def __init__(self, onnx_path, provider="CUDAExecutionProvider"):
        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(f"请求的 ONNX provider={provider} 不可用，可用 providers={available}")
        self.sess = ort.InferenceSession(onnx_path, providers=[provider])
        past_key_inputs = [
            x for x in self.sess.get_inputs()
            if x.name.startswith("past_key_values") and x.name.endswith(".key")
        ]
        if not past_key_inputs:
            raise RuntimeError("ONNX 输入不包含 past_key_values.*.key")
        self.num_layers = len(past_key_inputs)
        shape = past_key_inputs[0].shape
        self.num_heads = int(shape[1]) if isinstance(shape[1], int) else 8
        self.head_dim = int(shape[3]) if isinstance(shape[3], int) else 128

    def _empty_past(self, batch=1):
        past = []
        for _ in range(self.num_layers):
            k = np.zeros((batch, self.num_heads, 0, self.head_dim), dtype=np.float32)
            v = np.zeros((batch, self.num_heads, 0, self.head_dim), dtype=np.float32)
            past.append((k, v))
        return past

    def _step(self, input_ids, attention_mask, position_ids, past):
        feeds = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
            "position_ids": position_ids.astype(np.int64),
        }
        for i, (k, v) in enumerate(past):
            feeds[f"past_key_values.{i}.key"] = k
            feeds[f"past_key_values.{i}.value"] = v
        outs = self.sess.run(None, feeds)
        logits = outs[0]
        presents = outs[1:]
        new_past = []
        for i in range(self.num_layers):
            new_past.append((presents[2 * i], presents[2 * i + 1]))
        return logits, new_past

    def generate(self, input_ids, max_new_tokens, eos_token_ids):
        eos_token_ids = set(eos_token_ids or [])
        tokens = list(input_ids)
        past = self._empty_past(batch=1)

        seq = np.array(tokens, dtype=np.int64)[None, :]
        attn = np.ones_like(seq, dtype=np.int64)
        pos = np.arange(seq.shape[1], dtype=np.int64)[None, :]
        logits, past = self._step(seq, attn, pos, past)

        last = logits[:, -1, :]
        nid = int(np.argmax(last, axis=-1)[0])
        tokens.append(nid)
        if nid in eos_token_ids:
            return tokens

        for _ in range(max_new_tokens - 1):
            cur = np.array([[tokens[-1]]], dtype=np.int64)
            total_len = past[0][0].shape[2] + 1
            attn = np.ones((1, total_len), dtype=np.int64)
            pos = np.array([[total_len - 1]], dtype=np.int64)
            logits, past = self._step(cur, attn, pos, past)
            last = logits[:, -1, :]
            nid = int(np.argmax(last, axis=-1)[0])
            tokens.append(nid)
            if nid in eos_token_ids:
                break
        return tokens


onnx_runner = None
if BACKEND == "onnx":
    print(f"🚀 正在加载 ONNX 模型：{ONNX_MODEL_PATH} ...")
    print(f"⚙️ ONNX provider: {ONNX_PROVIDER}")
    onnx_runner = OnnxQwenRunner(ONNX_MODEL_PATH, provider=ONNX_PROVIDER)
    print("✅ ONNX 模型加载完成")

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
        
        if BACKEND == "hf":
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
                    eos_token_id=STOP_TOKEN_IDS
                )
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        else:
            np_inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=2048, padding=True)
            prompt_ids = np_inputs["input_ids"][0].tolist()
            out_ids = onnx_runner.generate(
                input_ids=prompt_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_ids=STOP_TOKEN_IDS
            )
            new_ids = out_ids[len(prompt_ids):]
            generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        
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
print(f"⚙️ 推理后端：{BACKEND}")
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
