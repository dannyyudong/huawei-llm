import os
import json
import csv
import random
import argparse
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from rule_based_evaluation import rule_evaluation_format, check_match
import rule_based_evaluation as rbe


# ================= Configuration =================
DATA_PATH = "/home/huawei/huawei/FollowBench/data_zh"
MODEL_PATH = "/home/huawei/huawei/Qwen3-0.6B-GPTQ-Int8"
MODEL_NAME = "Qwen3-0.6B-gptq"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 1024

# Set to None to evaluate all rule-evaluable constraint types.
# Example: ["content", "situation", "mixed", "format", "example"]
EVAL_CONSTRAINT_TYPES = None

# Small-sample quick test (None means full evaluation)
SMALL_SAMPLE_PER_TYPE = None
SAMPLE_SEED = 42

# Default thinking mode control for chat template
ENABLE_THINKING = False

# Prompt style aligned with eval_json: explicit output-only instructions.
PROMPT_STYLE = "eval_json"

GENERATION_CONFIG = {
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.0,
}

SAVE_DIR = f"rule_eval_results_{MODEL_NAME}"
DETAIL_CSV = os.path.join(SAVE_DIR, f"rule_eval_details_{MODEL_NAME}.csv")
SUMMARY_CSV = os.path.join(SAVE_DIR, f"rule_eval_summary_{MODEL_NAME}.csv")
LEVEL_CSV = os.path.join(SAVE_DIR, f"rule_eval_by_level_{MODEL_NAME}.csv")
SOURCE_CSV = os.path.join(SAVE_DIR, f"rule_eval_by_source_{MODEL_NAME}.csv")
CSL_CSV = os.path.join(SAVE_DIR, f"rule_eval_csl_{MODEL_NAME}.csv")

SAVE_GENERATION_LOGS = True
GEN_LOG_DIR = os.path.join(SAVE_DIR, "generation_logs")

RULE_BASED_SOURCES = {
    "E2E",
    "WIKIEVENTS",
    "CONLL2003",
    "text_editing",
    "cnn_dailymail",
    "xsum",
    "samsum",
    "gigaword",
    "arxiv",
    "BBH_logical",
    "BBH_time",
    "self_made_space",
    "gsm_8k",
}

RULE_BASED_FORMAT_EXAMPLE_IDS = {22}
ALL_CONSTRAINT_TYPES = ["content", "situation", "style", "format", "example", "mixed"]


def get_constraint_types():
    if EVAL_CONSTRAINT_TYPES is None or len(EVAL_CONSTRAINT_TYPES) == 0:
        return ALL_CONSTRAINT_TYPES
    valid = [x for x in EVAL_CONSTRAINT_TYPES if x in ALL_CONSTRAINT_TYPES]
    invalid = [x for x in EVAL_CONSTRAINT_TYPES if x not in ALL_CONSTRAINT_TYPES]
    if invalid:
        print(f"[WARN] Ignored invalid constraint types: {invalid}")
    return valid


def load_constraint_data(constraint_type):
    file_path = os.path.join(DATA_PATH, f"{constraint_type}_constraints.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing data file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_rule_evaluable(record, constraint_type):
    if record.get("level", 0) <= 0:
        return False, "base-level-not-evaluated"

    if constraint_type == "example":
        return True, "example-template-match"

    source = record.get("source")
    if source in RULE_BASED_SOURCES:
        return True, "source-rule"

    if (
        constraint_type == "format"
        and record.get("category") == "format"
        and record.get("example_id") in RULE_BASED_FORMAT_EXAMPLE_IDS
    ):
        return True, "format-special-rule"

    return False, "needs-llm-judge"


def build_prompt(record, constraint_type):
    instruction = record.get("instruction", "")

    if PROMPT_STYLE != "eval_json":
        return instruction

    source = record.get("source", "")

    # Global output discipline to reduce verbose/chatty generations.
    guardrail = (
        "\n\n请严格遵循以下输出要求：\n"
        "1. 只输出最终答案，不要输出分析、解释、思考过程。\n"
        "2. 不要输出多余前后缀，不要输出Markdown标记。\n"
        "3. 严格按题目要求的格式输出。"
    )

    # Task-specific format constraints for strict rule matching.
    if source == "BBH_logical":
        guardrail += "\n4. 最后一行只输出一个选项，格式必须是 (A) 或 (B) 或 (C) 等。"
    elif source == "BBH_time":
        guardrail += "\n4. 最后一行只输出日期，格式必须是 MM/DD/YYYY。"
    elif source == "gsm_8k":
        guardrail += "\n4. 最后一行只输出最终金额短语，格式形如 123美元。"
    elif source in {"E2E", "text_editing"}:
        guardrail += "\n4. 输出必须与目标格式完全一致，不要添加任何说明。"
    elif constraint_type == "example":
        guardrail += "\n4. 严格仿照题目中的示例模板输出。"

    return instruction + guardrail


def safe_filename(x):
    return str(x).replace("/", "_").replace(" ", "_")


def predict_text(tokenizer, model, prompt):
    messages = [{"role": "user", "content": prompt}]

    try:
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    except Exception:
        chat_text = prompt

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=GENERATION_CONFIG["do_sample"],
            temperature=GENERATION_CONFIG["temperature"],
            top_p=GENERATION_CONFIG["top_p"],
            top_k=GENERATION_CONFIG["top_k"],
            repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def evaluate_one(record, constraint_type, generation):
    level = int(record.get("level", 0))

    if constraint_type == "example":
        template = record.get("target", "").replace("{instruction}\n", "")
        return bool(check_match(template, generation)), "example_template"

    source = record.get("source", "")
    target = record.get("target", "")

    if source in RULE_BASED_SOURCES:
        fn_name = f"rule_evaluation_{source}"
        fn = getattr(rbe, fn_name, None)
        if fn is None:
            return False, f"missing_rule_fn:{fn_name}"
        return bool(fn(generation, target, level)), fn_name

    if (
        constraint_type == "format"
        and record.get("category") == "format"
        and record.get("example_id") in RULE_BASED_FORMAT_EXAMPLE_IDS
    ):
        return bool(rule_evaluation_format(generation, record.get("example_id"), level)), "rule_evaluation_format"

    return False, "not_rule_evaluable"


def compute_csl(details):
    grouped = defaultdict(dict)
    for d in details:
        key = (d["constraint_type"], d["example_id"])
        grouped[key][d["level"]] = 1 if d["is_correct"] else 0

    csl_by_type = defaultdict(list)
    for (constraint_type, _example_id), lv in grouped.items():
        if not all(x in lv for x in [1, 2, 3, 4, 5]):
            continue
        run = 0
        for k in [1, 2, 3, 4, 5]:
            if lv[k] == 1:
                run += 1
            else:
                break
        csl_by_type[constraint_type].append(run)

    rows = []
    for constraint_type in sorted(csl_by_type.keys()):
        arr = csl_by_type[constraint_type]
        avg = sum(arr) / len(arr) if arr else 0.0
        rows.append(
            {
                "model_name": MODEL_NAME,
                "constraint_type": constraint_type,
                "groups": len(arr),
                "csl": round(avg, 3),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Rule-only FollowBench-zh evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Model path to evaluate. Default uses script configuration MODEL_PATH.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Model name used in reports/file names. Default uses script configuration MODEL_NAME.",
    )
    parser.add_argument(
        "--small_sample_per_type",
        type=int,
        default=SMALL_SAMPLE_PER_TYPE,
        help="Randomly sample N rule-evaluable items per constraint type for quick testing. Default uses all items.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=SAMPLE_SEED,
        help="Random seed used when --small_sample_per_type is set.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Output directory for evaluation artifacts. Default: rule_eval_results_<MODEL_NAME>",
    )
    return parser.parse_args()


def main():
    global MODEL_PATH, MODEL_NAME

    args = parse_args()
    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name

    if args.small_sample_per_type is not None and args.small_sample_per_type <= 0:
        raise ValueError("--small_sample_per_type must be a positive integer.")

    save_dir = args.save_dir if args.save_dir else SAVE_DIR
    detail_csv = os.path.join(save_dir, f"rule_eval_details_{MODEL_NAME}.csv")
    summary_csv = os.path.join(save_dir, f"rule_eval_summary_{MODEL_NAME}.csv")
    level_csv = os.path.join(save_dir, f"rule_eval_by_level_{MODEL_NAME}.csv")
    source_csv = os.path.join(save_dir, f"rule_eval_by_source_{MODEL_NAME}.csv")
    csl_csv = os.path.join(save_dir, f"rule_eval_csl_{MODEL_NAME}.csv")
    gen_log_dir = os.path.join(save_dir, "generation_logs")

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    os.makedirs(save_dir, exist_ok=True)
    if SAVE_GENERATION_LOGS:
        os.makedirs(gen_log_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="left",
    )

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    model_kwargs["torch_dtype"] = torch.float16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Model loaded on: {DEVICE}")

    constraint_types = get_constraint_types()
    print(f"[INFO] Evaluating constraint types: {constraint_types}")
    if args.small_sample_per_type is not None:
        print(
            f"[INFO] Small-sample mode enabled: up to {args.small_sample_per_type} items per type "
            f"(seed={args.sample_seed})"
        )

    details = []
    skipped = []

    for constraint_type in constraint_types:
        data = load_constraint_data(constraint_type)

        eval_items = []
        for rec in data:
            ok, reason = is_rule_evaluable(rec, constraint_type)
            if ok:
                eval_items.append((rec, reason))
            elif rec.get("level", 0) > 0:
                skipped.append(
                    {
                        "constraint_type": constraint_type,
                        "source": rec.get("source", "<unknown>"),
                        "level": rec.get("level", 0),
                        "example_id": rec.get("example_id", -1),
                        "skip_reason": reason,
                    }
                )

        print(f"[INFO] {constraint_type}: {len(eval_items)} rule-evaluable items")

        if args.small_sample_per_type is not None and len(eval_items) > args.small_sample_per_type:
            rng = random.Random(args.sample_seed)
            eval_items = rng.sample(eval_items, args.small_sample_per_type)
            print(
                f"[INFO] {constraint_type}: sampled {len(eval_items)} items "
                f"from full set using seed {args.sample_seed}"
            )

        for idx, (rec, rule_reason) in enumerate(tqdm(eval_items, desc=f"Eval {constraint_type}", leave=False), 1):
            prompt = build_prompt(rec, constraint_type)
            generation = predict_text(tokenizer, model, prompt)
            is_correct, rule_name = evaluate_one(rec, constraint_type, generation)

            item = {
                "model_name": MODEL_NAME,
                "constraint_type": constraint_type,
                "source": rec.get("source", "example_constraints" if constraint_type == "example" else "<unknown>"),
                "category": rec.get("category", constraint_type),
                "example_id": rec.get("example_id", -1),
                "level": int(rec.get("level", 0)),
                "is_correct": bool(is_correct),
                "rule_name": rule_name,
                "rule_reason": rule_reason,
                "instruction": rec.get("instruction", ""),
                "target": rec.get("target", ""),
                "generation": generation,
            }
            details.append(item)

            if SAVE_GENERATION_LOGS:
                log_name = (
                    f"{constraint_type}_src_{safe_filename(item['source'])}_"
                    f"eid_{item['example_id']}_lvl_{item['level']}.txt"
                )
                with open(os.path.join(gen_log_dir, log_name), "w", encoding="utf-8") as f:
                    f.write(f"model_name: {MODEL_NAME}\n")
                    f.write(f"constraint_type: {constraint_type}\n")
                    f.write(f"source: {item['source']}\n")
                    f.write(f"example_id: {item['example_id']}\n")
                    f.write(f"level: {item['level']}\n")
                    f.write(f"rule_name: {rule_name}\n")
                    f.write(f"rule_reason: {rule_reason}\n")
                    f.write(f"is_correct: {item['is_correct']}\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"instruction:\n{item['instruction']}\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"target:\n{item['target']}\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"generation:\n{generation}\n")

    if not details:
        print("[WARN] No rule-evaluable items found. Nothing to save.")
        return

    df_details = pd.DataFrame(details)
    df_details.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    # Summary by constraint type
    summary_rows = []
    grouped_type = df_details.groupby("constraint_type")
    for ctype, g in grouped_type:
        total = len(g)
        correct = int(g["is_correct"].sum())
        acc = correct / total if total else 0.0
        summary_rows.append(
            {
                "model_name": MODEL_NAME,
                "constraint_type": ctype,
                "total": total,
                "correct": correct,
                "accuracy": f"{acc:.2%}",
            }
        )

    overall_total_summary = len(df_details)
    overall_correct_summary = int(df_details["is_correct"].sum())
    overall_acc_summary = (
        overall_correct_summary / overall_total_summary if overall_total_summary else 0.0
    )
    summary_rows.append(
        {
            "model_name": MODEL_NAME,
            "constraint_type": "overall",
            "total": overall_total_summary,
            "correct": overall_correct_summary,
            "accuracy": f"{overall_acc_summary:.2%}",
        }
    )

    df_summary = pd.DataFrame(summary_rows).sort_values(by=["constraint_type"])
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # Accuracy by constraint type and level
    level_rows = []
    grouped_lv = df_details.groupby(["constraint_type", "level"])
    for (ctype, lv), g in grouped_lv:
        total = len(g)
        correct = int(g["is_correct"].sum())
        acc = correct / total if total else 0.0
        level_rows.append(
            {
                "model_name": MODEL_NAME,
                "constraint_type": ctype,
                "level": int(lv),
                "total": total,
                "correct": correct,
                "accuracy": f"{acc:.2%}",
            }
        )
    pd.DataFrame(level_rows).sort_values(by=["constraint_type", "level"]).to_csv(
        level_csv, index=False, encoding="utf-8-sig"
    )

    # Accuracy by source
    source_rows = []
    grouped_src = df_details.groupby(["constraint_type", "source"])
    for (ctype, src), g in grouped_src:
        total = len(g)
        correct = int(g["is_correct"].sum())
        acc = correct / total if total else 0.0
        source_rows.append(
            {
                "model_name": MODEL_NAME,
                "constraint_type": ctype,
                "source": src,
                "total": total,
                "correct": correct,
                "accuracy": f"{acc:.2%}",
            }
        )
    pd.DataFrame(source_rows).sort_values(by=["constraint_type", "source"]).to_csv(
        source_csv, index=False, encoding="utf-8-sig"
    )

    # CSL for each constraint type (based on rule-evaluable groups)
    csl_rows = compute_csl(details)
    pd.DataFrame(csl_rows).to_csv(csl_csv, index=False, encoding="utf-8-sig")

    # Save skipped items report
    if skipped:
        skipped_path = os.path.join(save_dir, f"rule_eval_skipped_{MODEL_NAME}.csv")
        pd.DataFrame(skipped).to_csv(skipped_path, index=False, encoding="utf-8-sig")
    else:
        skipped_path = ""

    overall_total = len(df_details)
    overall_correct = int(df_details["is_correct"].sum())
    overall_acc = overall_correct / overall_total if overall_total else 0.0

    print("=" * 70)
    print("Rule-only evaluation finished")
    print(f"Model: {MODEL_NAME}")
    print(f"Total evaluated items: {overall_total}")
    print(f"Overall accuracy: {overall_acc:.2%}")
    print(f"Details: {os.path.abspath(detail_csv)}")
    print(f"Summary: {os.path.abspath(summary_csv)}")
    print(f"By-level: {os.path.abspath(level_csv)}")
    print(f"By-source: {os.path.abspath(source_csv)}")
    print(f"CSL: {os.path.abspath(csl_csv)}")
    if skipped_path:
        print(f"Skipped (need LLM judge): {os.path.abspath(skipped_path)}")
    if SAVE_GENERATION_LOGS:
        print(f"Generation logs: {os.path.abspath(gen_log_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
