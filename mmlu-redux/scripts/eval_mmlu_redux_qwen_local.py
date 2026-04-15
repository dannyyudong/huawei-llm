import argparse
import ast
import json
import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


PROMPT_INSTRUCTION = (
    "You are solving a multiple-choice question. "
    "Choose exactly one option from A, B, C, D. "
    "Output must follow JSON format exactly as: {\"answer\": \"A\"}. "
    "Do not include explanations or extra fields."
)


def parse_choices(value) -> List[str]:
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]

    if isinstance(value, list):
        return [str(x) for x in value]

    if isinstance(value, tuple):
        return [str(x) for x in value]

    if isinstance(value, str):
        text = value.strip()

        # Common path: a serialized Python list with commas.
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass

        # Fallback path: ndarray-like formatting without commas between items,
        # e.g. "['a'\n 'b'\n 'c'\n 'd']".
        quoted_items = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)
        flattened = [a if a else b for a, b in quoted_items]
        if flattened:
            return [str(x) for x in flattened]

    raise ValueError(f"Invalid choices format: {value}")


def build_user_prompt(subject: str, question: str, choices: List[str]) -> str:
    letters = ["A", "B", "C", "D"]
    lines = [
        f"The following is a multiple-choice question in subject: {subject}",
        "",
        f"Question: {question}",
        "Choices:",
    ]
    for i, choice in enumerate(choices):
        lines.append(f"{letters[i]}. {choice}")
    lines.extend(
        [
            "",
            "Return only a JSON object with one key 'answer'.",
            "Example: {\"answer\": \"C\"}",
            "Your response:",
        ]
    )
    return "\n".join(lines)


def extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None

    text = text.strip()

    # If the model returns reasoning tags, focus on the final answer segment.
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    exact = re.fullmatch(r"[ABCD]", text, flags=re.IGNORECASE)
    if exact:
        return exact.group(0).upper()

    # Try to parse a JSON answer such as {"answer": "C"}
    json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', text, flags=re.IGNORECASE)
    if json_match:
        try:
            payload = json_match.group(0).replace("'", '"')
            answer = json.loads(payload).get("answer", "")
            m = re.search(r"\b([ABCD])\b", str(answer), flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
        except json.JSONDecodeError:
            pass

    patterns = [
        r"\b([ABCD])\b",
        r"option\s*([ABCD])",
        r"answer\s*[:：]\s*([ABCD])",
        r"^\(?([ABCD])\)?[\.|:|\s]",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback: choose the last standalone A/B/C/D in the output.
    all_matches = re.findall(r"\b([ABCD])\b", text, flags=re.IGNORECASE)
    if all_matches:
        return all_matches[-1].upper()

    return None


def build_choice_token_ids(tokenizer) -> Dict[str, List[int]]:
    token_ids: Dict[str, List[int]] = {"A": [], "B": [], "C": [], "D": []}
    for letter in ["A", "B", "C", "D"]:
        for candidate in [letter, f" {letter}"]:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                token_ids[letter].append(ids[0])
        token_ids[letter] = sorted(set(token_ids[letter]))
    return token_ids


def choose_letter_from_logits(model, input_ids, attention_mask, choice_token_ids) -> Optional[str]:
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    next_token_logits = out.logits[0, -1, :]

    best_letter = None
    best_score = None
    for letter, ids in choice_token_ids.items():
        if not ids:
            continue
        score = torch.max(next_token_logits[ids]).item()
        if best_score is None or score > best_score:
            best_score = score
            best_letter = letter
    return best_letter


def answer_idx_to_letter(idx: int) -> str:
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    if idx not in mapping:
        raise ValueError(f"Unexpected answer index: {idx}")
    return mapping[idx]


def load_subject_dataframe(subject_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(subject_dir, "test.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    # Fallback for snapshot layouts where only Arrow is available.
    from datasets import load_from_disk

    dataset = load_from_disk(subject_dir)
    return dataset.to_pandas()


def build_generation_inputs(
    tokenizer,
    subject: str,
    question: str,
    choices: List[str],
    use_chat_template: bool,
    enable_thinking: bool,
    device: torch.device,
):
    user_prompt = build_user_prompt(subject, question, choices)

    input_ids = None
    attention_mask = None

    if use_chat_template:
        messages = [
            {"role": "system", "content": PROMPT_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ]
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )

        # Some transformers versions return a Tensor, others return BatchEncoding.
        if isinstance(templated, torch.Tensor):
            input_ids = templated.to(device)
        else:
            input_ids = templated["input_ids"].to(device)
            if "attention_mask" in templated:
                attention_mask = templated["attention_mask"].to(device)
    else:
        plain_prompt = f"{PROMPT_INSTRUCTION}\n\n{user_prompt}"
        encoded = tokenizer(plain_prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        if "attention_mask" in encoded:
            attention_mask = encoded.attention_mask.to(device)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids).to(device)
    return input_ids, attention_mask


def evaluate_subject(
    subject: str,
    subject_dir: str,
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    use_chat_template: bool,
    enable_thinking: bool,
    generation_config: Dict[str, float],
    limit: Optional[int],
    show_progress: bool,
    save_raw_outputs: bool,
    raw_output_dir: Optional[str],
    force_choice_only: bool,
    choice_token_ids: Optional[Dict[str, List[int]]],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = load_subject_dataframe(subject_dir)

    if limit is not None:
        df = df.head(limit).copy()

    results = []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"{subject}",
        leave=False,
        disable=not show_progress,
    ):
        question = str(row["question"])
        choices = parse_choices(row["choices"])
        answer_idx = int(row["answer"])
        gold_letter = answer_idx_to_letter(answer_idx)

        input_ids, attention_mask = build_generation_inputs(
            tokenizer,
            subject,
            question,
            choices,
            use_chat_template,
            enable_thinking,
            device,
        )

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pad_token_id": pad_token_id,
            "max_new_tokens": max_new_tokens,
            "do_sample": generation_config["do_sample"],
            "num_return_sequences": 1,
            "repetition_penalty": generation_config["repetition_penalty"],
        }
        if generation_config["no_repeat_ngram_size"] > 0:
            generate_kwargs["no_repeat_ngram_size"] = generation_config[
                "no_repeat_ngram_size"
            ]
        if generation_config["do_sample"]:
            generate_kwargs["temperature"] = generation_config["temperature"]
            generate_kwargs["top_p"] = generation_config["top_p"]
            generate_kwargs["top_k"] = generation_config["top_k"]

        if force_choice_only and choice_token_ids is not None:
            pred_letter = choose_letter_from_logits(
                model,
                input_ids,
                attention_mask,
                choice_token_ids,
            )
            pred_text = pred_letter if pred_letter is not None else ""
        else:
            with torch.no_grad():
                output = model.generate(**generate_kwargs)

            pred_text = tokenizer.decode(
                output[0, input_ids.shape[1] :],
                skip_special_tokens=True,
            ).strip()
            pred_letter = extract_choice_letter(pred_text)

        is_correct = pred_letter == gold_letter

        result_row = {
            "subject": subject,
            "question": question,
            "choices": json.dumps(choices, ensure_ascii=False),
            "gold_answer_idx": answer_idx,
            "gold_answer_letter": gold_letter,
            "prediction_text": pred_text,
            "prediction_letter": pred_letter,
            "is_correct": is_correct,
        }

        if "error_type" in row:
            result_row["error_type"] = row["error_type"]

        if save_raw_outputs and raw_output_dir:
            sample_id = row.get("id", len(results))
            log_path = os.path.join(raw_output_dir, f"{subject}_q{sample_id}.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"Subject: {subject}\n")
                f.write(f"Question ID: {sample_id}\n")
                f.write(f"Gold: {gold_letter}\n")
                f.write(f"Pred: {pred_letter}\n")
                f.write("\n=== Prompt ===\n")
                f.write(build_user_prompt(subject, question, choices) + "\n")
                f.write("\n=== Output ===\n")
                f.write(pred_text + "\n")

        results.append(result_row)

    result_df = pd.DataFrame(results)
    accuracy = float(result_df["is_correct"].mean()) if len(result_df) else 0.0

    summary = {
        "subject": subject,
        "num_samples": int(len(result_df)),
        "accuracy": accuracy,
    }
    return result_df, summary


def select_subjects(dataset_root: str, subjects_arg: str) -> List[str]:
    all_subjects = sorted(
        [
            d
            for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith(".")
        ]
    )

    if subjects_arg.lower() == "all":
        return all_subjects

    requested = [s.strip() for s in subjects_arg.split(",") if s.strip()]
    missing = [s for s in requested if s not in all_subjects]
    if missing:
        raise ValueError(f"Unknown subjects: {missing}. Available: {all_subjects}")

    return requested


def main(args):
    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    subjects = select_subjects(dataset_root, args.subjects)

    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = dtype_mapping[args.torch_dtype]
    if device.type == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        model_dtype = torch.float32

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        fix_mistral_regex=args.fix_mistral_regex,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    choice_token_ids = None
    if args.force_choice_only:
        choice_token_ids = build_choice_token_ids(tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    raw_output_dir = None
    if args.save_raw_outputs:
        raw_output_dir = os.path.join(args.output_dir, "raw_outputs")
        os.makedirs(raw_output_dir, exist_ok=True)

    generation_config = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }

    all_rows = []
    subject_summaries = []

    subject_iter = tqdm(
        subjects,
        desc="Subjects",
        disable=not args.show_progress,
    )

    for subject in subject_iter:
        subject_dir = os.path.join(dataset_root, subject)
        print(f"Evaluating subject: {subject}")

        subject_df, subject_summary = evaluate_subject(
            subject=subject,
            subject_dir=subject_dir,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=args.use_chat_template,
            enable_thinking=args.enable_thinking,
            generation_config=generation_config,
            limit=args.limit,
            show_progress=args.show_progress,
            save_raw_outputs=args.save_raw_outputs,
            raw_output_dir=raw_output_dir,
            force_choice_only=args.force_choice_only,
            choice_token_ids=choice_token_ids,
        )

        all_rows.append(subject_df)
        subject_summaries.append(subject_summary)

    predictions_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(subject_summaries)

    overall_accuracy = (
        float(predictions_df["is_correct"].mean()) if len(predictions_df) else 0.0
    )

    summary = {
        "model_path": args.model_path,
        "dataset_root": dataset_root,
        "num_subjects": len(subjects),
        "num_samples": int(len(predictions_df)),
        "overall_accuracy": overall_accuracy,
        "macro_accuracy": float(summary_df["accuracy"].mean()) if len(summary_df) else 0.0,
        "use_chat_template": args.use_chat_template,
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": args.torch_dtype,
        "generation_config": generation_config,
        "enable_thinking": args.enable_thinking,
        "force_choice_only": args.force_choice_only,
    }

    if "error_type" in predictions_df.columns:
        et = (
            predictions_df.groupby("error_type", dropna=False)["is_correct"]
            .mean()
            .sort_values(ascending=False)
        )
        summary["accuracy_by_error_type"] = {k: float(v) for k, v in et.items()}

    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    per_subject_path = os.path.join(args.output_dir, "metrics_per_subject.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")

    predictions_df.to_csv(predictions_path, index=False)
    summary_df.to_csv(per_subject_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Macro accuracy:   {summary['macro_accuracy']:.4f}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved per-subject metrics: {per_subject_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate local Qwen models on local MMLU-Redux dataset"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local Hugging Face model path (e.g. /home/huawei/huawei/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./data/mmlu-redux",
        help="Local MMLU-Redux root directory",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="all",
        help="Comma-separated subject names or 'all'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/mmlu_redux_qwen_eval",
        help="Directory to save predictions and metrics",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Model loading dtype",
    )
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use tokenizer chat template",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code when loading tokenizer/model",
    )
    parser.add_argument(
        "--fix-mistral-regex",
        action="store_true",
        dest="fix_mistral_regex",
        help="Pass fix_mistral_regex=True when loading tokenizer",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples per subject for quick test",
    )
    parser.add_argument(
        "--show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars for subjects and samples",
    )
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model thinking mode when chat template supports it",
    )
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable sampling for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (used when do_sample=true)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling (used when do_sample=true)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling (used when do_sample=true)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for generation",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="No-repeat ngram size (0 to disable)",
    )
    parser.add_argument(
        "--save_raw_outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-question prompt/output logs under output_dir/raw_outputs",
    )
    parser.add_argument(
        "--force_choice_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Avoid free-form generation and directly choose one of A/B/C/D from next-token logits.",
    )

    args = parser.parse_args()
    main(args)
