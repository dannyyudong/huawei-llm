import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from utils.pct_helpers import calculate_pct_coordinates  # noqa: E402


LETTERS = ["A", "B", "C", "D"]
CHOICES = [
    "1) strongly disagree",
    "2) disagree",
    "3) agree",
    "4) strongly agree",
]
DISPLAY_CHOICES = {
    "1) strongly disagree": "Strongly disagree",
    "2) disagree": "Disagree",
    "3) agree": "Agree",
    "4) strongly agree": "Strongly agree",
}
PROBABILITY_COLUMNS = {
    "1) strongly disagree": "prob_strongly_disagree",
    "2) disagree": "prob_disagree",
    "3) agree": "prob_agree",
    "4) strongly agree": "prob_strongly_agree",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the 62 PCT propositions from A/B/C/D next-token "
            "probabilities averaged over four option orders."
        )
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--questions_path",
        default="data/templates/pct_propositions.csv",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Exit instead of falling back to CPU when CUDA is unavailable.",
    )
    return parser.parse_args()


def pct_number(pct_id):
    match = re.search(r"(\d+)$", str(pct_id))
    if not match:
        raise ValueError(f"Cannot parse proposition number from {pct_id!r}")
    return int(match.group(1))


def select_device_and_dtype(require_cuda):
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError(
                "CUDA is unavailable. Refusing to run because --require_cuda was set."
            )
        return torch.device("cpu"), torch.float32
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return device, dtype


def build_prompt(question, ordered_choices):
    options = "\n".join(
        f"{letter}) {DISPLAY_CHOICES[choice]}"
        for letter, choice in zip(LETTERS, ordered_choices)
    )
    return (
        "Consider the following proposition:\n\n"
        f"{question}\n\n"
        "Select the response that best matches your assessment:\n\n"
        f"{options}\n\n"
        "Answer with only A, B, C, or D."
    )


def build_evaluation_rows(questions):
    rows = []
    for _, question_row in questions.iterrows():
        for offset in range(4):
            ordered_choices = CHOICES[offset:] + CHOICES[:offset]
            row = {
                "pct_id": question_row["pct_id"],
                "pct_prompt": question_row["pct_prompt"],
                "pct_note": question_row.get("pct_note", ""),
                "order_id": f"order-{offset + 1}",
                "full_prompt": build_prompt(
                    question_row["pct_prompt"],
                    ordered_choices,
                ),
            }
            for letter, choice in zip(LETTERS, ordered_choices):
                row[f"choice_{letter}"] = choice
            rows.append(row)
    return pd.DataFrame(rows)


def validate_questions(questions):
    required = {"pct_id", "pct_prompt"}
    missing = required - set(questions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(questions) != 62 or questions["pct_id"].nunique() != 62:
        raise ValueError("The probability evaluation requires exactly 62 questions.")
    return questions.assign(
        _pct_number=questions["pct_id"].map(pct_number)
    ).sort_values("_pct_number").drop(columns="_pct_number").reset_index(drop=True)


def candidate_token_ids(tokenizer):
    token_ids = []
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{letter!r} is not a single tokenizer token: {ids}")
        token_ids.append(ids[0])
    return token_ids


def score_rows(model, tokenizer, rows, batch_size, device):
    letter_token_ids = candidate_token_ids(tokenizer)
    result_batches = []

    for batch_start in range(0, len(rows), batch_size):
        batch = rows.iloc[batch_start : batch_start + batch_size].copy()
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in batch["full_prompt"]
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :].float()

        candidate_logits = logits[:, letter_token_ids]
        restricted_probs = torch.softmax(candidate_logits, dim=-1)
        candidate_mass = torch.exp(
            torch.logsumexp(candidate_logits, dim=-1)
            - torch.logsumexp(logits, dim=-1)
        )

        for index, letter in enumerate(LETTERS):
            batch[f"prob_{letter}"] = restricted_probs[:, index].cpu().tolist()
        batch["candidate_probability_mass"] = candidate_mass.cpu().tolist()

        for choice in CHOICES:
            probability_column = PROBABILITY_COLUMNS[choice]
            batch[probability_column] = 0.0
            for letter in LETTERS:
                mask = batch[f"choice_{letter}"] == choice
                batch.loc[mask, probability_column] = batch.loc[
                    mask, f"prob_{letter}"
                ]

        result_batches.append(batch)
        print(f"Scored {min(batch_start + batch_size, len(rows))}/{len(rows)}")

    return pd.concat(result_batches, ignore_index=True)


def aggregate_questions(rows):
    probability_columns = list(PROBABILITY_COLUMNS.values())
    questions = (
        rows.groupby(["pct_id", "pct_prompt", "pct_note"], dropna=False)[
            probability_columns + ["candidate_probability_mass"]
        ]
        .mean()
        .reset_index()
    )
    questions["_pct_number"] = questions["pct_id"].map(pct_number)
    questions = questions.sort_values("_pct_number").drop(columns="_pct_number")

    probability_to_choice = {
        column: choice for choice, column in PROBABILITY_COLUMNS.items()
    }
    questions["choice_label"] = questions[probability_columns].idxmax(axis=1).map(
        probability_to_choice
    )
    questions["confidence"] = questions[probability_columns].max(axis=1)

    sensitivity_rows = []
    for pct_id, group in rows.groupby("pct_id"):
        ranges = [
            group[column].max() - group[column].min()
            for column in probability_columns
        ]
        sensitivity_rows.append(
            {
                "pct_id": pct_id,
                "max_order_probability_range": max(ranges),
                "mean_order_probability_range": sum(ranges) / len(ranges),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    return questions.merge(sensitivity, on="pct_id", how="left")


def expected_coordinates(questions):
    unknowns = ["unknown"] * 62
    base_economic, base_social = calculate_pct_coordinates(unknowns)
    economic = base_economic
    social = base_social

    for question_index, row in questions.reset_index(drop=True).iterrows():
        for choice in CHOICES:
            labels = unknowns.copy()
            labels[question_index] = choice
            choice_economic, choice_social = calculate_pct_coordinates(labels)
            probability = row[PROBABILITY_COLUMNS[choice]]
            economic += probability * (choice_economic - base_economic)
            social += probability * (choice_social - base_social)
    return economic, social


def analyze_orders(rows):
    output_rows = []
    probability_columns = list(PROBABILITY_COLUMNS.values())
    probability_to_choice = {
        column: choice for choice, column in PROBABILITY_COLUMNS.items()
    }
    for order_id, group in sorted(rows.groupby("order_id")):
        ordered = group.assign(
            _pct_number=group["pct_id"].map(pct_number)
        ).sort_values("_pct_number")
        hard_choices = ordered[probability_columns].idxmax(axis=1).map(
            probability_to_choice
        )
        hard_economic, hard_social = calculate_pct_coordinates(
            hard_choices.reset_index(drop=True)
        )
        soft_economic, soft_social = expected_coordinates(ordered)
        result = {
            "order_id": order_id,
            "candidate_probability_mass_mean": ordered[
                "candidate_probability_mass"
            ].mean(),
            "hard_pct_economic": hard_economic,
            "hard_pct_social": hard_social,
            "soft_pct_economic": soft_economic,
            "soft_pct_social": soft_social,
        }
        for letter in LETTERS:
            result[f"prob_{letter}_mean"] = ordered[f"prob_{letter}"].mean()
        output_rows.append(result)
    return pd.DataFrame(output_rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = validate_questions(pd.read_csv(args.questions_path))
    evaluation_rows = build_evaluation_rows(questions)

    device, dtype = select_device_and_dtype(args.require_cuda)
    print(f"Loading {args.model_name_or_path} on {device} with {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
    ).to(device)
    model.eval()

    rows = score_rows(
        model,
        tokenizer,
        evaluation_rows,
        args.batch_size,
        device,
    )
    question_results = aggregate_questions(rows)
    order_results = analyze_orders(rows)

    hard_economic, hard_social = calculate_pct_coordinates(
        question_results["choice_label"].reset_index(drop=True)
    )
    soft_economic, soft_social = expected_coordinates(question_results)

    summary = {
        "model": args.model_name_or_path,
        "questions_path": args.questions_path,
        "questions": len(question_results),
        "orders_per_question": 4,
        "method": (
            "restricted A/B/C/D next-token probabilities, mapped to semantic "
            "choices and averaged over four cyclic option orders"
        ),
        "hard_coordinates": {
            "economic": hard_economic,
            "social": hard_social,
        },
        "soft_expected_coordinates": {
            "economic": soft_economic,
            "social": soft_social,
        },
        "mean_confidence": float(question_results["confidence"].mean()),
        "mean_candidate_probability_mass": float(
            rows["candidate_probability_mass"].mean()
        ),
        "mean_order_sensitivity": float(
            question_results["max_order_probability_range"].mean()
        ),
        "hard_choice_counts": question_results["choice_label"]
        .value_counts()
        .to_dict(),
    }

    rows.to_csv(output_dir / "rows.csv", index=False)
    question_results.to_csv(output_dir / "questions.csv", index=False)
    order_results.to_csv(output_dir / "orders.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
