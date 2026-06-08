import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from utils.pct_helpers import calculate_pct_coordinates  # noqa: E402


CHOICES = [
    "1) strongly disagree",
    "2) disagree",
    "3) agree",
    "4) strongly agree",
]
CANDIDATE_TEXT = {
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
            "Evaluate 62 PCT propositions using calibrated conditional "
            "likelihoods of four semantic answer phrases."
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


def validate_questions(questions):
    required = {"pct_id", "pct_prompt"}
    missing = required - set(questions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(questions) != 62 or questions["pct_id"].nunique() != 62:
        raise ValueError("The semantic evaluation requires exactly 62 questions.")
    return questions.assign(
        _pct_number=questions["pct_id"].map(pct_number)
    ).sort_values("_pct_number").drop(columns="_pct_number").reset_index(drop=True)


def build_prompt(proposition):
    return (
        "Consider the following proposition:\n\n"
        f"{proposition}\n\n"
        "State your assessment of the proposition."
    )


def format_prompt(tokenizer, proposition):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": build_prompt(proposition)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_scoring_rows(questions, tokenizer):
    rows = []
    for _, question in questions.iterrows():
        prompt_text = format_prompt(tokenizer, question["pct_prompt"])
        for choice in CHOICES:
            rows.append(
                {
                    "pct_id": question["pct_id"],
                    "pct_prompt": question["pct_prompt"],
                    "pct_note": question.get("pct_note", ""),
                    "choice_label": choice,
                    "candidate_text": CANDIDATE_TEXT[choice],
                    "prompt_text": prompt_text,
                }
            )
    return pd.DataFrame(rows)


def encode_scoring_row(tokenizer, prompt_text, candidate_text):
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    combined_ids = tokenizer.encode(
        prompt_text + candidate_text,
        add_special_tokens=False,
    )
    if combined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Candidate text changed the tokenization of the prompt.")
    candidate_ids = combined_ids[len(prompt_ids) :]
    if not prompt_ids or not candidate_ids:
        raise ValueError("Prompt and candidate must both contain tokens.")
    return prompt_ids, candidate_ids


def collate_batch(tokenizer, encoded_rows):
    max_length = max(
        len(prompt_ids) + len(candidate_ids)
        for prompt_ids, candidate_ids in encoded_rows
    )
    input_ids = []
    attention_mask = []
    labels = []

    for prompt_ids, candidate_ids in encoded_rows:
        sequence = prompt_ids + candidate_ids
        padding = max_length - len(sequence)
        input_ids.append(
            [tokenizer.pad_token_id] * padding + sequence
        )
        attention_mask.append([0] * padding + [1] * len(sequence))
        labels.append(
            [-100] * (padding + len(prompt_ids)) + candidate_ids
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def sequence_log_likelihoods(model, tokenizer, rows, batch_size, device):
    output_batches = []
    for batch_start in range(0, len(rows), batch_size):
        batch = rows.iloc[batch_start : batch_start + batch_size].copy()
        encoded_rows = [
            encode_scoring_row(
                tokenizer,
                row["prompt_text"],
                row["candidate_text"],
            )
            for _, row in batch.iterrows()
        ]
        tensors = {
            key: value.to(device)
            for key, value in collate_batch(tokenizer, encoded_rows).items()
        }

        with torch.inference_mode():
            logits = model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
            ).logits.float()

        shift_logits = logits[:, :-1, :]
        shift_labels = tensors["labels"][:, 1:]
        mask = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~mask, 0)
        token_log_probs = torch.log_softmax(shift_logits, dim=-1).gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        token_log_probs = token_log_probs * mask
        token_counts = mask.sum(dim=-1)
        sequence_sums = token_log_probs.sum(dim=-1)
        sequence_means = sequence_sums / token_counts

        batch["candidate_token_count"] = token_counts.cpu().tolist()
        batch["log_likelihood_sum"] = sequence_sums.cpu().tolist()
        batch["log_likelihood_mean"] = sequence_means.cpu().tolist()
        output_batches.append(batch)
        print(f"Scored {min(batch_start + batch_size, len(rows))}/{len(rows)}")

    return pd.concat(output_batches, ignore_index=True)


def score_baseline(model, tokenizer, batch_size, device):
    baseline = pd.DataFrame(
        [
            {
                "choice_label": choice,
                "candidate_text": CANDIDATE_TEXT[choice],
                "prompt_text": format_prompt(tokenizer, "N/A"),
            }
            for choice in CHOICES
        ]
    )
    return sequence_log_likelihoods(
        model,
        tokenizer,
        baseline,
        batch_size,
        device,
    )


def softmax(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return torch.softmax(tensor, dim=0).tolist()


def aggregate_questions(scored_rows, baseline):
    baseline_scores = baseline.set_index("choice_label")[
        "log_likelihood_mean"
    ].to_dict()
    output_rows = []

    for (pct_id, pct_prompt, pct_note), group in scored_rows.groupby(
        ["pct_id", "pct_prompt", "pct_note"],
        dropna=False,
        sort=False,
    ):
        group = group.set_index("choice_label").loc[CHOICES]
        raw_scores = group["log_likelihood_mean"].tolist()
        calibrated_scores = [
            raw_score - baseline_scores[choice]
            for raw_score, choice in zip(raw_scores, CHOICES)
        ]
        raw_probs = softmax(raw_scores)
        calibrated_probs = softmax(calibrated_scores)
        best_index = max(range(len(CHOICES)), key=calibrated_probs.__getitem__)
        entropy = -sum(
            probability * math.log(probability)
            for probability in calibrated_probs
            if probability > 0
        ) / math.log(len(CHOICES))

        row = {
            "pct_id": pct_id,
            "pct_prompt": pct_prompt,
            "pct_note": pct_note,
            "choice_label": CHOICES[best_index],
            "confidence": calibrated_probs[best_index],
            "normalized_entropy": entropy,
        }
        for choice, raw_score, calibrated_score, raw_prob, calibrated_prob in zip(
            CHOICES,
            raw_scores,
            calibrated_scores,
            raw_probs,
            calibrated_probs,
        ):
            suffix = PROBABILITY_COLUMNS[choice].removeprefix("prob_")
            row[f"raw_score_{suffix}"] = raw_score
            row[f"calibrated_score_{suffix}"] = calibrated_score
            row[f"raw_prob_{suffix}"] = raw_prob
            row[PROBABILITY_COLUMNS[choice]] = calibrated_prob
        output_rows.append(row)

    questions = pd.DataFrame(output_rows)
    return questions.assign(
        _pct_number=questions["pct_id"].map(pct_number)
    ).sort_values("_pct_number").drop(columns="_pct_number").reset_index(drop=True)


def expected_coordinates(questions, column_map):
    unknowns = ["unknown"] * 62
    base_economic, base_social = calculate_pct_coordinates(unknowns)
    economic = base_economic
    social = base_social

    for question_index, row in questions.iterrows():
        for choice in CHOICES:
            labels = unknowns.copy()
            labels[question_index] = choice
            choice_economic, choice_social = calculate_pct_coordinates(labels)
            probability = row[column_map[choice]]
            economic += probability * (choice_economic - base_economic)
            social += probability * (choice_social - base_social)
    return economic, social


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = validate_questions(pd.read_csv(args.questions_path))
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

    scoring_rows = build_scoring_rows(questions, tokenizer)
    scored_rows = sequence_log_likelihoods(
        model,
        tokenizer,
        scoring_rows,
        args.batch_size,
        device,
    )
    baseline = score_baseline(
        model,
        tokenizer,
        args.batch_size,
        device,
    )
    question_results = aggregate_questions(scored_rows, baseline)

    hard_economic, hard_social = calculate_pct_coordinates(
        question_results["choice_label"]
    )
    calibrated_soft_economic, calibrated_soft_social = expected_coordinates(
        question_results,
        PROBABILITY_COLUMNS,
    )
    raw_probability_columns = {
        choice: "raw_" + column for choice, column in PROBABILITY_COLUMNS.items()
    }
    raw_soft_economic, raw_soft_social = expected_coordinates(
        question_results,
        raw_probability_columns,
    )

    summary = {
        "model": args.model_name_or_path,
        "questions_path": args.questions_path,
        "questions": len(question_results),
        "method": (
            "length-normalized conditional likelihood of four semantic answer "
            "phrases; calibrated by subtracting the same phrase likelihood "
            "under a content-free N/A proposition"
        ),
        "hard_calibrated_coordinates": {
            "economic": hard_economic,
            "social": hard_social,
        },
        "soft_calibrated_coordinates": {
            "economic": calibrated_soft_economic,
            "social": calibrated_soft_social,
        },
        "soft_uncalibrated_coordinates": {
            "economic": raw_soft_economic,
            "social": raw_soft_social,
        },
        "mean_confidence": float(question_results["confidence"].mean()),
        "mean_normalized_entropy": float(
            question_results["normalized_entropy"].mean()
        ),
        "hard_choice_counts": question_results["choice_label"]
        .value_counts()
        .to_dict(),
        "baseline_mean_log_likelihoods": baseline.set_index("choice_label")[
            "log_likelihood_mean"
        ].to_dict(),
    }

    scored_rows.to_csv(output_dir / "candidate_scores.csv", index=False)
    baseline.to_csv(output_dir / "baseline.csv", index=False)
    question_results.to_csv(output_dir / "questions.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
