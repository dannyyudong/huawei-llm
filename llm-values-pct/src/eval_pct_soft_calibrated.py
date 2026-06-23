import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

import eval_pct_semantic as semantic


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the PCT soft-calibrated evaluation standard. This is the "
            "recommended non-thinking diagnostic when generated labels show "
            "position bias."
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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = semantic.validate_questions(pd.read_csv(args.questions_path))
    device, dtype = semantic.select_device_and_dtype(args.require_cuda)
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

    scoring_rows = semantic.build_scoring_rows(questions, tokenizer)
    candidate_scores = semantic.sequence_log_likelihoods(
        model,
        tokenizer,
        scoring_rows,
        args.batch_size,
        device,
    )
    baseline = semantic.score_baseline(
        model,
        tokenizer,
        args.batch_size,
        device,
    )
    question_results = semantic.aggregate_questions(candidate_scores, baseline)

    soft_economic, soft_social = semantic.expected_coordinates(
        question_results,
        semantic.PROBABILITY_COLUMNS,
    )
    hard_economic, hard_social = semantic.calculate_pct_coordinates(
        question_results["choice_label"]
    )

    summary = {
        "standard": "soft_calibrated_coordinates",
        "model": args.model_name_or_path,
        "questions_path": args.questions_path,
        "questions": len(question_results),
        "method": (
            "For each proposition, compare length-normalized conditional "
            "likelihoods of the four semantic phrases: Strongly disagree, "
            "Disagree, Agree, Strongly agree. Subtract each phrase's "
            "content-free N/A baseline score, softmax the calibrated scores, "
            "then compute probability-weighted Political Compass coordinates."
        ),
        "soft_calibrated_coordinates": {
            "economic": soft_economic,
            "social": soft_social,
        },
        "reference_hard_calibrated_coordinates": {
            "economic": hard_economic,
            "social": hard_social,
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

    candidate_scores.to_csv(output_dir / "candidate_scores.csv", index=False)
    baseline.to_csv(output_dir / "baseline.csv", index=False)
    question_results.to_csv(output_dir / "questions.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
