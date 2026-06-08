import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from utils.completion_helpers import extract_choice, validate_completion  # noqa: E402
from utils.pct_helpers import calculate_pct_coordinates  # noqa: E402


CHOICES = [
    "1) strongly disagree",
    "2) disagree",
    "3) agree",
    "4) strongly agree",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze explicit PCT multiple-choice completions."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument(
        "--output_dir",
        help="Defaults to <input directory>/<input stem>_analysis.",
    )
    return parser.parse_args()


def clean_completion(completion):
    text = "" if pd.isna(completion) else str(completion)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def pct_number(pct_id):
    match = re.search(r"(\d+)$", str(pct_id))
    if not match:
        raise ValueError(f"Cannot parse proposition number from {pct_id!r}")
    return int(match.group(1))


def analyze_templates(df):
    rows = []
    for template, template_df in sorted(df.groupby("templ_id")):
        ordered = template_df.assign(
            _pct_number=template_df["pct_id"].map(pct_number)
        ).sort_values("_pct_number")
        if len(ordered) != 62 or ordered["_pct_number"].nunique() != 62:
            raise ValueError(
                f"{template} must contain exactly one answer for each of 62 propositions."
            )
        x_coord, y_coord = calculate_pct_coordinates(
            ordered["choice_label"].reset_index(drop=True)
        )
        rows.append(
            {
                "templ_id": template,
                "n": len(ordered),
                "valid_rate": (ordered["validation_label"] == "valid").mean(),
                "recognized_choice_rate": (
                    ordered["choice_label"] != "unknown"
                ).mean(),
                "pct_economic": x_coord,
                "pct_social": y_coord,
            }
        )
    return pd.DataFrame(rows)


def analyze_stability(df):
    proposition_rows = []
    for pct_id, proposition_df in sorted(
        df.groupby("pct_id"),
        key=lambda item: pct_number(item[0]),
    ):
        recognized = proposition_df[
            proposition_df["choice_label"].isin(CHOICES)
        ]["choice_label"]
        counts = recognized.value_counts()
        proposition_rows.append(
            {
                "pct_id": pct_id,
                "recognized_n": len(recognized),
                "majority_choice": counts.index[0] if len(counts) else "unknown",
                "majority_share": (
                    counts.iloc[0] / len(recognized) if len(recognized) else None
                ),
                "unanimous": bool(len(counts) == 1 and len(recognized) > 0),
            }
        )
    return pd.DataFrame(proposition_rows)


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.parent / f"{input_path.stem}_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {"completion", "templ_id", "pct_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["clean_completion"] = df["completion"].map(clean_completion)
    df["validation_label"] = df["clean_completion"].map(validate_completion)
    df["choice_label"] = df["clean_completion"].map(extract_choice)

    template_df = analyze_templates(df)
    stability_df = analyze_stability(df)
    validation_counts = df["validation_label"].value_counts().to_dict()
    choice_counts = df["choice_label"].value_counts().to_dict()

    summary = {
        "input_path": str(input_path),
        "rows": len(df),
        "templates": int(df["templ_id"].nunique()),
        "propositions": int(df["pct_id"].nunique()),
        "validation_counts": validation_counts,
        "choice_counts": choice_counts,
        "valid_rate": float((df["validation_label"] == "valid").mean()),
        "recognized_choice_rate": float((df["choice_label"] != "unknown").mean()),
        "coordinate_range": {
            "economic_min": float(template_df["pct_economic"].min()),
            "economic_max": float(template_df["pct_economic"].max()),
            "social_min": float(template_df["pct_social"].min()),
            "social_max": float(template_df["pct_social"].max()),
        },
        "supplemental_stability": {
            "mean_majority_share": float(
                stability_df["majority_share"].dropna().mean()
            ),
            "unanimous_proposition_rate": float(stability_df["unanimous"].mean()),
        },
    }

    df.to_csv(output_dir / "rows.csv", index=False)
    template_df.to_csv(output_dir / "templates.csv", index=False)
    stability_df.to_csv(output_dir / "propositions.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Analysis written to {output_dir}")


if __name__ == "__main__":
    main()
