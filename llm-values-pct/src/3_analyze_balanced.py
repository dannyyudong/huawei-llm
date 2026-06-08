import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze balanced A/B/C/D PCT completions."
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


def extract_letter(completion):
    text = clean_completion(completion).upper()
    match = re.match(r"^\s*(?:OPTION\s*)?[\(\[]?([ABCD])[\)\].,:;\s-]*", text)
    return match.group(1) if match else "unknown"


def pct_number(pct_id):
    match = re.search(r"(\d+)$", str(pct_id))
    if not match:
        raise ValueError(f"Cannot parse proposition number from {pct_id!r}")
    return int(match.group(1))


def map_choice(row):
    if row["answer_letter"] not in LETTERS:
        return "unknown"
    return row[f"choice_{row['answer_letter']}"]


def calculate_group_coordinate(group, group_name):
    ordered = group.assign(
        _pct_number=group["pct_id"].map(pct_number)
    ).sort_values("_pct_number")
    if len(ordered) != 62 or ordered["_pct_number"].nunique() != 62:
        raise ValueError(
            f"{group_name} must contain exactly one answer for each of 62 propositions."
        )
    x_coord, y_coord = calculate_pct_coordinates(
        ordered["choice_label"].reset_index(drop=True)
    )
    return {
        "n": len(ordered),
        "recognized_rate": (ordered["choice_label"] != "unknown").mean(),
        "pct_economic": x_coord,
        "pct_social": y_coord,
    }


def analyze_rotations(df):
    rows = []
    for (template, order_id), group in sorted(
        df.groupby(["templ_id", "balanced_order_id"])
    ):
        result = calculate_group_coordinate(
            group,
            f"{template}/{order_id}",
        )
        rows.append(
            {
                "templ_id": template,
                "balanced_order_id": order_id,
                **result,
            }
        )
    return pd.DataFrame(rows)


def analyze_templates(rotation_df):
    rows = []
    for template, group in sorted(rotation_df.groupby("templ_id")):
        rows.append(
            {
                "templ_id": template,
                "orders": len(group),
                "recognized_rate_mean": group["recognized_rate"].mean(),
                "pct_economic_mean": group["pct_economic"].mean(),
                "pct_economic_min": group["pct_economic"].min(),
                "pct_economic_max": group["pct_economic"].max(),
                "pct_economic_std": group["pct_economic"].std(ddof=0),
                "pct_social_mean": group["pct_social"].mean(),
                "pct_social_min": group["pct_social"].min(),
                "pct_social_max": group["pct_social"].max(),
                "pct_social_std": group["pct_social"].std(ddof=0),
            }
        )
    return pd.DataFrame(rows)


def analyze_position_bias(df):
    rows = []
    for order_id, group in sorted(df.groupby("balanced_order_id")):
        counts = group["answer_letter"].value_counts()
        row = {
            "balanced_order_id": order_id,
            "recognized_rate": (group["answer_letter"] != "unknown").mean(),
        }
        for letter in LETTERS:
            row[f"answer_{letter}_rate"] = counts.get(letter, 0) / len(group)
        rows.append(row)
    return pd.DataFrame(rows)


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
    required = {
        "completion",
        "templ_id",
        "pct_id",
        "balanced_order_id",
        "choice_A",
        "choice_B",
        "choice_C",
        "choice_D",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["clean_completion"] = df["completion"].map(clean_completion)
    df["answer_letter"] = df["clean_completion"].map(extract_letter)
    df["choice_label"] = df.apply(map_choice, axis=1)

    rotation_df = analyze_rotations(df)
    template_df = analyze_templates(rotation_df)
    position_df = analyze_position_bias(df)

    summary = {
        "input_path": str(input_path),
        "rows": len(df),
        "templates": int(df["templ_id"].nunique()),
        "propositions": int(df["pct_id"].nunique()),
        "orders": int(df["balanced_order_id"].nunique()),
        "recognized_rate": float((df["answer_letter"] != "unknown").mean()),
        "answer_letter_counts": df["answer_letter"].value_counts().to_dict(),
        "mapped_choice_counts": df["choice_label"].value_counts().to_dict(),
        "template_mean_coordinate_range": {
            "economic_min": float(template_df["pct_economic_mean"].min()),
            "economic_max": float(template_df["pct_economic_mean"].max()),
            "social_min": float(template_df["pct_social_mean"].min()),
            "social_max": float(template_df["pct_social_mean"].max()),
        },
    }

    df.to_csv(output_dir / "rows.csv", index=False)
    rotation_df.to_csv(output_dir / "orders.csv", index=False)
    template_df.to_csv(output_dir / "templates.csv", index=False)
    position_df.to_csv(output_dir / "position_bias.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Analysis written to {output_dir}")


if __name__ == "__main__":
    main()
