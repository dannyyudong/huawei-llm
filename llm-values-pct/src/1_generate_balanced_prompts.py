import argparse
from pathlib import Path

import pandas as pd


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
LETTERS = ["A", "B", "C", "D"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create four balanced A/B/C/D option-order variants for each "
            "explicit PCT prompt."
        )
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def build_prompt(row, ordered_choices):
    answer_options = "\n".join(
        f"{letter}) {DISPLAY_CHOICES[choice]}"
        for letter, choice in zip(LETTERS, ordered_choices)
    )
    return row["templ_prompt"].format(
        pct_prompt=row["pct_prompt"],
        ans_prompt=answer_options,
        jail_prompt="Answer with only A, B, C, or D.",
    )


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    df = pd.read_csv(input_path)

    required = {"templ_id", "templ_prompt", "pct_id", "pct_prompt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for _, row in df.iterrows():
        for offset in range(4):
            ordered_choices = CHOICES[offset:] + CHOICES[:offset]
            output_row = row.copy()
            output_row["balanced_order_id"] = f"order-{offset + 1}"
            for letter, choice in zip(LETTERS, ordered_choices):
                output_row[f"choice_{letter}"] = choice
            output_row["ans_id"] = f"balanced-order-{offset + 1}"
            output_row["ans_prompt"] = "\n".join(
                f"{letter}) {DISPLAY_CHOICES[choice]}"
                for letter, choice in zip(LETTERS, ordered_choices)
            )
            output_row["ans_note"] = "balanced A/B/C/D order"
            output_row["jail_id"] = "balanced-neutral"
            output_row["jail_prompt"] = "Answer with only A, B, C, or D."
            output_row["jail_note"] = "neutral"
            output_row["full_prompt"] = build_prompt(row, ordered_choices)
            rows.append(output_row)

    output_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(
        f"Wrote {len(output_df)} prompts to {output_path} "
        f"({len(df)} source prompts x 4 balanced orders)."
    )


if __name__ == "__main__":
    main()
