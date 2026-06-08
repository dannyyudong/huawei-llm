import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate completions with a local Hugging Face chat model."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--test_data_input_path", required=True)
    parser.add_argument("--test_data_output_path", required=True)
    parser.add_argument("--input_col", default="full_prompt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--n_test_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode. Disabled by default for label extraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output instead of resuming it.",
    )
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Exit instead of falling back to CPU when CUDA is unavailable.",
    )
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


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


def prepare_dataframe(args):
    input_df = pd.read_csv(args.test_data_input_path)
    if args.n_test_samples > 0:
        input_df = input_df.sample(
            n=min(args.n_test_samples, len(input_df)),
            random_state=args.seed,
        )
    input_df = input_df.reset_index(drop=True)

    output_path = Path(args.test_data_output_path)
    if output_path.exists() and not args.overwrite:
        output_df = pd.read_csv(output_path)
        if len(output_df) > len(input_df):
            raise ValueError("Existing output has more rows than the selected input.")
        expected = input_df.iloc[: len(output_df)][args.input_col].tolist()
        actual = output_df[args.input_col].tolist()
        if expected != actual:
            raise ValueError(
                "Existing output does not match the selected input. "
                "Use --overwrite to start again."
            )
        logging.info("Resuming after %d completed rows", len(output_df))
        return input_df, output_df

    output_df = input_df.iloc[:0].copy()
    output_df["completion"] = pd.Series(dtype="object")
    output_df["model"] = pd.Series(dtype="object")
    return input_df, output_df


def format_prompts(tokenizer, prompts, enable_thinking):
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
    return texts


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    torch.manual_seed(args.seed)

    input_df, output_df = prepare_dataframe(args)
    output_path = Path(args.test_data_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device, dtype = select_device_and_dtype(args.require_cuda)
    logging.info("Loading %s on %s with %s", args.model_name_or_path, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
    ).to(device)
    model.eval()

    start = len(output_df)
    for batch_start in range(start, len(input_df), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(input_df))
        batch_df = input_df.iloc[batch_start:batch_end].copy()
        texts = format_prompts(
            tokenizer,
            batch_df[args.input_col].tolist(),
            args.enable_thinking,
        )
        model_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[:, model_inputs["input_ids"].shape[1] :]
        batch_df["completion"] = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )
        batch_df["model"] = args.model_name_or_path
        output_df = pd.concat([output_df, batch_df], ignore_index=True)
        output_df.to_csv(output_path, index=False)
        logging.info("Saved %d/%d completions", len(output_df), len(input_df))


if __name__ == "__main__":
    main()
