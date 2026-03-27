#!/usr/bin/env python3
"""Generate IFBench responses with local HuggingFace Transformers models."""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(input_file: str) -> list[dict]:
    prompts = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompts.append({"key": row["key"], "prompt": row["prompt"]})
    return prompts


def remove_think_block(text: str) -> str:
    start_tag = "<think>"
    end_tag = "</think>"
    if start_tag in text and end_tag in text:
        start = text.find(start_tag)
        end = text.find(end_tag, start)
        if end != -1:
            return (text[:start] + text[end + len(end_tag) :]).strip()
    return text.strip()


def build_inputs(tokenizer, prompt: str, enable_thinking=None):
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking

        try:
            text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
        except TypeError:
            # Backward compatibility for tokenizers without enable_thinking.
            chat_template_kwargs.pop("enable_thinking", None)
            text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)

        tokenized = tokenizer(text, return_tensors="pt")
        return {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
        }

    # Fallback for base tokenizers without chat template.
    tokenized = tokenizer(prompt, return_tensors="pt")
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses for IFBench using local HF models"
    )
    parser.add_argument("--model-path", required=True, help="Local model directory")
    parser.add_argument(
        "--input-file",
        default="data/IFBench_test.jsonl",
        help="Path to IFBench prompt file",
    )
    parser.add_argument(
        "--output-file",
        default="data/hf-local-responses.jsonl",
        help="Path to write responses jsonl",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p for sampling",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling; default is greedy decoding",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference: cuda/cpu",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--strip-think",
        action="store_true",
        help="Strip <think>...</think> blocks from outputs",
    )
    parser.add_argument(
        "--fix-mistral-regex",
        action="store_true",
        help="Pass fix_mistral_regex=True when loading tokenizer (if supported)",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking mode when supported by the tokenizer/model.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Qwen thinking mode when supported by the tokenizer/model.",
    )
    args = parser.parse_args()

    enable_thinking = None
    if args.enable_thinking:
        enable_thinking = True
    elif args.disable_thinking:
        enable_thinking = False

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, "auto")

    tokenizer_kwargs = {"trust_remote_code": True}
    if args.fix_mistral_regex:
        tokenizer_kwargs["fix_mistral_regex"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, **tokenizer_kwargs)
    except TypeError:
        # Backward compatibility for transformers versions without fix_mistral_regex.
        tokenizer_kwargs.pop("fix_mistral_regex", None)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()

    prompts = load_prompts(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(prompts, desc="Generating"):
            prompt = item["prompt"]
            model_inputs = build_inputs(
                tokenizer, prompt, enable_thinking=enable_thinking
            )
            input_ids = model_inputs["input_ids"].to(args.device)
            attention_mask = model_inputs["attention_mask"].to(args.device)

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "attention_mask": attention_mask,
            }
            if args.do_sample:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p

            with torch.no_grad():
                out = model.generate(input_ids, **gen_kwargs)

            new_tokens = out[0][input_ids.shape[-1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if args.strip_think:
                response = remove_think_block(response)

            f.write(
                json.dumps(
                    {"prompt": prompt, "response": response}, ensure_ascii=False
                )
                + "\n"
            )

    print(f"Saved {len(prompts)} responses to {output_path}")
    print("Run eval with:")
    print(
        "python -m run_eval "
        f"--input_data={args.input_file} "
        f"--input_response_data={args.output_file} "
        "--output_dir=eval"
    )


if __name__ == "__main__":
    main()
