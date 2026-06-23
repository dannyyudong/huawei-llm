import json
import math
import os
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import dataloader
from evaluation import ScoreEvaluator


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Local path or HF id for Qwen/Qwen quantized model.")
    parser.add_argument("--input-file", default="../data/dev.json")
    parser.add_argument("--output-dir", default="predictions/")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--results-output-file", default=None)
    parser.add_argument("--results-model-name", default=None)
    parser.add_argument("--skip-scoring", default=False, action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-cuda", default=False, action="store_true")
    parser.add_argument("--device-map", default=None, help="Pass 'auto' for large or quantized HF models.")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--load-in-8bit", default=False, action="store_true")
    parser.add_argument("--load-in-4bit", default=False, action="store_true")
    parser.add_argument("--bnb-4bit-compute-dtype", choices=["float32", "bfloat16", "float16"], default="float16")
    parser.add_argument("--use-slow-tokenizer", default=False, action="store_true")
    parser.add_argument("--unconditional-start-token", default="<|endoftext|>")
    parser.add_argument("--max-intrasentence-examples", type=int, default=None)
    parser.add_argument("--max-intersentence-examples", type=int, default=None)
    parser.add_argument("--skip-intrasentence", default=False, action="store_true")
    parser.add_argument("--skip-intersentence", default=False, action="store_true")
    return parser.parse_args()


def dtype_from_arg(dtype):
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return "auto"


def model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def default_results_path(output_dir, output_name):
    stem = os.path.splitext(output_name)[0]
    if stem.startswith("predictions_"):
        stem = stem[len("predictions_"):]
    return os.path.join(output_dir, f"{stem}_results.json")


def default_results_model_name(args, output_name):
    if args.results_model_name:
        return args.results_model_name
    stem = os.path.splitext(output_name)[0]
    if stem.startswith("predictions_"):
        return stem[len("predictions_"):]
    return os.path.basename(os.path.normpath(args.model_path))


def should_skip_scoring(args):
    return (
        args.skip_scoring
        or args.skip_intrasentence
        or args.skip_intersentence
        or args.max_intrasentence_examples is not None
        or args.max_intersentence_examples is not None
    )


def score_predictions(args, predictions_file, output_name):
    results_file = args.results_output_file or default_results_path(args.output_dir, output_name)
    model_name = default_results_model_name(args, output_name)
    evaluator = ScoreEvaluator(
        gold_file_path=os.path.abspath(args.input_file),
        predictions_file_path=predictions_file,
    )
    results = to_jsonable(evaluator.get_overall_results())
    wrapped_results = {model_name: results}

    os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(wrapped_results, f, indent=2)
    print(f"Wrote {results_file}")


class OriginalGenerativeQwenEvaluator:
    def __init__(self, args):
        self.args = args
        self.dataset = dataloader.StereoSet(os.path.abspath(args.input_file))
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=not args.use_slow_tokenizer,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}
        if args.device_map is not None:
            model_kwargs["device_map"] = args.device_map

        torch_dtype = dtype_from_arg(args.dtype)
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype
        else:
            model_kwargs["torch_dtype"] = "auto"

        if args.load_in_4bit and args.load_in_8bit:
            raise ValueError("Only one of --load-in-4bit and --load-in-8bit can be set.")
        if args.load_in_4bit or args.load_in_8bit:
            from transformers import BitsAndBytesConfig

            if args.load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
        if args.device_map is None:
            device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
            self.model.to(device)
        self.model.eval()

        self.start_ids = self.tokenizer.encode(args.unconditional_start_token, add_special_tokens=False)
        if len(self.start_ids) != 1:
            fallback_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            if fallback_id is None:
                raise ValueError(
                    "--unconditional-start-token must encode to one token, or tokenizer must define bos/eos token."
                )
            print(
                f"Warning: {args.unconditional_start_token!r} encodes to {len(self.start_ids)} tokens; "
                f"using tokenizer bos/eos id {fallback_id} instead."
            )
            self.start_ids = [fallback_id]

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _logprobs_for_targets(self, target_token_ids, prefix_token_ids=None):
        prefix_token_ids = prefix_token_ids or self.start_ids
        scores = []
        total = math.ceil(len(target_token_ids) / self.args.batch_size)

        for start in tqdm(range(0, len(target_token_ids), self.args.batch_size), total=total):
            batch_targets = target_token_ids[start:start + self.args.batch_size]
            sequences = [prefix_token_ids + target for target in batch_targets]
            encoded = self.tokenizer.pad(
                {"input_ids": sequences},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            device = model_device(self.model)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            for row, target in enumerate(batch_targets):
                prefix_len = len(prefix_token_ids)
                target_len = len(target)
                if target_len == 0:
                    scores.append([])
                    continue

                label_positions = torch.arange(prefix_len, prefix_len + target_len, device=device)
                logit_positions = label_positions - 1
                labels = input_ids[row, label_positions]
                token_log_probs = log_probs[row, logit_positions, labels].detach().cpu().tolist()
                scores.append(token_log_probs)

        return scores

    def evaluate_intrasentence(self):
        examples = self.dataset.get_intrasentence_examples()
        if self.args.max_intrasentence_examples is not None:
            examples = examples[:self.args.max_intrasentence_examples]

        items = []
        for example in examples:
            for sentence in example.sentences:
                items.append((sentence.ID, self._encode(sentence.sentence)))

        token_log_probs = self._logprobs_for_targets([tokens for _, tokens in items])
        predictions = []
        for (sentence_id, _), log_probs in zip(items, token_log_probs):
            score = float(np.exp(np.mean(log_probs))) if log_probs else 0.0
            predictions.append({"id": sentence_id, "score": score})
        return predictions

    def evaluate_intersentence(self):
        examples = self.dataset.get_intersentence_examples()
        if self.args.max_intersentence_examples is not None:
            examples = examples[:self.args.max_intersentence_examples]

        sentence_items = []
        context_items = []
        for example in examples:
            context = example.context
            if context and context[-1] not in [".", "!", "?"]:
                context = f"{context}."
            context_tokens = self._encode(context)
            for sentence in example.sentences:
                sentence_items.append((sentence.ID, self._encode(sentence.sentence)))
                context_items.append(context_tokens)

        print("Scoring intersentence candidates without context...")
        no_context_log_probs = self._logprobs_for_targets([tokens for _, tokens in sentence_items])
        print("Scoring intersentence contexts...")
        context_log_probs = self._logprobs_for_targets(context_items)

        predictions = []
        for (sentence_id, _), no_context_logs, context_logs in zip(
            sentence_items, no_context_log_probs, context_log_probs
        ):
            no_context_score = float(np.mean(no_context_logs)) if no_context_logs else 0.0
            context_score = float(np.mean(context_logs)) if context_logs else 0.0
            if context_score == 0.0:
                score = 0.0
            else:
                score = no_context_score / context_score
            predictions.append({"id": sentence_id, "score": score})
        return predictions

    def evaluate(self):
        results = {}
        if not self.args.skip_intrasentence:
            print("Evaluating intrasentence examples with original generative scoring...")
            results["intrasentence"] = self.evaluate_intrasentence()
        if not self.args.skip_intersentence:
            print("Evaluating intersentence examples with original generative scoring...")
            results["intersentence"] = self.evaluate_intersentence()
        return results


def main():
    args = parse_args()
    evaluator = OriginalGenerativeQwenEvaluator(args)
    results = evaluator.evaluate()

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name
    if output_name is None:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        output_name = f"predictions_{model_name}_OriginalGenerative.json"
    output_file = os.path.join(args.output_dir, output_name)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output_file}")

    if should_skip_scoring(args):
        if not args.skip_scoring:
            print("Skipping scoring because this run did not produce the full StereoSet prediction set.")
        return

    print("Scoring predictions with StereoSet evaluation...")
    score_predictions(args, output_file, output_name)


if __name__ == "__main__":
    main()
