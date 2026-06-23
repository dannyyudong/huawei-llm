import json
import math
import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import dataloader


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained-class", required=True)
    parser.add_argument("--input-file", default="../data/dev.json")
    parser.add_argument("--output-dir", default="predictions/")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-cuda", default=False, action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
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


class CausalLMEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.dataset = dataloader.StereoSet(os.path.abspath(args.input_file))
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_class, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}
        torch_dtype = dtype_from_arg(args.dtype)
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(args.pretrained_class, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _score_texts(self, texts):
        scores = []
        for start in tqdm(range(0, len(texts), self.args.batch_size), total=math.ceil(len(texts) / self.args.batch_size)):
            batch_texts = texts[start:start + self.args.batch_size]
            encoded = self.tokenizer(batch_texts, return_tensors="pt", padding=True)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:].bool()
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

            token_log_probs = token_log_probs.masked_fill(~shift_mask, 0.0)
            lengths = shift_mask.sum(dim=1).clamp_min(1)
            batch_scores = (token_log_probs.sum(dim=1) / lengths).detach().cpu().tolist()
            scores.extend(batch_scores)
        return scores

    def evaluate_intrasentence(self):
        examples = self.dataset.get_intrasentence_examples()
        if self.args.max_intrasentence_examples is not None:
            examples = examples[:self.args.max_intrasentence_examples]

        items = []
        for example in examples:
            for sentence in example.sentences:
                items.append((sentence.ID, sentence.sentence))

        text_scores = self._score_texts([text for _, text in items])
        return [{"id": sentence_id, "score": score} for (sentence_id, _), score in zip(items, text_scores)]

    def evaluate_intersentence(self):
        examples = self.dataset.get_intersentence_examples()
        if self.args.max_intersentence_examples is not None:
            examples = examples[:self.args.max_intersentence_examples]

        items = []
        for example in examples:
            context = example.context
            if context and context[-1] not in [".", "!", "?"]:
                context = f"{context}."
            for sentence in example.sentences:
                items.append((sentence.ID, f"{context} {sentence.sentence}"))

        text_scores = self._score_texts([text for _, text in items])
        return [{"id": sentence_id, "score": score} for (sentence_id, _), score in zip(items, text_scores)]

    def evaluate(self):
        results = {}
        if not self.args.skip_intrasentence:
            print("Evaluating intrasentence examples...")
            results["intrasentence"] = self.evaluate_intrasentence()
        if not self.args.skip_intersentence:
            print("Evaluating intersentence examples...")
            results["intersentence"] = self.evaluate_intersentence()
        return results


def main():
    args = parse_args()
    evaluator = CausalLMEvaluator(args)
    results = evaluator.evaluate()

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name
    if output_name is None:
        model_name = os.path.basename(os.path.normpath(args.pretrained_class))
        output_name = f"predictions_{model_name}_CausalLM.json"
    output_file = os.path.join(args.output_dir, output_name)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
