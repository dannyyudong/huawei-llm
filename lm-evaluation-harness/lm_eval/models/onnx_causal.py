import logging
import os
from typing import Any

import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from transformers import AutoTokenizer

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import handle_stop_sequences, postprocess_generated_text


eval_logger = logging.getLogger(__name__)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class OnnxCausalRunner:
    def __init__(self, onnx_path: str, provider: str, use_amct: bool = False):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(
                f"ONNX model file does not exist: {onnx_path}. "
                "Replace the example placeholder with a real .onnx path."
            )

        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(
                f"requested ONNX provider={provider!r} is unavailable; available providers={available}"
            )

        session_options = None
        if use_amct:
            try:
                import amct_onnx as amct
            except ImportError as exc:
                raise RuntimeError(
                    "use_amct=true requires amct_onnx to be installed in this environment"
                ) from exc
            session_options = amct.AMCT_SO

        if session_options is None:
            self.sess = ort.InferenceSession(onnx_path, providers=[provider])
        else:
            self.sess = ort.InferenceSession(
                onnx_path, session_options, providers=[provider]
            )

        self.input_shapes = {x.name: x.shape for x in self.sess.get_inputs()}
        past_key_inputs = [
            x
            for x in self.sess.get_inputs()
            if x.name.startswith("past_key_values") and x.name.endswith(".key")
        ]
        if not past_key_inputs:
            raise RuntimeError("ONNX input does not contain past_key_values.*.key")

        self.num_layers = len(past_key_inputs)
        shape = past_key_inputs[0].shape
        self.num_heads = int(shape[1]) if isinstance(shape[1], int) else 8
        self.head_dim = int(shape[3]) if isinstance(shape[3], int) else 128

        eval_logger.info(
            "Loaded ONNX cache graph: layers=%s kv_heads=%s head_dim=%s",
            self.num_layers,
            self.num_heads,
            self.head_dim,
        )

    def _empty_past(self, batch: int = 1):
        past = []
        for _ in range(self.num_layers):
            key = np.zeros((batch, self.num_heads, 0, self.head_dim), dtype=np.float32)
            value = np.zeros(
                (batch, self.num_heads, 0, self.head_dim), dtype=np.float32
            )
            past.append((key, value))
        return past

    def _step(self, input_ids, attention_mask, position_ids, past):
        feeds = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
            "position_ids": position_ids.astype(np.int64),
        }
        for i, (key, value) in enumerate(past):
            feeds[f"past_key_values.{i}.key"] = key
            feeds[f"past_key_values.{i}.value"] = value

        outputs = self.sess.run(None, feeds)
        logits = outputs[0]
        presents = outputs[1:]
        new_past = []
        for i in range(self.num_layers):
            new_past.append((presents[2 * i], presents[2 * i + 1]))
        return logits, new_past

    def _next_token(
        self,
        logits,
        tokens: list[int],
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> int:
        scores = logits[:, -1, :].astype(np.float64)[0]
        if repetition_penalty and repetition_penalty != 1.0:
            for token_id in set(tokens):
                if scores[token_id] < 0:
                    scores[token_id] *= repetition_penalty
                else:
                    scores[token_id] /= repetition_penalty

        if not do_sample:
            return int(np.argmax(scores))

        temperature = max(float(temperature), 1e-6)
        scores = scores / temperature

        if top_k and top_k > 0 and top_k < scores.shape[-1]:
            keep = np.argpartition(scores, -top_k)[-top_k:]
            filtered = np.full_like(scores, -np.inf)
            filtered[keep] = scores[keep]
            scores = filtered

        probs = np.exp(scores - np.nanmax(scores))
        probs = probs / probs.sum()

        if top_p and 0 < top_p < 1:
            sorted_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_idx]
            keep = np.cumsum(sorted_probs) <= top_p
            keep[0] = True
            filtered_probs = np.zeros_like(probs)
            filtered_probs[sorted_idx[keep]] = sorted_probs[keep]
            probs = filtered_probs / filtered_probs.sum()

        return int(np.random.choice(np.arange(probs.shape[-1]), p=probs))

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        eos_token_ids,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ):
        eos_token_ids = set(eos_token_ids or [])
        tokens = list(input_ids)
        past = self._empty_past(batch=1)
        logits = None

        for token_index, token_id in enumerate(tokens):
            cur = np.array([[token_id]], dtype=np.int64)
            attention_mask = np.ones((1, token_index + 1), dtype=np.int64)
            position_ids = np.array([[token_index]], dtype=np.int64)
            logits, past = self._step(cur, attention_mask, position_ids, past)

        if logits is None or max_new_tokens <= 0:
            return tokens

        for _ in range(max_new_tokens):
            next_id = self._next_token(
                logits,
                tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            tokens.append(next_id)
            if next_id in eos_token_ids:
                break

            cur = np.array([[next_id]], dtype=np.int64)
            total_len = past[0][0].shape[2] + 1
            attention_mask = np.ones((1, total_len), dtype=np.int64)
            position_ids = np.array([[total_len - 1]], dtype=np.int64)
            logits, past = self._step(cur, attention_mask, position_ids, past)

        return tokens


@register_model("onnx", "onnx-causal")
class OnnxCausalLM(LM):
    def __init__(
        self,
        pretrained: str,
        tokenizer: str | None = None,
        provider: str = "CUDAExecutionProvider",
        use_amct: bool = False,
        max_length: int = 2048,
        max_gen_toks: int = 256,
        trust_remote_code: bool = True,
        fix_mistral_regex: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.onnx_path = pretrained
        self.tokenizer_path = tokenizer or pretrained.rsplit("/", 1)[0]
        self.provider = provider
        self.max_length = int(max_length)
        self.max_gen_toks = int(max_gen_toks)
        self._device = provider

        tokenizer_kwargs = {
            "trust_remote_code": _as_bool(trust_remote_code),
            "padding_side": "left",
        }
        if _as_bool(fix_mistral_regex):
            tokenizer_kwargs["fix_mistral_regex"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            self.eot_token_id = eos_token_id
            self.eos_token_ids = [eos_token_id]
        elif isinstance(eos_token_id, (list, tuple)):
            self.eot_token_id = eos_token_id[0]
            self.eos_token_ids = list(eos_token_id)
        else:
            self.eot_token_id = None
            self.eos_token_ids = []

        self.runner = OnnxCausalRunner(
            self.onnx_path, provider=self.provider, use_amct=_as_bool(use_amct)
        )

        unused = ", ".join(sorted(kwargs))
        if unused:
            eval_logger.warning("Unused onnx model_args: %s", unused)

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer_path

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        encode_kwargs = {}
        if add_special_tokens is not None:
            encode_kwargs["add_special_tokens"] = add_special_tokens
        return self.tokenizer.encode(string, **encode_kwargs)

    def tok_decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "The onnx model currently supports generation tasks only. Use tasks whose output_type is generate_until."
        )

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "The onnx model currently supports generation tasks only. Use tasks whose output_type is generate_until."
        )

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        results = []
        eos = (
            self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            if self.eot_token_id is not None
            else None
        )

        for request in tqdm(
            requests, disable=disable_tqdm, desc="Running ONNX generate_until requests"
        ):
            context, gen_kwargs = request.args
            original_gen_kwargs = dict(gen_kwargs)
            gen_kwargs = dict(gen_kwargs)
            until = handle_stop_sequences(gen_kwargs.pop("until", None), eos=eos)
            max_gen_toks = int(
                gen_kwargs.pop(
                    "max_gen_toks",
                    gen_kwargs.pop("max_new_tokens", self.max_gen_toks),
                )
            )

            unsupported = [
                key
                for key in gen_kwargs
                if key
                not in {
                    "do_sample",
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "repetition_penalty",
                }
            ]
            if unsupported:
                eval_logger.warning(
                    "Ignoring generation kwargs unsupported by greedy ONNX runner: %s",
                    ", ".join(sorted(unsupported)),
                )

            input_ids = self.tok_encode(context)[-(self.max_length - max_gen_toks) :]
            output_ids = self.runner.generate(
                input_ids=input_ids,
                max_new_tokens=max_gen_toks,
                eos_token_ids=self.eos_token_ids,
                do_sample=_as_bool(gen_kwargs.get("do_sample", False)),
                temperature=float(gen_kwargs.get("temperature", 1.0) or 1.0),
                top_p=float(gen_kwargs.get("top_p", 1.0) or 1.0),
                top_k=int(gen_kwargs.get("top_k", 0) or 0),
                repetition_penalty=float(
                    gen_kwargs.get("repetition_penalty", 1.0) or 1.0
                ),
            )
            generated_ids = output_ids[len(input_ids) :]
            generated = self.tok_decode(generated_ids)
            generated = postprocess_generated_text(
                generated, stop=until, think_end_token=None
            )
            results.append(generated)
            self.cache_hook.add_partial(
                "generate_until", (context, original_gen_kwargs), generated
            )

        return results
