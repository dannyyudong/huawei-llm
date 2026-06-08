import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import requests
import torch
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.eval_config import LOCAL_SERVER_PORT
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    system_prompt_pre_processing_chat_model,
)
from bfcl_eval.utils import contain_multi_turn_interaction
from openai import OpenAI
from overrides import EnforceOverrides, final, override


class OnnxAmctCausalLMRunner:
    """Minimal greedy decoder for AMCT-exported ONNX causal LM cache graphs."""

    def __init__(self, onnx_path: str, provider: str = "CUDAExecutionProvider"):
        import amct_onnx as amct
        import numpy as np
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        if provider == "auto":
            provider = (
                "CUDAExecutionProvider"
                if "CUDAExecutionProvider" in available_providers
                else "CPUExecutionProvider"
            )
        if provider not in available_providers:
            raise RuntimeError(
                f"Requested ONNX provider={provider} is not available. "
                f"Available providers={available_providers}"
            )

        self.np = np
        self.session = ort.InferenceSession(onnx_path, amct.AMCT_SO, providers=[provider])
        self.input_shapes = {input_.name: input_.shape for input_ in self.session.get_inputs()}

        past_key_inputs = [
            input_
            for input_ in self.session.get_inputs()
            if input_.name.startswith("past_key_values") and input_.name.endswith(".key")
        ]
        if not past_key_inputs:
            raise RuntimeError("ONNX graph does not expose past_key_values.*.key inputs.")

        self.num_layers = len(past_key_inputs)
        first_past_shape = past_key_inputs[0].shape
        self.num_heads = int(first_past_shape[1]) if isinstance(first_past_shape[1], int) else 8
        self.head_dim = int(first_past_shape[3]) if isinstance(first_past_shape[3], int) else 128

        print(
            "Loaded AMCT ONNX cache graph: "
            f"layers={self.num_layers}, kv_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, provider={provider}"
        )

    def _empty_past(self, batch: int = 1):
        past = []
        for _ in range(self.num_layers):
            key = self.np.zeros(
                (batch, self.num_heads, 0, self.head_dim), dtype=self.np.float32
            )
            value = self.np.zeros(
                (batch, self.num_heads, 0, self.head_dim), dtype=self.np.float32
            )
            past.append((key, value))
        return past

    def _step(self, input_ids, attention_mask, position_ids, past):
        feeds = {
            "input_ids": input_ids.astype(self.np.int64),
            "attention_mask": attention_mask.astype(self.np.int64),
            "position_ids": position_ids.astype(self.np.int64),
        }
        for layer_idx, (key, value) in enumerate(past):
            feeds[f"past_key_values.{layer_idx}.key"] = key
            feeds[f"past_key_values.{layer_idx}.value"] = value

        outputs = self.session.run(None, feeds)
        logits = outputs[0]
        presents = outputs[1:]
        new_past = []
        for layer_idx in range(self.num_layers):
            new_past.append((presents[2 * layer_idx], presents[2 * layer_idx + 1]))
        return logits, new_past

    def generate(self, input_ids: list[int], max_new_tokens: int, eos_token_ids: list[int]):
        eos_token_ids = set(eos_token_ids or [])
        tokens = list(input_ids)
        past = self._empty_past(batch=1)
        logits = None

        # AMCT cache graphs are decode-shaped, so feed the prompt one token at a time.
        for token_index, token_id in enumerate(tokens):
            current = self.np.array([[token_id]], dtype=self.np.int64)
            total_len = token_index + 1
            attention_mask = self.np.ones((1, total_len), dtype=self.np.int64)
            position_ids = self.np.array([[token_index]], dtype=self.np.int64)
            logits, past = self._step(current, attention_mask, position_ids, past)

        if logits is None:
            return tokens

        for _ in range(max_new_tokens):
            next_token_id = int(self.np.argmax(logits[:, -1, :], axis=-1)[0])
            tokens.append(next_token_id)
            if next_token_id in eos_token_ids:
                break

            current = self.np.array([[next_token_id]], dtype=self.np.int64)
            total_len = past[0][0].shape[2] + 1
            attention_mask = self.np.ones((1, total_len), dtype=self.np.int64)
            position_ids = self.np.array([[total_len - 1]], dtype=self.np.int64)
            logits, past = self._step(current, attention_mask, position_ids, past)

        return tokens


class OSSHandler(BaseHandler, EnforceOverrides):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name
        self.model_style = ModelStyle.OSSMODEL
        self.dtype = dtype

        # Will be overridden in batch_inference method
        # Used to indicate where the tokenizer and config should be loaded from
        self.model_path_or_id = None

        # Read from env vars with fallbacks
        self.local_server_endpoint = os.getenv("LOCAL_SERVER_ENDPOINT", "localhost")
        self.local_server_port = os.getenv("LOCAL_SERVER_PORT", LOCAL_SERVER_PORT)

        # Support custom base_url and api_key for remote/local OpenAI-compatible deployments (e.g., vLLM)
        # Use REMOTE_OPENAI_* variables to avoid conflicts with main OPENAI_* variables
        self.base_url = os.getenv("REMOTE_OPENAI_BASE_URL", f"http://{self.local_server_endpoint}:{self.local_server_port}/v1")
        self.api_key = os.getenv("REMOTE_OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.use_transformers_backend = False
        self.use_onnx_amct_backend = False
        self.local_model = None
        self.onnx_runner = None
        self.enable_think: Optional[bool] = None

    @override
    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        # TODO: Let oss model support FC methods as well, depends on their model type
        if contain_multi_turn_interaction(test_entry["id"]):
            return self.inference_multi_turn_prompting(
                test_entry, include_input_log, exclude_state_log
            )
        else:
            return self.inference_single_turn_prompting(test_entry, include_input_log)

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    @override
    def decode_execute(self, result, has_tool_call_tag):
        return default_decode_execute_prompting(result, has_tool_call_tag)

    @final
    def spin_up_local_server(
        self,
        num_gpus: int,
        gpu_memory_utilization: float,
        backend: str,
        skip_server_setup: bool,
        local_model_path: Optional[str],
        fix_mistral_regex: bool = False,
        enable_think: Optional[bool] = None,
        lora_modules: Optional[list[str]] = None,
        enable_lora: bool = False,
        max_lora_rank: Optional[int] = None,
        onnx_model_path: Optional[str] = None,
        onnx_provider: str = "CUDAExecutionProvider",
    ):
        """
        Spin up a local server for the model.
        If the server is already running, skip the setup.
        """
        from transformers import AutoConfig, AutoTokenizer
        from transformers import AutoModelForCausalLM

        def _normalize_onnx_path(path: str) -> str:
            candidate = Path(path)
            if candidate.is_dir():
                matches = sorted(candidate.glob("*.onnx"))
                if not matches:
                    raise ValueError(f"No .onnx file found in ONNX model directory '{path}'.")
                return str(matches[0])
            if candidate.suffix == ".data":
                candidate = candidate.with_suffix(".onnx")
            if not candidate.exists():
                raise ValueError(f"ONNX model file '{candidate}' does not exist.")
            return str(candidate)

        # Determine the model source
        if backend == "onnx-amct":
            if onnx_model_path is None and local_model_path is None:
                raise ValueError(
                    "backend='onnx-amct' requires --onnx-model-path or --local-model-path."
                )

            resolved_onnx_path = _normalize_onnx_path(
                onnx_model_path if onnx_model_path is not None else local_model_path
            )
            tokenizer_source = (
                local_model_path
                if local_model_path is not None and os.path.isdir(local_model_path)
                else str(Path(resolved_onnx_path).parent)
            )
            self.model_path_or_id = resolved_onnx_path
            load_kwargs = {
                "pretrained_model_name_or_path": tokenizer_source,
                "local_files_only": os.path.isdir(tokenizer_source),
                "trust_remote_code": True,
            }
        elif local_model_path is not None:
            # Validate the local_model_path
            if not os.path.isdir(local_model_path):
                raise ValueError(
                    f"local_model_path '{local_model_path}' does not exist or is not a directory."
                )

            required_files = ["config.json", "tokenizer_config.json"]
            for file_name in required_files:
                if not os.path.exists(os.path.join(local_model_path, file_name)):
                    raise ValueError(
                        f"Required file '{file_name}' not found in local_model_path '{local_model_path}'."
                    )

            self.model_path_or_id = local_model_path
            load_kwargs = {
                "pretrained_model_name_or_path": self.model_path_or_id,
                "local_files_only": True,
                "trust_remote_code": True,
            }
        else:
            self.model_path_or_id = self.model_name_huggingface
            load_kwargs = {
                "pretrained_model_name_or_path": self.model_path_or_id,
                "trust_remote_code": True,
            }

        # For remote OpenAI-compatible endpoints, use specified tokenizer path if provided
        is_remote_endpoint = bool(os.getenv("REMOTE_OPENAI_BASE_URL"))
        tokenizer_path = os.getenv("REMOTE_OPENAI_TOKENIZER_PATH", self.model_path_or_id)
        self.enable_think = enable_think

        def _load_tokenizer_with_compat(kwargs):
            if not fix_mistral_regex:
                return AutoTokenizer.from_pretrained(**kwargs)

            # Newer transformers can fix known Mistral tokenizer regex issues via this flag.
            tokenizer_kwargs = dict(kwargs)
            tokenizer_kwargs["fix_mistral_regex"] = True
            try:
                return AutoTokenizer.from_pretrained(**tokenizer_kwargs)
            except TypeError:
                # Backward compatibility for transformers versions that do not expose this flag.
                return AutoTokenizer.from_pretrained(**kwargs)

        if is_remote_endpoint and os.getenv("REMOTE_OPENAI_TOKENIZER_PATH"):
            # Use specified tokenizer for remote endpoints
            tokenizer_kwargs = {
                "pretrained_model_name_or_path": tokenizer_path,
                "trust_remote_code": True,
            }
            try:
                self.tokenizer = _load_tokenizer_with_compat(tokenizer_kwargs)
                config = AutoConfig.from_pretrained(**tokenizer_kwargs)
                print(f"Loaded tokenizer from REMOTE_OPENAI_TOKENIZER_PATH: {tokenizer_path}")
            except Exception as e:
                print(f"Failed to load tokenizer from {tokenizer_path}, falling back to model path: {e}")
                self.tokenizer = _load_tokenizer_with_compat(load_kwargs)
                config = AutoConfig.from_pretrained(**load_kwargs)
        else:
            # Standard loading for local models or when no specific tokenizer path is provided
            self.tokenizer = _load_tokenizer_with_compat(load_kwargs)
            config = AutoConfig.from_pretrained(**load_kwargs)

        if hasattr(config, "max_position_embeddings"):
            self.max_context_length = config.max_position_embeddings
        elif self.tokenizer.model_max_length is not None:
            self.max_context_length = self.tokenizer.model_max_length
        else:
            if not hasattr(self, "max_context_length"):
                raise ValueError(
                    "Model does not have a max_position_embeddings attribute or tokenizer.model_max_length attribute. Please set the max_context_length attribute in the corresponding model handler."
                )
        print(f"Max context length: {self.max_context_length}")

        self._server_process = process = None
        self._stdout_thread = stdout_thread = None
        self._stderr_thread = stderr_thread = None
        # Event to signal threads to stop; no need to see logs after server is ready
        # declare early so it always exists
        self._stop_event = threading.Event()
        try:
            if not skip_server_setup or backend in {"transformers", "onnx-amct"}:
                if backend == "vllm":
                    process = subprocess.Popen(
                        [
                            "vllm",
                            "serve",
                            str(self.model_path_or_id),
                            "--port",
                            str(self.local_server_port),
                            "--dtype",
                            str(self.dtype),
                            "--tensor-parallel-size",
                            str(num_gpus),
                            "--gpu-memory-utilization",
                            str(gpu_memory_utilization),
                            "--trust-remote-code",
                        ]
                        + (["--enable-lora"] if enable_lora else [])
                        + (
                            ["--max-lora-rank", str(max_lora_rank)]
                            if max_lora_rank is not None
                            else []
                        )
                        + (
                            sum(
                                [["--lora-modules", lora_module] for lora_module in lora_modules],
                                [],
                            )
                            if lora_modules
                            else []
                        ),
                        stdout=subprocess.PIPE,  # Capture stdout
                        stderr=subprocess.PIPE,  # Capture stderr
                        text=True,  # To get the output as text instead of bytes
                    )
                elif backend == "sglang":

                    process = subprocess.Popen(
                        [
                            "python",
                            "-m",
                            "sglang.launch_server",
                            "--model-path",
                            str(self.model_path_or_id),
                            "--port",
                            str(self.local_server_port),
                            "--dtype",
                            str(self.dtype),
                            "--tp",
                            str(num_gpus),
                            "--mem-fraction-static",
                            str(gpu_memory_utilization),
                            "--trust-remote-code",
                        ],
                        stdout=subprocess.PIPE,  # Capture stdout
                        stderr=subprocess.PIPE,  # Capture stderr
                        text=True,  # To get the output as text instead of bytes
                    )
                elif backend == "transformers":
                    # Run in-process local inference with transformers, no OpenAI-compatible server needed.
                    self.use_transformers_backend = True

                    torch_dtype_map = {
                        "bfloat16": torch.bfloat16,
                        "float16": torch.float16,
                        "float32": torch.float32,
                    }

                    model_load_kwargs = dict(load_kwargs)
                    if torch.cuda.is_available():
                        model_load_kwargs["device_map"] = "auto"
                        dtype = torch_dtype_map.get(str(self.dtype))
                        if dtype is not None:
                            model_load_kwargs["torch_dtype"] = dtype
                    else:
                        # CPU path prefers float32 for compatibility.
                        model_load_kwargs["torch_dtype"] = torch.float32

                    self.local_model = AutoModelForCausalLM.from_pretrained(
                        **model_load_kwargs
                    )
                    self.local_model.eval()
                elif backend == "onnx-amct":
                    self.use_onnx_amct_backend = True
                    self.onnx_runner = OnnxAmctCausalLMRunner(
                        self.model_path_or_id, provider=onnx_provider
                    )
                else:
                    raise ValueError(f"Backend {backend} is not supported.")

                def log_subprocess_output(pipe, stop_event):
                    # Read lines until the pipe is closed (EOF)
                    for line in iter(pipe.readline, ""):
                        if not stop_event.is_set():
                            print(line, end="")
                    print("server log tracking thread stopped successfully.")

                if process is not None:
                    # Start threads to read and print stdout and stderr
                    stdout_thread = threading.Thread(
                        target=log_subprocess_output,
                        args=(process.stdout, self._stop_event),
                    )
                    stderr_thread = threading.Thread(
                        target=log_subprocess_output,
                        args=(process.stderr, self._stop_event),
                    )
                    stdout_thread.setDaemon(True)
                    stderr_thread.setDaemon(True)
                    stdout_thread.start()
                    stderr_thread.start()

            self._server_process = process
            self._stdout_thread = stdout_thread
            self._stderr_thread = stderr_thread

            # Wait for the server to be ready when using endpoint-based backends.
            server_ready = self.use_transformers_backend or self.use_onnx_amct_backend
            while not server_ready:
                # Check if the process has terminated unexpectedly
                if not skip_server_setup and process is not None and process.poll() is not None:
                    # Output the captured logs
                    stdout, stderr = process.communicate()
                    print(stdout)
                    print(stderr)
                    raise Exception(
                        f"Subprocess terminated unexpectedly with code {process.returncode}"
                    )
                try:
                    # Make a simple request to check if the server is up
                    response = requests.get(f"{self.base_url}/models")
                    if response.status_code == 200:
                        server_ready = True
                        print("server is ready!")
                except requests.exceptions.ConnectionError:
                    # If the connection is not ready, wait and try again
                    time.sleep(1)

            # Signal threads to stop reading output
            self._stop_event.set()

        except Exception as e:
            # Clean-up everything we already started, then re-raise
            if self._server_process and self._server_process.poll() is None:
                self._server_process.terminate()
            if self._stop_event:
                self._stop_event.set()
            if self._stdout_thread:
                self._stdout_thread.join(timeout=2)
            if self._stderr_thread:
                self._stderr_thread.join(timeout=2)
            raise e

    def shutdown_local_server(self):
        """Terminate the locally launched OSS model server if it is still running."""
        # Ensure the server process is terminated properly
        process = getattr(self, "_server_process", None)
        if process and process.poll() is None:
            process.terminate()
            try:
                # Wait for the process to terminate fully
                process.wait(timeout=15)
                print("Process terminated successfully.")
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()  # Wait again to ensure it's fully terminated
                print("Process killed.")

        # Tell the log-reader threads to stop and wait for them
        if getattr(self, "_stop_event", None):
            self._stop_event.set()
        if getattr(self, "_stdout_thread", None):
            self._stdout_thread.join(timeout=2)
        if getattr(self, "_stderr_thread", None):
            self._stderr_thread.join(timeout=2)

    #### Prompting methods ####

    def _format_prompt(self, messages, function):
        """
        Manually apply the chat template to construct the formatted prompt.
        This way, we can have full control over the final formatted prompt and is generally recommended for advanced use cases.
        """
        raise NotImplementedError(
            "OSS Models should implement their own prompt formatting."
        )

    @override
    def _query_prompting(self, inference_data: dict):
        # We use the OpenAI Completions API
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        # Tokenize the formatted prompt to get token count
        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))

        # Determine the number of tokens to request. Cap it at 4096 if the model has a larger limit.
        if self.max_context_length < input_token_count + 2:
            # If the prompt is already at the max length, just request 1000 token, we will get an error anyway
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                4096,
                self.max_context_length - input_token_count - 2,
            )

        extra_body = {}
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens

        start_time = time.time()
        if self.use_transformers_backend:
            if self.local_model is None:
                raise RuntimeError(
                    "Transformers backend is enabled but local model is not initialized."
                )

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            model_device = next(self.local_model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            generation_kwargs = {
                "max_new_tokens": int(leftover_tokens_count),
            }

            # Match deterministic evaluation behavior unless temperature is explicitly > 0.
            if self.temperature is not None and self.temperature > 0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = self.temperature
            else:
                generation_kwargs["do_sample"] = False

            with torch.no_grad():
                output_ids = self.local_model.generate(**inputs, **generation_kwargs)

            prompt_len = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[:, prompt_len:]
            skip_special_tokens = bool(getattr(self, "skip_special_tokens", False))
            generated_text = self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=skip_special_tokens
            )

            api_response = SimpleNamespace(
                choices=[SimpleNamespace(text=generated_text)],
                usage=SimpleNamespace(
                    prompt_tokens=input_token_count,
                    completion_tokens=int(generated_ids.shape[-1]),
                ),
            )
        elif self.use_onnx_amct_backend:
            if self.onnx_runner is None:
                raise RuntimeError(
                    "ONNX-AMCT backend is enabled but ONNX runner is not initialized."
                )

            inputs = self.tokenizer(formatted_prompt, return_tensors="np")
            prompt_ids = inputs["input_ids"][0].tolist()

            eos_token_ids = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_ids is None:
                eos_token_ids = []
            elif isinstance(eos_token_ids, int):
                eos_token_ids = [eos_token_ids]
            else:
                eos_token_ids = list(eos_token_ids)

            output_ids = self.onnx_runner.generate(
                input_ids=prompt_ids,
                max_new_tokens=int(leftover_tokens_count),
                eos_token_ids=eos_token_ids,
            )
            generated_ids = output_ids[len(prompt_ids):]
            skip_special_tokens = bool(getattr(self, "skip_special_tokens", False))
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=skip_special_tokens
            )

            api_response = SimpleNamespace(
                choices=[SimpleNamespace(text=generated_text)],
                usage=SimpleNamespace(
                    prompt_tokens=input_token_count,
                    completion_tokens=len(generated_ids),
                ),
            )
        elif len(extra_body) > 0:
            api_response = self.client.completions.create(
                model=self.model_path_or_id,
                temperature=self.temperature,
                prompt=formatted_prompt,
                max_tokens=leftover_tokens_count,
                extra_body=extra_body,
                timeout=72000,  # Avoid timeout errors
            )
        else:
            api_response = self.client.completions.create(
                model=self.model_path_or_id,
                temperature=self.temperature,
                prompt=formatted_prompt,
                max_tokens=leftover_tokens_count,
                timeout=72000,  # Avoid timeout errors
            )
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        return {
            "model_responses": api_response.choices[0].text,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": decoded_model_response,
                    "content": execution_result,
                }
            )

        return inference_data
