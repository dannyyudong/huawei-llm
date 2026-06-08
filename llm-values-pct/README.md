This repo contains code and data to reproduce the experiments from "**Political Compass or Spinning Arrow? Towards More Meaningful Evaluations for Values and Opinions in Large Language Models**" by Paul Röttger, Valentin Hofmann, Valentina Pyatkin, Musashi Hinck, Hannah Rose Kirk, Hinrich Schütze, and Dirk Hovy.

For details, please refer to our ACL paper [here](https://aclanthology.org/2024.acl-long.816/).

### Repo Structure
```
.
├── bash                # bash scripts to run get_completions scripts in src
│
├── data
│   ├── annotations     # data annotations and annotation guidelines
│   ├── completions     # model completions on the prompts
│   ├── prompts         # instantiated prompts for the model
│   ├── templates       # templates used for prompt generation
│
├── notebooks           # .ipynb notebooks to analyse the completions
│   ├── figures         # figures generated from the notebooks
│   ├── utils           # utility functions used in the notebooks
│   ├── pct_validation  # scripts for validating our pct implementation
│
├── src                 # .py scripts to generate prompts and get completions
```

Note: In the naming convention used in this repo, "explicit" corresponds to multiple-choice prompts, and "implicit" corresponds to open-ended prompts. "jailbreak" corresponds to experiments that vary the forced choice prompt, and "paraphrase" corresponds to experiments that vary the prompt template itself. See for example ./data/prompts.

### Qwen3 local evaluation

The original paper-style workflow remains unchanged. For local Hugging Face
models, `src/2_get_completions_hf.py` generates completions and
`src/3_analyze_explicit.py` analyzes the original numbered answer labels.

An additional balanced-order diagnostic is available without changing the
paper workflow:

1. `src/1_generate_balanced_prompts.py` creates four cyclic A/B/C/D option
   orders for every source prompt.
2. `src/2_get_completions_hf.py` generates one letter response per prompt.
3. `src/3_analyze_balanced.py` maps each letter back to the original PCT
   choice and reports coordinates by option order and by prompt template.

Example for one neutral template:

```bash
python src/1_generate_balanced_prompts.py \
  --input_path data/prompts/PCT-neutral-templ-01.csv \
  --output_path data/prompts/PCT-balanced-templ-01.csv

CUDA_VISIBLE_DEVICES=0 python src/2_get_completions_hf.py \
  --model_name_or_path /home/huawei/huawei/Qwen3-0.6B \
  --test_data_input_path data/prompts/PCT-balanced-templ-01.csv \
  --test_data_output_path data/completions/Qwen3-0.6B-balanced-templ-01.csv \
  --batch_size 16 \
  --max_new_tokens 8 \
  --require_cuda \
  --overwrite

python src/3_analyze_balanced.py \
  --input_path data/completions/Qwen3-0.6B-balanced-templ-01.csv
```

The balanced-order method is a separate diagnostic and should not be reported
as a direct reproduction of the paper's original prompt condition.

For small models that collapse to one generated label, the probability-based
evaluator avoids free-form generation entirely. It scores the next-token
probabilities of A/B/C/D for all four cyclic option orders, maps those
probabilities back to the four semantic PCT choices, and averages by question:

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval_pct_probabilities.py \
  --model_name_or_path /home/huawei/huawei/Qwen3-0.6B \
  --questions_path data/templates/pct_propositions.csv \
  --output_dir data/completions/Qwen3-0.6B-probability-eval \
  --batch_size 16 \
  --require_cuda
```

This produces both hard argmax coordinates and soft probability-weighted
coordinates. It is also a separate diagnostic, not the original paper method.

To remove option labels and option-order effects entirely, use the semantic
likelihood evaluator. It directly scores the complete phrases "Strongly
disagree", "Disagree", "Agree", and "Strongly agree", then applies a
content-free baseline calibration:

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval_pct_semantic.py \
  --model_name_or_path /home/huawei/huawei/Qwen3-0.6B \
  --questions_path data/templates/pct_propositions.csv \
  --output_dir data/completions/Qwen3-0.6B-semantic-eval \
  --batch_size 16 \
  --require_cuda
```

This semantic method is the recommended diagnostic when a small model exhibits
strong first-option or answer-label bias.
