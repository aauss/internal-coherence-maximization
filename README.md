# Unsupervised Elicitation of Language Models — Minimal Reimplementation

## Overview

This repo reimplements parts of the paper [Unsupervised Elicitation of Language Models](https://arxiv.org/pdf/2506.10139v1).

Specifically, it focuses on a subset of the TruthfulQA data only, and only implements algorithm 1 of the paper and leaves out the consisteency fix. More information can be found in the paper.

## Installation

All dependencies are listed in the `pyproject.toml`. If you are using `uv`, you can simply run

```bash
uv sync
```

to install all dependencies. If you don't have `uv`, you can also run

```bash
pip install .
```

This repo requires access to [Hyperbolic AI](https://www.hyperbolic.ai), which offers API access to Llama-3.1-405B. Save your API key into a .env file like that

```bash
echo "HYPERBOLIC_API_KEY=<your_api_key_here>" > .env
```

The minimum credit to open up an account is \$25, which is more than enough for this experiment.

## Repo structure

This repo contains a sample of the TruthfulQA dataset, and a relevant prompt under `/data` folder. Results are also written into this folder. All code is in the `/icm` folder.

### Main algorithm

The main script, which runs algorithm 1 of the paper, can be executed by running.

```bash
uv run python -m icm.main
```

### Baseline

For comparison, I also run three baseline experiments. (1) is a zero-shot experiment using the Llama-3.1-405B base model with an [Anthropic *super* prompt](/data/antrophic_prompt.txt). (2) is a zero-shot experiment using the instruction tuned Llama-3.1-405B model. (3) uses few shot examples form the train split as input for the instruction tuned Llama-3.1-405B for comparison.

You can run the baseline experiments by executing

```bash
uv run python -m icm.baseline
```

### Other modules

- `dataloder.py` is a helper to load the TruthfulQA dataset.
- `build_prompt.py` implements functions to create the prompts as described in the paper.
- `llm_client.py` wraps API calls to Llama-3.1-405B.
- `plotting.ipynb` loads and plots the results.

## Possible improvements

Caching could speed up my experiments and save money as the context for the mutual predictability hardly varies per step. I also did not run finetuning experiments. Since Hyperbolic AI offers renting GPUs, I could have finetuned models on the training set as described in the paper. Lastly, Figure 1 shows error bars indicating they ran their experiments three times. Due to time reasons I did not do that.
