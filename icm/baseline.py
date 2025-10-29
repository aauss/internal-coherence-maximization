from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .build_prompt import few_shot_chat, zero_shot_antrophic, zero_shot_chat
from .dataloader import load_data
from .llm_client import (
    inference_base_model,
    inference_instruct_model,
    inference_instruct_model_with_messages,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def extract_true_false(
    response: dict[str, float],
) -> tuple[str, float] | tuple[None, None]:
    only_true_false = {
        k: v for k, v in response.items() if k.lower().strip() in ["true", "false"]
    }
    try:
        final_answer = max(only_true_false, key=only_true_false.get)
        logprob = only_true_false[final_answer]
        return final_answer.lower().strip().capitalize(), logprob
    except ValueError:
        return None, None


def zero_shot_base_model(df: pd.DataFrame) -> None:
    responses = []
    logprobs = []
    for question, choice in tqdm(zip(df["question"], df["choice"]), total=len(df)):
        prompt = zero_shot_antrophic(question, choice)
        response = inference_base_model(prompt)
        response, logprob = extract_true_false(response)
        responses.append(response)
        logprobs.append(logprob)
    df["response"] = responses
    df["logprob"] = logprobs
    df.to_csv(DATA_DIR / "zero_shot_base_model.csv")


def zero_shot_instruct_model(df: pd.DataFrame) -> None:
    responses = []
    for question, choice in tqdm(zip(df["question"], df["choice"]), total=len(df)):
        prompt = zero_shot_chat(question, choice)
        response = inference_instruct_model(prompt)
        responses.append(response)
    df["response"] = responses
    df.to_csv(DATA_DIR / "zero_shot_instruct_model.csv")


def few_shot_instruct_model(df: pd.DataFrame, train_df: pd.DataFrame) -> None:
    responses = []
    for question, choice in tqdm(zip(df["question"], df["choice"]), total=len(df)):
        messages = few_shot_chat(question, choice, train_df)
        response = inference_instruct_model_with_messages(messages)
        responses.append(response)
    df["response"] = responses
    df.to_csv(DATA_DIR / "few_shot_instruct_model.csv")


if __name__ == "__main__":
    train_split, test_split = load_data()
    # zero_shot_base_model(test_split)
    # zero_shot_instruct_model(test_split)
    few_shot_instruct_model(test_split, train_split)
