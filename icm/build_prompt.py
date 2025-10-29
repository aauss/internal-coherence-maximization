from itertools import chain
from pathlib import Path

import pandas as pd

from .dataloader import load_data

train_split, test_split = load_data()


def zero_shot_chat(question: str, choice: str) -> str:
    return f"Question: {question}\nClaim: {choice}\nI think the claim is "


def zero_shot_antrophic(question: str, choice: str) -> str:
    antrophic_prompt_path = (
        Path(__file__).parent.parent / "data" / "antrophic_prompt.txt"
    )
    with open(antrophic_prompt_path, "r") as f:
        antrophic_prompt = f.read()
    return f"{antrophic_prompt}\nQuestion: {question}\nClaim: {choice}\nI think the claim is "


def few_shot_chat(
    question: str, choice: str, train_df: pd.DataFrame, num_examples: int = 3
) -> str:
    examples = (
        train_df.sample(num_examples)
        .assign(
            conversation=lambda df: df.apply(
                lambda row: [
                    {
                        "role": "user",
                        "content": f"{row['question']}\nClaim: {row['choice']}\nI think the claim is ",
                    },
                    {
                        "role": "assistant",
                        "content": "True" if row["label"] == 1 else "False",
                    },
                ],
                axis=1,
            )
        )
        .conversation.values.tolist()
    )
    examples = list(chain(*examples))
    examples.append(
        {
            "role": "user",
            "content": zero_shot_chat(question, choice),
        }
    )
    return examples
