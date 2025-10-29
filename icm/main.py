import asyncio
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .baseline import extract_true_false
from .build_prompt import zero_shot_chat
from .dataloader import load_data
from .llm_client import inference_base_model, log_props_base_model

DATA_DIR = Path(__file__).parent.parent / "data"


def internal_coherence_maximization(
    temp_init: float,
    temp_min: float,
    cooling_rate: float,
    k_examples: int,
    n_steps: int,
):
    _, test_df = load_data()
    test_df.index.name = "id"
    labelled_dataset = init_data_labelling(test_df, k_examples)
    # labelled_dataset = pd.read_csv(
    #     DATA_DIR / "temp_labelled_dataset.csv", index_col="id"
    # ).assign(pred=lambda df: df["pred"].astype(str))

    for n in tqdm(range(1, n_steps + 1)):
        temp = cool_temperature(temp_init, temp_min, cooling_rate, n)

        unseen_mask = ~test_df.index.isin(labelled_dataset.index)
        weights = pd.Series(np.where(unseen_mask, 0.99, 0.01))
        sample = test_df.sample(1, weights=weights)

        label, logprob = label_sample(sample, labelled_dataset)
        sample["pred"] = label
        sample["logprob"] = logprob
        temp_labelled_dataset = labelled_dataset.drop(sample.index, errors="ignore")
        temp_labelled_dataset = pd.concat([temp_labelled_dataset, sample])

        mutual_predictability_new = mutual_predictability(temp_labelled_dataset)
        mutual_predictability_old = mutual_predictability(labelled_dataset)
        delta = mutual_predictability_new - mutual_predictability_old

        if delta > 0:
            labelled_dataset = temp_labelled_dataset
        else:
            if random.random() < np.exp(delta / temp):
                labelled_dataset = temp_labelled_dataset
        print(f"Delta: {delta}")
        print(f"Dataset shape: {labelled_dataset.shape}")
        print(
            f"Accuracy: {
                labelled_dataset.assign(
                    acc=lambda df: df['label'].replace({0: 'False', 1: 'True'})
                    == df['pred']
                ).acc.mean()
            }"
        )
        labelled_dataset.assign(iteration=n).to_csv(
            DATA_DIR / "temp_labelled_dataset.csv"
        )
    labelled_dataset.to_csv(DATA_DIR / "labelled_dataset.csv")


def cool_temperature(
    temp_init: float, temp_min: float, cooling_rate: float, iteration: int
) -> float:
    return max(temp_min, (temp_init / (1 + cooling_rate * np.log(iteration))))


def init_data_labelling(test_df: pd.DataFrame, k_examples: int) -> pd.DataFrame:
    temp_test = test_df.sample(k_examples)
    labels = []
    logprobs = []
    for question, choice in zip(temp_test["question"], temp_test["choice"]):
        prompt = zero_shot_chat(question, choice)
        response = inference_base_model(prompt)
        label, logprob = extract_true_false(response)
        labels.append(label)
        logprobs.append(logprob)
    temp_test["pred"] = labels
    temp_test["logprob"] = logprobs
    return temp_test


def label_sample(
    sample: pd.DataFrame, labelled_dataset: pd.DataFrame
) -> tuple[str, float]:
    prompt = rest_as_few_shot(sample, labelled_dataset)
    response = inference_base_model(prompt)
    return extract_true_false(response)


def rest_as_few_shot(sample: pd.DataFrame, labelled_dataset: pd.DataFrame) -> str:
    context = labelled_dataset.drop(sample.index, errors="ignore")
    fewshot_examples = construct_few_shot_examples(context)
    sample_prompt = zero_shot_chat(
        sample["question"].values[0], sample["choice"].values[0]
    )
    full_prompt = "\n".join(fewshot_examples) + "\n" + sample_prompt
    return full_prompt


def mutual_predictability(labelled: pd.DataFrame) -> float:
    return asyncio.run(mutual_predictability_async(labelled))


async def mutual_predictability_async(labelled: pd.DataFrame) -> float:
    semaphore = asyncio.Semaphore(3)

    async def get_logprob(idx):
        async with semaphore:
            df = move_index_to_end(labelled, idx)
            fewshot_examples = construct_few_shot_examples(df)
            prompt = "\n".join(fewshot_examples)
            logprob = await log_props_base_model(prompt)
            return logprob[-1]

    logprobs = await asyncio.gather(*[get_logprob(idx) for idx in labelled.index])
    return sum(logprobs)


def move_index_to_end(labelled: pd.DataFrame, idx: int) -> pd.DataFrame:
    indices = labelled.index.tolist()
    indices.remove(idx)
    indices.append(idx)
    return labelled.reindex(indices, copy=True)


def construct_few_shot_examples(labeled_dataset: pd.DataFrame) -> list[str]:
    return labeled_dataset.assign(
        prompt=lambda df: df.apply(
            lambda row: zero_shot_chat(row["question"], row["choice"])
            + str(row["pred"]),
            axis=1,
        )
    ).prompt.values.tolist()


if __name__ == "__main__":
    k_examples = 8
    temp_init = 10
    temp_min = 0.01

    cooling_rate = 0.99
    n_steps = 150
    internal_coherence_maximization(
        temp_init, temp_min, cooling_rate, k_examples, n_steps
    )
