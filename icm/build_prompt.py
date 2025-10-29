from pathlib import Path
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
