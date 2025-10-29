from pathlib import Path

import pandas as pd


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = Path(__file__).parent.parent / "data" / "truthfulqa_train.json"
    train_split = pd.read_json(train_path)
    test_path = Path(__file__).parent.parent / "data" / "truthfulqa_test.json"
    test_split = pd.read_json(test_path)
    return train_split, test_split
