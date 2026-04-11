from pathlib import Path
import json
import re

BASE_DIR = Path.cwd()
TRAIN_PATH = BASE_DIR / "train-data"
MODEL_PATH = BASE_DIR / "model"
MODEL_PATH.mkdir(exist_ok=True)


# Match letters, digits, unicode characters
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def load_model(folder: Path, clean: bool) -> dict[str, int]:
    model_file = folder / "freq_model"
    if clean or not model_file.is_file():
        return {}
    with model_file.open("r", encoding="utf-8") as file:
        return json.load(file)
