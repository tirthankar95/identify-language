from pathlib import Path
import json
import re


def is_chinese(text):
    # Check if text contains any character in the CJK range
    return bool(re.search(r"[\u4e00-\u9fff]", text))


BASE_DIR = Path.cwd()
TRAIN_PATH = BASE_DIR / "train-data"
TEST_PATH = BASE_DIR / "test-data"
MODEL_PATH = BASE_DIR / "model"
MODEL_PATH.mkdir(exist_ok=True)


# Match letters, digits, unicode characters
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    step0 = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    step1 = [re.sub(r"[,!,.:(]", "", token) for token in step0]
    step2 = []
    for token in step1:
        if bool(re.search(r"[\u4e00-\u9fff]", token)):
            for ch in token:
                step2.append(ch)
        else:
            step2.append(token)
    return step2


def load_model(folder: Path, ng: int, clean: bool = False) -> dict[str, int]:
    model_file = folder / f"freq_model_{ng}gram"
    if clean or not model_file.is_file():
        return {}
    with model_file.open("r", encoding="utf-8") as file:
        return json.load(file)
