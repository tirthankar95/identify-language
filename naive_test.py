import json
import math
import logging
from pathlib import Path
from pypdf import PdfReader
from naive_train import tokenize_text
from commons import tokenize_text, load_model, MODEL_PATH, TRAIN_PATH

logger = logging.getLogger(__name__)


def score_tokens(model, tokens) -> float:
    score = 0.0
    for token in tokens:
        if token in model:
            score += model[token]
    return score


def laplace_smoothen(model, nc, alpha: int = 0.1):
    total = 0
    for k, v in model.items():
        total += v
    for k, v in model.items():
        v_temp = math.log((v + alpha) / (total + nc * alpha))
        model[k] = v_temp


def predict_text(text: str) -> str:
    with open("folder_to_label.json", "r") as file:
        f2l = json.load(file)
    c = len(f2l)  # no. of classes
    tokens = tokenize_text(text)
    scores = []
    for folder, label in f2l.items():
        folder_path = TRAIN_PATH / folder
        model = load_model(folder_path)
        model = laplace_smoothen(model, c)
        scores.append((score_tokens(model, tokens), label))
    scores.sort()
    return scores[-1][0]


def test_file(filepath: str) -> str:
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as file:
        return predict_text(file.read())


def test_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return predict_text("\n".join(pages))
