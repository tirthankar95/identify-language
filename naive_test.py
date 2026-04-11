import json
import math
import logging
from pathlib import Path
from pypdf import PdfReader
from naive_train import tokenize_text
from commons import tokenize_text, load_model, MODEL_PATH

logger = logging.getLogger(__name__)
# No language has 1 billion tokens
DEFAULT_SCORE = math.log(1 / (10**9))


def score_tokens(model, tokens) -> float:
    score = 0.0
    for token in tokens:
        score += model[token] if token in model else DEFAULT_SCORE
    return score


def token_matches(model, tokens) -> float:
    cnt = 0
    for token in tokens:
        cnt += 1 if token in model else 0
    return cnt


def laplace_smoothen(model, nc, alpha: int = 1.0) -> dict:
    total = 0
    for k, v in model.items():
        total += v
    new_model = {}
    for k, v in model.items():
        v_temp = math.log((v + alpha) / (total + nc * alpha))
        new_model[k] = v_temp
    return new_model


def predict_text(text: str) -> str:
    with open("folder_to_label.json", "r") as file:
        f2l = json.load(file)
    c = len(f2l)  # no. of classes
    tokens, scores = tokenize_text(text), []
    for folder, label in f2l.items():
        folder_path = MODEL_PATH / label
        model = load_model(folder_path)
        n_model = laplace_smoothen(model, c)
        _match = token_matches(model, tokens)
        scores.append([score_tokens(n_model, tokens), label, _match])
    scores.sort()
    ref = scores[0][0]
    for idx, x in enumerate(scores):
        scores[idx][0] = scores[idx][0] - ref
    logger.debug(f"{scores=}")
    return scores[-1][1]


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
