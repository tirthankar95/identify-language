import json
import logging
from pathlib import Path
from pypdf import PdfReader
from commons import tokenize_text, load_model, MODEL_PATH

logger = logging.getLogger(__name__)


def save_model(folder: Path, freq: dict[str, int]) -> None:
    with (folder / "freq_model").open("w", encoding="utf-8") as file:
        json.dump(freq, file, indent=4, sort_keys=True)


def update_model(freq: dict[str, int], text: str):
    tokens = tokenize_text(text)
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1


def train_file(lang: str, filename: str, clean: bool = False) -> int:
    folder = MODEL_PATH / lang
    folder.mkdir(parents=True, exist_ok=True)
    freq = load_model(folder, clean=clean)
    with Path(filename).open("r", encoding="utf-8") as file:
        update_model(freq, file.read())
    save_model(folder, freq)


def train_pdf(lang: str, filename: str, clean: bool = False) -> int:
    folder = MODEL_PATH / lang
    folder.mkdir(parents=True, exist_ok=True)
    freq = load_model(folder, clean=clean)
    reader = PdfReader(filename)
    for page in reader.pages:
        update_model(freq, page.extract_text() or "")
    save_model(folder, freq)
