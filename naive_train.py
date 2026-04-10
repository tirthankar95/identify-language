from pypdf import PdfReader
from pathlib import Path
import json
import os

MODEL_PATH = Path(os.getcwd() + "/model")


def train_file(lang: str, filename: str):
    folder = MODEL_PATH / lang
    folder.mkdir(parents=False, exist_ok=True)
    freq = {}
    if (folder / "freq_model").is_file():
        with open(str(folder / "freq_model"), "r") as file:
            freq = json.load(file)
    with open(filename, "r") as file:
        for line in file:
            for word in line.split():
                if word not in freq:
                    freq[word] = 0
                freq[word] += 1
    with open(str(folder / "freq_model"), "w") as file:
        json.dump(freq, file, indent=4)


def train_pdf(lang: str, filename: str):
    folder = MODEL_PATH / lang
    folder.mkdir(parents=False, exist_ok=True)
    freq = {}
    if (folder / "freq_model").is_file():
        with open(str(folder / "freq_model"), "r") as file:
            freq = json.load(file)
    reader = PdfReader(filename)
    for page in reader.pages:
        text = page.extract_text()
        for line in text:
            for word in line.split():
                if word not in freq:
                    freq[word] = 0
                freq[word] += 1
    with open(str(folder / "freq_model"), "w") as file:
        json.dump(freq, file, ident=4)
