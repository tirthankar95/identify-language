from naive_train import train_file, train_pdf
from pathlib import Path
import glob
import os

TRAIN_PATH = Path(os.getcwd() + "/train-data")


def train():
    import json

    with open("folder_to_label.json", "r") as file:
        f2l = json.load(file)
    for k, v in f2l.items():
        # If it's a text file.
        txt_files = glob.glob(str(TRAIN_PATH / k) + "/*.txt")
        for txt_file in txt_files:
            train_file(v, txt_file)
        # If it's a pdf.
        pdf_files = glob.glob(str(TRAIN_PATH / k) + "/*.pdf")
        for pdf_file in pdf_files:
            print(v, pdf_file)
            train_pdf(v, pdf_file)


def main():
    train()


if __name__ == "__main__":
    main()
