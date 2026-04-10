from naive_train import train_file, train_pdf
from pathlib import Path
import glob
import os

TRAIN_PATH = Path(os.getcwd() + "/train-data")
MODEL_PATH = Path(os.getcwd() + "/model")


def train():
    import json

    # Get folder to label mapping
    with open("folder_to_label.json", "r") as file:
        f2l = json.load(file)
    # Load a json which has
    proc, proc_path = {}, MODEL_PATH / "processed.json"
    if proc_path.is_file():
        with open(str(proc_path), "r") as file:
            proc = json.load(file)
    for k, v in f2l.items():
        # If it's a text file.
        txt_files = glob.glob(str(TRAIN_PATH / k) + "/*.txt")
        for txt_file in txt_files:
            if txt_file not in proc:
                train_file(v, txt_file)
                proc[txt_file] = True
        # If it's a pdf.
        pdf_files = glob.glob(str(TRAIN_PATH / k) + "/*.pdf")
        for pdf_file in pdf_files:
            if pdf_file not in proc:
                train_pdf(v, pdf_file)
                proc[pdf_file] = True
    # Save files which have been processed to stop re-processing.
    with open(str(proc_path), "w") as file:
        json.dump(proc, file, indent=4)


def main():
    train()


if __name__ == "__main__":
    main()
