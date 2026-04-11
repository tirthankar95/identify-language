import json
import argparse
import logging
from pathlib import Path
from naive_train import train_file, train_pdf
from naive_test import test_file, test_pdf
from commons import MODEL_PATH, TRAIN_PATH, TEST_PATH
from colorama import Fore, Style, init

init(autoreset=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


# ---------------------- TRAIN ----------------------
def train(ngram, clean: bool = False):
    logging.info(f"Starting training | clean={clean}")
    # Load folder-to-label mapping
    with open("folder_to_label.json", "r") as file:
        f2l = json.load(file)
    proc_path = MODEL_PATH / "processed.json"
    proc = {}
    # Load already processed files (if not clean)
    if proc_path.exists() and not clean:
        with open(proc_path, "r") as file:
            proc = json.load(file)
    for folder, label in f2l.items():
        folder_path = TRAIN_PATH / folder
        # Process TXT files
        for txt_file in folder_path.glob("*.txt"):
            txt_file = str(txt_file)
            if txt_file not in proc:
                logging.info(f"Training TXT: {txt_file}")
                train_file(label, txt_file, clean, ngram)
                proc[txt_file] = True
        # Process PDF files
        for pdf_file in folder_path.glob("*.pdf"):
            pdf_file = str(pdf_file)
            if pdf_file not in proc:
                logging.info(f"Training PDF: {pdf_file}")
                train_pdf(label, pdf_file, clean, ngram)
                proc[pdf_file] = True
    # Save processed files
    with open(proc_path, "w") as file:
        json.dump(proc, file, indent=4)
    logging.info("Training complete")


# ---------------------- TEST ----------------------
def test(ngram, filepath: str):
    if not Path(filepath).exists():
        logging.error(f"File does not exist: {filepath}")
        return
    ext = filepath.split(".")[-1]
    if ext == "txt":
        return test_file(filepath, ngram)
    elif ext == "pdf":
        return test_pdf(filepath, ngram)
    else:
        logging.warning(f"Unsupported file extension: {ext}")


def main():
    parser = argparse.ArgumentParser(description="Train/Test Naive Model")
    parser.add_argument(
        "--mode", choices=["train", "test"], required=True, help="Mode: train or test"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean training (ignore processed.json)"
    )
    parser.add_argument("--ngram", required=True, help="N-grams to consider", type=int)
    parser.add_argument("--filepath", type=str, help="File path for testing")
    args = parser.parse_args()
    if args.mode == "train":
        train(args.ngram, clean=args.clean)
    elif args.mode == "test":
        if args.filepath:
            test(args.ngram, args.filepath)
        else:
            for file in TEST_PATH.glob("*"):
                filename = str(file).split("/")[-1]
                prediction = test(args.ngram, str(file))
                WIDTH = 25  # adjust based on longest filename
                print(
                    f"{Fore.CYAN}{Style.BRIGHT}{filename:<{WIDTH}}{Style.RESET_ALL}"
                    f"{Fore.GREEN}{Style.BRIGHT}: {prediction}{Style.RESET_ALL}"
                )
                print("\n\n")


if __name__ == "__main__":
    main()
    """
    Example 1:
    python3 main.py --mode train --clean --ngram 2
    
    Example 2:
    python3 main.py --mode train
    """
