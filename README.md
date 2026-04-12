# Contents
1. [About](#about)
2. [Project structure](#project-structure)
3. [How to run the code](#how-to-run-the-code)
4. [Adding a new language](#adding-a-new-language)
5. [Challenges](#challenges)

## About 
This project is used to automatically detect the language in a document. Note that detecting a language is not the same as detecting the script because multiple languages can share the same script.

We can represent this task as a conditional probability
$$p(E_0|w_0 w_1... w_N) = \frac{p(E_0) \cdot p(w_0 w_1... w_N|E_0)}{p(w_0 w_1... w_N)}$$
where,
$E_0$ ~ Is a language category
$w_0 w_1...$ ~ Represents words in a text

If we divide the above equation with another language $E_1$, we get the following equation
$$\frac{p(E_0|w_0 w_1... w_N)}{p(E_1|w_0 w_1... w_N)} = \frac{p(w_0 w_1... w_N|E_0)}{p(w_0 w_1... w_N|E_1)}$$
provided we assume.
$p(E_0) = p(E_1)$ i.e. the expectation of getting documents of different languages are equally likely.
We can also use p(E_0) = \frac{# of people who speak language }{} which is a proxy for , if we use this expectations this means the likelihood our model gets this document is proportional to the number of people in the world who speak this language. 


## How to run the code

### 1. Activate the environment
```bash
uv sync
source .venv/bin/activate
```

### 2. Run the code

To explore all the options/modes to run the code paste the following command. 
```python
python3 main.py --help
```

There are two modes to run this code:
1. **train mode** — Trains the model using the corpus
2. **test mode** — Uses the trained model for inference

In train mode, you can run it in two ways. By activating the `--clean` option, the model is trained from scratch. Without it, the model continues training on existing data.
In test mode, the trained model is used to identify languages in new documents.

```python
    Example 1:
    python3 main.py --mode train --clean --ngram 2
    
    Example 2:
    python3 main.py --mode train
    
    Example 3:
    python3 main.py --mode test
```

## Adding a new language
Broadly speaking, there are about 7,168 living languages in the world today. However, linguistic diversity is heavily skewed: just 23 languages account for more than half of the world's population.
I'll take the top 10 languages

Rank,Language,Total Speakers,Primary Script
1,English,1.5 Billion,Latin
2,Mandarin Chinese,1.2 Billion,Han (Simplified/Traditional)
3,Hindi,629 Million,Devanagari
4,Spanish,590 Million,Latin
5,French,416 Million,Latin
6,Modern Arabic,335 Million,Arabic
7,Portuguese,282 Million,Latin
8,Bengali,278 Million,Bengali-Assamese

## Project structure
- **train-data/** — Language-specific corpus for training models
- **test-data/** — Sample documents for testing language identification
- **model/** — Contains trained language models (frequency data for 1-gram and 2-gram)
- **main.py** — Primary script to run train or test modes
- **commons.py** — Shared utilities and helper functions
- **pyproject.toml** — Project configuration and dependencies

## Challenges
1. A document may contain multiple languages, requiring a percentage-based representation of detected languages.
2. With many languages available, should focus on top popular languages or support all?
3. PDF documents come in two types: text-based and image-based. Currently, text-based PDFs are supported.    


The "Middle" Way: Script Detection (Unicode)
Before identifying the language (e.g., "Is this Spanish or Italian?"), you can identify the Script (e.g., "Is this Latin or Cyrillic?"). Every character has a "Script" property in the Unicode Standard.

Unicode Ranges: Characters are organized into blocks. For example:

U+0000 to U+007F: Basic Latin (English, etc.)

U+0600 to U+06FF: Arabic

U+0900 to U+097F: Devanagari (Hindi)

How it works: You iterate through the characters of your string and check which Unicode block they fall into. If 90% of characters are in the Devanagari block, you know the script is Devanagari, which narrows the language down to Hindi, Marathi, Nepali, etc.


Language and scripts are different. 
So simple identification of script will not work if you check the unicode ranges.

