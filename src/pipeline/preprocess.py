import os
import re
import string
import sys

import joblib

try:
    from utils.config import MODELS_DIR, ROOT_DIR, TOKENIZER_NAME
except ImportError:
    from ..utils.config import MODELS_DIR, ROOT_DIR, TOKENIZER_NAME

TOKENIZER_PATH = ROOT_DIR / MODELS_DIR / TOKENIZER_NAME


def clean_text(text: str) -> str:

    """To make this function faster, load all loadables outside function
    and just pass to function -> To-Do
    """

    text = text.strip().lower()

    # special strings like \n, \t to be replaced with a single space
    special_strings = r"\n+|\t+\|\r+|\f+|\v+"
    clean_text = re.sub(special_strings, " ", text)

    # remove punctuations at edge of text
    start_end_puncts = (
        r"^[" + string.punctuation + "]+" + "|[" + string.punctuation + "]+$"
    )
    clean_text = re.sub(start_end_puncts, " ", clean_text)

    # remove more than 2 spaces, mostlry for where there is multiple spaces between text
    strip_spaces = r"\s{2,}"
    clean_text = re.sub(strip_spaces, " ", clean_text)

    # remove numbers
    numbers_pattern = r"\d+"
    clean_text = re.sub(numbers_pattern, " ", clean_text)

    # add pattern for website and e-mail, do this before removing any punctuation

    # try removing text between 2 paranthesis

    # remove punctuations
    # r'\s+[' + string.punctuation + ']+' + '|[' + string.punctuation + ']+$\s+'
    puncts_pattern = (
        r"[" + string.punctuation.replace("'", "") + "]+"
    )  # remove ' from puncts to match
    clean_text = re.sub(puncts_pattern, " ", clean_text)

    # replace more than 2 spaces with a single space, mostly for where there is multiple spaces between text
    # use this also as the last cleaning method
    strip_spaces = r"\s{2,}"
    clean_text = re.sub(strip_spaces, " ", clean_text)
    return clean_text.strip()


tokenizer = joblib.load(TOKENIZER_PATH)
