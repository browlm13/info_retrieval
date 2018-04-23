#!/usr/bin/env python

__author__ = "L.J. Brown"
__version__ = "1.0.1"

import logging
import string
import sys
from urllib.parse import urljoin

# external
import nltk
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("output/output_log.txt"))
logger.addHandler(logging.StreamHandler(sys.stdout))

# Text Processing


def normalize_urls(base_url, raw_links):
    """ takes in list of raw links both relative and absolute and returns list of absolute links. """
    raw_links = [l for l in raw_links if l is not None]
    absolute_links = [l if l.startswith('http') else urljoin(base_url, l) for l in raw_links]
    return list(absolute_links)


def extract_words(sentence):
    """ Extract words from raw document and return list """
    words = nltk.word_tokenize(sentence)
    return words


def is_stopword(word):
    """ Return True of word is in stop word list """
    stop_words = nltk.corpus.stopwords.words('english')
    return word in stop_words


def list_from_stopwords_file(stopwords_file):
    plain_text = None
    with open(stopwords_file) as sf:
        plain_text = sf.read()
    return plain_text_to_tokens(plain_text)


def is_punctuation(word):
    """ Return True if word is composed entirly of punctuation and whitespace """
    if set(word) < set(string.punctuation + string.whitespace):
        return True
    return False


def is_number(word):
    """ Return True if word is number """
    try:
        float(word)
        return True
    except ValueError:
        logger.debug('ValueError is_number')
    try:
        import unicodedata
        unicodedata.numeric(word)
        return True
    except (TypeError, ValueError):
        logger.debug('ValueError is_number')
    return False


def is_shorter(word, n=3):
    """ Return True if word is shorter than n """
    if len(word) < n:
        return True
    return False


def stem(word):
    """ stem word with PorterStemmer """
    ps = PorterStemmer()
    return ps.stem(word)


def alphabetic_start(word):
    """ Return True if word starts with an alphabetic character """
    if word[0].isalpha():
        return True
    return False


def alphanumeric_end(word):
    """ Return True if word ends with an alphanumeric (letter or number) character """
    if word[-1].isalnum():
        return True
    return False


def plain_text_to_tokens(plain_text, stopwords_file=None):
    """ tokenize sentence, convert to lower, stem, remove stop words, numbers, punctuation"""
    logger.debug('Cleaning Plain Text')

    assert plain_text is not None

    # tokenize sentence
    tokens = extract_words(plain_text)

    # lower
    tokens = [t.lower() for t in tokens]

    # remove words with no alphabetic start characters
    tokens = [w for w in tokens if alphabetic_start(w)]

    # remove words with no alphanumeric end characters
    tokens = [w for w in tokens if alphanumeric_end(w)]

    # remove stop words
    if stopwords_file is None:
        tokens = [w for w in tokens if not is_stopword(w)]
    else:
        stopwords = list_from_stopwords_file(stopwords_file)
        tokens = [w for w in tokens if w not in stopwords]

    # remove punctuation
    tokens = [w for w in tokens if not is_punctuation(w)]

    # remove short
    tokens = [w for w in tokens if not is_shorter(w)]

    # remove number
    tokens = [w for w in tokens if not is_number(w)]

    # stem words
    tokens = map(stem, tokens)

    return list(tokens)


# Text Statistics

def word_frequency_dict(tokens):
    """ returns a dictionary of word and their assosiated frequencies from token list """

    fdist = FreqDist(tokens)  # fdist.keys() fdist.values()
    return dict(fdist)
