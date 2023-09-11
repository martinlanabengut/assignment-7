import re
import subprocess
import unicodedata
from typing import List, Optional

import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer

from src.contractions import CONTRACTION_MAP

# Download the models used
nltk.download("stopwords")
nltk.download("punkt")
subprocess.run(["spacy", "download", "en_core_web_sm"])

# Load NLP models
tokenizer = ToktokTokenizer()
nlp = spacy.load("en_core_web_sm")
stopword_list = nltk.corpus.stopwords.words("english")


def remove_html_tags(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def stem_text(text: str) -> str:
    stemmer = nltk.porter.PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_tokens)


def remove_accented_chars(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_special_chars(text: str, remove_digits: Optional[bool] = False) -> str:
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    return re.sub(pattern, '', text)


def remove_stopwords(text: str, is_lower_case: Optional[bool] = False,
                     stopwords: Optional[List[str]] = stopword_list) -> str:
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    return ' '.join(filtered_tokens)


def remove_extra_new_lines(text: str) -> str:
    return re.sub(r'[\r|\n|\r\n]+', ' ', text)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP) -> str:
    """
    Expand english contractions on input string.

    Args:
        text : str
            Input string.
    Return:
        str
            Output string.
    """
    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = (
            contraction_mapping.get(match)
            if contraction_mapping.get(match)
            else contraction_mapping.get(match.lower())
        )
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    text = re.sub("'", "", expanded_text)

    return text


def normalize_corpus(
        corpus: List[str],
        html_stripping: Optional[bool] = True,
        contraction_expansion: Optional[bool] = True,
        accented_char_removal: Optional[bool] = True,
        text_lower_case: Optional[bool] = True,
        text_stemming: Optional[bool] = False,
        text_lemmatization: Optional[bool] = False,
        special_char_removal: Optional[bool] = True,
        remove_digits: Optional[bool] = True,
        stopword_removal: Optional[bool] = True,
        stopwords: Optional[List[str]] = stopword_list,
) -> List[str]:
    """
    Normalize list of strings (corpus)

    Args:
        corpus : List[str]
            Text corpus.
        html_stripping : bool
            Html stripping,
        contraction_expansion : bool
            Contraction expansion,
        accented_char_removal : bool
            accented char removal,
        text_lower_case : bool
            Text lower case,
        text_stemming : bool
            Text stemming,
        text_lemmatization : bool
            Text lemmatization,
        special_char_removal : bool
            Special char removal,
        remove_digits : bool
            Remove digits,
        stopword_removal : bool
            Stopword removal,
        stopwords : List[str]
            Stopword list.

    Return:
        List[str]
            Normalized corpus.
    """

    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)

        # Remove extra newlines
        doc = remove_extra_new_lines(doc)

        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        # Expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)

        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)

        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)

        # Remove special chars and\or digits
        if special_char_removal:
            doc = remove_special_chars(doc, remove_digits=remove_digits)

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

        # Lowercase the text
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc, is_lower_case=text_lower_case, stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()

        normalized_corpus.append(doc)

    return normalized_corpus
