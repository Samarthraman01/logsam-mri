"""
nlp_module.py
"""
import spacy
from negspacy.negation import Negex
from typing import Tuple 
from src.utils.data_utils import PROMPT_MAP

SYNONYM_MAP = {
    "glioblastoma"  : "glioma",
    "astrocytoma"   : "glioma",
    "oligodendroglioma" : "glioma",
    "meningeal"     : "meningioma",
    "meningothelial": "meningioma",
    "macroadenoma"  : "pituitary",
    "microadenoma"  : "pituitary",
    "adenoma"       : "pituitary",
}

NEGATION_PHRASES = [
    "no evidence of",
    "without",
    "absence of",
    "no signs of",
    "normal mri",
    "normal study",
    "unremarkable",
    "no tumor",
    "no mass",
]
#function takes in a sentense adn then returns a Tuple of three output which is a string, bool and a str again
def extract_tumor_info(sentence: str) -> Tuple[str, bool, str]:
    sentence = sentence.lower()
    negated = False
    for phrase in NEGATION_PHRASES:
        if phrase in sentence:
            negated = True
            break

    tumor_class = "healthy"
    for word in sentence.split():
        if word in SYNONYM_MAP:
            tumor_class = SYNONYM_MAP[word]
            break
        if word in PROMPT_MAP:
            tumor_class = word
            break
    if negated:
        prompt = PROMPT_MAP["healthy"]
    else:
        prompt = PROMPT_MAP[tumor_class]

    return tumor_class, negated, prompt

            

