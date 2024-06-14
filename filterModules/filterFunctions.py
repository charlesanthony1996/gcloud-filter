import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
import os

from .__init__ import english_processor, german_processor

def detect_language(text):
    # Detect the language of the input text
    return detect(text)

def direct_test(sentence):
    language = detect_language(sentence)
    print("Detected Language:", language)
    # Process English text
    if language == "en":
        filtered, processed_text, score = english_processor.process_text(sentence)
        print(f"Sentence: '{sentence}'\nFiltered: {filtered}\nNegativity Score: {score}\n")
    # Process German text
    elif language == "de":
        filtered, processed_text, score = german_processor.process_text(sentence)
        print(f"Sentence: '{sentence}'\nFiltered: {filtered}\nNegativity Score: {score}\n")