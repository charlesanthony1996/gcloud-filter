import json

from .filterClasses import TextProcessor

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

english_processor = TextProcessor(config["models"][config["active_model"]])
german_processor = TextProcessor(config["models"][config["active_model2"]])

__all__ = ['english_processor', 'german_processor', 'config']