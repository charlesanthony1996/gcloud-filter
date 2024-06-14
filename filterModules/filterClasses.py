import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextProcessor:
    def __init__(self, model_config):
        # Initialize model and tokenizer from pre-trained model path
        self.model = AutoModelForSequenceClassification.from_pretrained(model_config["model_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"])
        self.threshold = model_config["threshold"]
        self.preprocess_config = model_config["preprocess"]

    def preprocess(self, text):
        # Convert to lower_case and/or remove punctuation
        if self.preprocess_config["lower_case"]:
            text = text.lower()
        if self.preprocess_config["remove_punctuation"]:
            text = ''.join(char for char in text if char.isalnum() or char.isspace())
        return text

    def process_text(self, text):
        # Steps short
        # Preprocess the input text, tokenize it, disable gradient computation for inference
        # Get model output, extract logits (raw predictions), apply softmax get probabilities
        # Get the negativity score
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = self.model(**encoded_input)
        logits = output.logits
        scores = torch.nn.functional.softmax(logits, dim=-1).numpy()
        negativity_score = scores[0][1]
        
        # Determine if the negativity score exceeds the threshold
        if negativity_score > self.threshold:
            return True, text, negativity_score
        else:
            return False, text, 0
        