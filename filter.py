import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from langdetect import detect
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TextProcessor:
    def __init__(self, model_config):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_config["model_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"])
        self.threshold = model_config["threshold"]
        self.preprocess_config = model_config["preprocess"]

    def preprocess(self, text):
        if self.preprocess_config["lower_case"]:
            text = text.lower()
        if self.preprocess_config["remove_punctuation"]:
            text = ''.join(char for char in text if char.isalnum() or char.isspace())
        return text

    def process_text(self, text):
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = self.model(**encoded_input)
        logits = output.logits
        scores = torch.nn.functional.softmax(logits, dim=-1).numpy()
        negativity_score = scores[0][1]

        if negativity_score > self.threshold:
            return True, text, negativity_score
        else:
            return False, text, 0


app = Flask(__name__)
app.debug = True
CORS(app, resources={r"/api/*": {"origins": "*"}})


with open('config.json', 'r') as config_file:
    config = json.load(config_file)


english_processor = TextProcessor(config["models"][config["active_model"]])
german_processor = TextProcessor(config["models"][config["active_model2"]])

def detect_language(text):
    return detect(text)

def direct_test(sentence):
    language = detect_language(sentence)
    print("Detected Language:", language)
    if language == "en":
        filtered, processed_text, score = english_processor.process_text(sentence)
        print(f"Sentence: '{sentence}'\nFiltered: {filtered}\nNegativity Score: {score}\n")
    elif language == "de":
        filtered, processed_text, score = german_processor.process_text(sentence)
        print(f"Sentence: '{sentence}'\nFiltered: {filtered}\nNegativity Score: {score}\n")

@app.route('/')
def home():
    return jsonify({"message": "Filter Service Running"})

@app.route("/api/test", methods=["POST"])
def filter_text():
    try:
        data = request.json
        text = data.get("text", '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        language = detect_language(text)
        print("Detected Language:", language)
        
        if language == "en":
            filtered, processed_text, score = english_processor.process_text(text)
        elif language == "de":
            filtered, processed_text, score = german_processor.process_text(text)
        else:
            return jsonify({"filtered_text": "Unsupported language", "negativity_score": 0})

        response = {
            "filtered_text": processed_text if filtered else "Is not HS",
            "negativity_score": float(score)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/test_db_communication", methods=["GET"])
def test_db_communication():
    try:
        db_service_url = 'https://gcloud-server-db-met2pwr7xq-uc.a.run.app:5001/api/test'
        payload = {"test_message": "Hello from Text Processor Service"}
        response = requests.post(db_service_url, json=payload)
        if response.status_code == 200:
            return jsonify({"message": "Successfully communicated with DB service", "db_response": response.json()})
        else:
            return jsonify({"error": "Failed to communicate with DB service"}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # print("Before!!")
    # current_filter = config["active_model"]
    # print(f"Currently using filter: {current_filter}")
    # current_filter2 = config["active_model2"]
    # print(f"Currently using filter: {current_filter2}")
    # print("After")


    # direct_test("you are a very horrible person")
    # direct_test("""
    #             Dwight Howard ist eine verdeckte Schwuchtel. 
    #             Dieser Typ wurde vor der Kamera erwischt und Isiah Canaan Penis streichelte, 
    #             als Sie fÃ¼r die Rockets spielten.Dwights Karriere ist vorbei, 
    #             weil er weicher als Marmelade ist und keinen Strandball in einen 
    #             Ozean schieÃŸen konnte. Hasse dich bitch!
    #             """)

    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)