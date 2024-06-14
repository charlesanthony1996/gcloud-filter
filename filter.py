import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from langdetect import detect
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restx import Api, Resource, fields
import os

import filterModules
from filterModules import english_processor, german_processor, config
from filterModules.filterClasses import TextProcessor
from filterModules.filterFunctions import detect_language, direct_test

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
app.debug = True
CORS(app, resources={r"/api/*": {"origins": "*"}})

api = Api(app, version='1.0', title='Text Processing API',
          description='A simple Text Processing API')
ns = api.namespace('api', description='Text operations')

text_model = api.model('TextModel', {
    'text': fields.String(required=True, description='Text to be processed')
})

response_model = api.model('ResponseModel', {
    'filtered_text': fields.String(description='Filtered text or message'),
    'negativity_score': fields.Float(description='Negativity score')
})

@ns.route('/test')
class TextFilter(Resource):
    @ns.expect(text_model)
    @ns.marshal_with(response_model)
    def post(self):
        data = request.json
        text = data.get("text", '')

        if not text:
            api.abort(400, "No text provided")

        language = detect_language(text)
        print("Detected Language:", language)
        
        if language == "en":
            filtered, processed_text, score = english_processor.process_text(text)
        elif language == "de":
            filtered, processed_text, score = german_processor.process_text(text)
        else:
            return {"filtered_text": "Unsupported language", "negativity_score": 0}

        response = {
            "filtered_text": processed_text if filtered else "Is not HS",
            "negativity_score": float(score)
        }
        return response

if __name__ == "__main__":

    direct_test("you are a very horrible person")
    direct_test("""
                Dwights Karriere ist vorbei, weil er weicher als Marmelade ist 
                und keinen Strandball in einen 
                Ozean schieÃŸen konnte. Hasse dich bitch!
                """)

    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)