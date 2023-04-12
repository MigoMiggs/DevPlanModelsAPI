# app.py
from flask import Flask, jsonify, request
from transformers import pipeline
import torch

import logging
app = Flask(__name__)

# Configure logging settings
logging.basicConfig(filename='app.log', level=logging.DEBUG)

sentiment_pipeline = pipeline("sentiment-analysis")
zeroshot_risks = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


## risk configs
risk_labels= {
    "RISK-TIME-DELAY": "project deadline risk",
    "RISK-TEAM-RESOURCES": "team resources risk",
    "RISK-SCOPE": "proejct scope change risk"
}

risks = [
    "RISK-TIME-DELAY", "RISK-TEAM-RESOURCES", "RISK-SCOPE"
]

@app.route('/DevPlanAPI/getSentimentStatusText', methods=['POST'])
def getSentimentStatusText():
    probabilities = []
    text_array = request.json['status']
    for text in text_array:
        probability = sentiment_pipeline(text)
        probabilities.append(probability)

    return {'data': probabilities}

@app.route('/DevPlanAPI/getRiskProbabilities', methods=['POST'])
def get_risk_probabilities():
    # Get the list of strings from the JSON request
    text_list = request.json['status']

    concatenated_status = ""

    for status in text_list:
        concatenated_status += status + " "

    risk_probs = {}
    risk_probs = zeroshot_risks(concatenated_status, risks)

    return {'risk_probs': risk_probs}

if __name__ == '__main__':
    app.run(debug=True)