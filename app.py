import os
import fitz  # PyMuPDF
from openai import AzureOpenAI
import json
from flask_cors import CORS
from flask import Flask, request, Response, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import pytesseract
from PIL import Image
from werkzeug.utils import secure_filename
import re
import pycountry
# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Handle CORS
# Azure OpenAI Client Initialization

@app.route("/")
def hello():
    return "Hello, World from Flask in Azure!"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
