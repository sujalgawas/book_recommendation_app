from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/data', methods=['GET', 'POST'])
def data():
    return jsonify({
        "title": "haryy",
        "description": "this is test",
        "img": "https://images.ctfassets.net/usf1vwtuqyxm/6S51pK7uwnyhkS9Io9DsAn/320c162c5150f853b8d8568c4715dcef/English_Harry_Potter_7_Epub_9781781100264.jpg?w=914&q=70&fm=jpg",
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
