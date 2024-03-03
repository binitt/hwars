from flask import Flask, request
from PIL import Image
import io
import json
import logging

from hwars.buttons.extract_text import load_models, locate_buttons
from hwars import utils

app = Flask(__name__)

@app.route('/buttons', methods=['POST'])
def handle_request():
    logging.info(f"Got new request")
    try:
        http_resp = find_buttons()
    except Exception as e:
        http_resp = json.dumps({'error': str(e)}), 500
    
    logging.info(f"Sending response: {http_resp}")
    return http_resp

def find_buttons():
    if 'file' not in request.files:
        return json.dumps({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return json.dumps({'error': 'No file given'}), 400

    file_data = file.read()
    img = Image.open(io.BytesIO(file_data))
    resp = locate_buttons(img)
    return json.dumps(resp)

if __name__ == '__main__':
    utils.logging_init_file()
    load_models()
    app.run(debug=True)
