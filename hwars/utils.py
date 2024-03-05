import logging
import logging.handlers
import sys
import io
import requests
import json

def logging_init_stdout():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def logging_init_file():
    root_logger= logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(
            'logs/trace.log', maxBytes=50 * 1024 * 1024, 
            backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter(
            '%(levelname)s %(threadName)s %(asctime)s %(filename)s:%(lineno)d %(funcName)s %(message)s'))
    root_logger.addHandler(handler)
    root_logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Logger initialized")

def text_from_image(image):
    """Takes PIL Image, invokes API call and returns matches array"""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    url = 'http://binit2:15000/buttons'
    files = {'file': img_bytes}

    response = requests.post(url, files=files)
    resp_json = response.text
    logging.info(f"Got resp: {resp_json}")

    resp_obj = json.loads(resp_json)
    return resp_obj
