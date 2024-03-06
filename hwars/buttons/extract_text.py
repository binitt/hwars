from datasets import load_dataset
import torch
from transformers import AutoImageProcessor
from transformers import AutoModelForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import logging
import sys
import os
import numpy as np
import easyocr


from hwars import utils
from hwars.cfg import Cfg

image_processor = None
model = None
reader = None
def load_models():
    global image_processor, model, reader
    logging.info(f'Loading models')
    model_name = "binitt/hwars-buttons-model"
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    reader = easyocr.Reader(['en'])
    logging.info(f"Models loaded")

def main():
    if len(sys.argv) == 2:
        imfile = sys.argv[1]
    else:
        imfile = r"data/buttons/ss-7-victory.png"
    
    if not os.path.exists(imfile):
        logging.error(f"File not found: {imfile}")
    
    load_models()
    logging.info(f"Processing file: {imfile}")

    with open(imfile, "rb") as f:
        image = Image.open(f)
        results = locate_buttons(image)
    logging.info(f"Got results: {results}")
    image.show()

def extract_text(image):
    """Extract text from button image. Only 1 is expected
    If >1 is found then choose with biggest length"""
    global reader

    results = reader.readtext(np.array(image))

    if len(results) > 1:
        logging.warning(f"Got more than 1 text results: {results}")
    elif len(results) == 0:
        return None
    txt = ""
    for result in results:
        if len(result[1]) > len(txt):
            txt = result[1]
    return txt

def locate_buttons(image):
    """Takes PIL Image and returns:
    [text, [x1,y1,x2,y2]]
    Data are sorted by (x1+y1)"""
    global image_processor, model
    buttons = []
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]

    i = 0
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = model.config.id2label[label.item()]
        box = [round(i, 2) for i in box.tolist()]
        box[2] += 50 # manual adjustment
        if label == 'button':
            button_image = image.crop(box)
            button_text = extract_text(button_image)
        else:
            button_text = label

        logging.info(
            f"{i}: Detected {label} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
        i += 1
        buttons.append([button_text, box])
        

    if Cfg.DEBUG_MODE:
        logging.info(f"Debug mode enabled")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 20)

        i = 0
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = tuple(box)
            x2 += 50 # manual adjustment
            draw.rectangle((x, y, x2, y2), outline="red", width=3)
            draw.text((x, y), f"{i}: {model.config.id2label[label.item()]}", fill="white", font=font)
            i += 1
        tmpfile = "logs/tmpimage.png"
        image.save(tmpfile)
        logging.info(f"Saved to {tmpfile}")
    
    buttons = sorted(buttons, key=lambda x:x[1][0]+x[1][1])
    return buttons

if __name__ == "__main__":
    utils.logging_init_stdout()
    logging.info(f"extract started")
    main()
    logging.info(f"extract finished")