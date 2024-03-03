from datasets import load_dataset
import torch
from transformers import AutoImageProcessor
from transformers import AutoModelForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import logging
import sys
import os

from hwars import utils

def main():
    if len(sys.argv) == 2:
        imfile = sys.argv[1]
    else:
        imfile = r"data/buttons/ss-7-victory.png"
    
    if not os.path.exists(imfile):
        logging.error(f"File not found: {imfile}")
    
    logging.info(f"Processing file: {imfile}")

    with open(imfile, "rb") as f:
        image = Image.open(f)
        process_image(image)

def process_image(image):
    model_name = "binitt/hwars-buttons-model"
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name)

    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]

    i = 0
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"{i}: Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
        i += 1

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
    image.show()

if __name__ == "__main__":
    utils.logging_init_stdout()
    logging.info(f"Apply started")
    main()
    logging.info(f"Apply finished")