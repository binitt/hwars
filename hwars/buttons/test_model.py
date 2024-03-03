from datasets import load_dataset
import torch
from transformers import AutoImageProcessor
from transformers import AutoModelForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import logging
import sys
import os
import json
import pandas as pd

from hwars import utils

model_name = "binitt/hwars-buttons-model"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(model_name)

def main():
    """We are trying to optimize the recall, i.e. we should get all the known BB
    But if we are identifying false positives we don't care that much"""
    from_file = r'data/buttons/metadata.jsonl'
    metrics = {}
    with open(from_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            logging.info(f"Processing for {obj['file_name']}, {obj['objects']['bbox']}")
            imfile = os.path.join('data/buttons', obj['file_name'])
            with open(imfile, "rb") as f:
                image = Image.open(f)
                metrics[imfile] = estimate(image, obj['objects']['bbox'])
    show_output(metrics)

def estimate(image, bbox):
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]

    metrics = []
    for real_box in bbox:
        correct = 0
        for pred_box in results["boxes"]:
            if check_box(real_box, pred_box):
                correct = 1
                break
        metrics.append({
                "bbox": real_box,
                "result": correct, # correct(1) or incorrect(0)
            })
    return metrics

def check_box(real, pred):
    """We check by checking if mid point of real is in pred
    format is [x,y,w,h]"""
    mid = (real[0] + int(real[2] / 2), real[1]  + real[3] / 2)
    if mid[0] > pred[0] and mid[0] < pred[2] and mid[1] > pred[1] and mid[1] < pred[3]:
        return True
    else:
        return False

def show_output(metrics):
    display_data = {
        "imfile": [],
        "correct": [],
        "incorrect": []
    }
    total_correct = 0
    total_incorrect = 0
    for file,metric in metrics.items():
        display_data["imfile"].append(file)
        correct,incorrect = 0,0
        for item in metric:
            if item["result"] == 1:
                correct += 1
            else:
                incorrect += 1

        display_data["correct"].append(correct)
        display_data["incorrect"].append(incorrect)
        total_correct += correct
        total_incorrect += incorrect

    df = pd.DataFrame(display_data)
    print(f"=============\nSummary: \n {str(df)}")
    acc = total_correct / (total_correct + total_incorrect) * 100
    print(f"=============\nCorrect: {total_correct}, Incorrect: {total_incorrect}, Acc: {acc}%")

if __name__ == "__main__":
    utils.logging_init_stdout()
    logging.info(f"Test started")
    main()
    logging.info(f"Test finished")