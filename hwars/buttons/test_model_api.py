from PIL import Image, ImageDraw, ImageFont
import logging
import sys
import os
from PIL import ImageGrab
import pyautogui
import time

from hwars import utils

def main():
    if len(sys.argv) == 2:
        imfile = sys.argv[1]
        if not os.path.exists(imfile):
            logging.error(f"File not found: {imfile}")
        
        logging.info(f"Processing for file: {imfile}")
        image = Image.open(imfile)
    else:
        pyautogui.hotkey('alt', 'tab')
        time.sleep(1)
        image = ImageGrab.grab()
        time.sleep(2)
        pyautogui.hotkey('alt', 'tab')        
        logging.info(f"Processing for screenshot")

    process_image(image)

def process_image(image):
    resp_obj = utils.text_from_image(image)
    i = 0
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 20)

    for entry in resp_obj:
        txt, box = entry
        if txt is None: txt = ""
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{i}: {txt}", fill="white", font=font)
        i += 1        
    image.show()

if __name__ == "__main__":
    utils.logging_init_stdout()
    logging.info(f"Test started")
    main()
    logging.info(f"Test finished")