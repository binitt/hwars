import logging
import sys
import json
from PIL import ImageGrab
import time
import pyautogui
import time
import requests
import io

from hwars import utils

def main():
    """Take screenshot, get buttons and follow instructions"""
    if len(sys.argv) == 2:
        cmd = json.loads(sys.argv[1])
    else:
        print(f"Args given: {len(sys.argv)}")
        samplecmd = """
{
  "repeat": 2, 
  "commands":[
    {"button":"Find", "index":2},
    {"button":"Start"},
    {"button":"To battle!"},
    {"button":"Return to the City"}
  ]
}
"""
        print(f"Usage: play '{samplecmd}'", file=sys.stderr)
        return
    try:
        play_cmd(cmd)
    except:
        logging.exception(f"Failed to run cmd: {cmd}")


def play_cmd(cmd):
    logging.info(f"Waiting for 2s before Alt-tab")
    time.sleep(2)
    pyautogui.hotkey('alt', 'tab')
    time.sleep(1)

    repeat = cmd.get("repeat", 1)
    for i in range(repeat):
        logging.info(f"Run: {i+1}/{repeat}")
        for command in cmd["commands"]:
            success = play_command(command)
            if not success:
                logging.error(f"Failed to complete at run {i+1}/{repeat}")
                pyautogui.hotkey('alt', 'tab') #revert
                return
    logging.info(f"Successfully completed all tasks")
    pyautogui.hotkey('alt', 'tab') #revert

def play_command(command):
    button, index, timeout = command["button"], command.get("index", 0), command.get("timeout", 2*60)
    logging.info(f"Running {button}[{index}] with timeout {timeout}s")

    end = time.time() + timeout
    success = False
    while time.time() < end:
        logging.info(f"Running another iteration")
        success = play_command_iter(button, index)
        if success:
            break
        time.sleep(2)
    logging.info(f"Returning with result: {success} for button: {button}")
    return success

def play_command_iter(button, index):
    """Grab screenshot, find the button and click
    Also wait for 2s if was able to click
    Return true if successful in clicking"""
    screenshot = ImageGrab.grab()
    img_bytes = io.BytesIO()
    screenshot.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    url = 'http://127.0.0.1:5000/buttons'
    files = {'file': img_bytes}

    response = requests.post(url, files=files)
    resp_json = response.text
    logging.info(f"Got resp: {resp_json}")

    resp_obj = json.loads(resp_json)
    candidate_index = 0
    success = False
    for candidate in resp_obj:
        if button_match(candidate[0], button):
            if candidate_index == index:
                send_click(candidate[1])
                time.sleep(2)
                success = True
                break
            else:
                candidate_index += 1
    if not success and candidate_index > 0:
        logging.warning(f"Matching buttons found {candidate_index}, expected index: {index}")
    return success

def button_match(pred_text_orig, real_text_orig):
    if pred_text_orig is None:
        return False
    real_text = real_text_orig.lower()
    if pred_text_orig.lower().find(real_text) != -1:
        return True
    pred_text = pred_text_orig.lower().replace("retumn", "return")
    if pred_text.find(real_text) != -1:
        return True
    pred_text = pred_text_orig.lower().replace("retum", "return")
    if pred_text.find(real_text) != -1:
        return True    
    return False
def send_click(box):
    """Input is x1,y1,x2,y2"""
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    pyautogui.moveTo(x=int(x), y=int(y), duration=0.3)
    pyautogui.click()

if __name__ == "__main__":
    utils.logging_init_stdout()
    logging.info(f"Play started")
    main()
    logging.info(f"Play finished")