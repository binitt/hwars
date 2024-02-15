from PIL import ImageGrab
import time
import pyautogui
import time

pyautogui.hotkey('alt', 'tab')

#give time to user to switch
time.sleep(1)

screenshot = ImageGrab.grab()

screenshot.save("logs/screenshot.png")

screenshot.show()