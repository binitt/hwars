import os
import logging
import time
from ultralytics import YOLO

from hwars import utils

def main():
    model = YOLO("yolo11n.pt")

    start_at = time.time()
    train_results = model.train(
        data="data/yolo/buttons.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  #1024 hangs, def: 640 # training image size
        device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        # device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )
    end_at = time.time()
    logging.info(f">>>>>>>>>Time taken {end_at - start_at}s")
    #logging.info(f"Results: {train_results}")

    metrics = model.val()
    #logging.info(f"Metrics: {metrics}")
    
    os.makedirs("data/yolo/models", exist_ok=True)
    model.save("data/yolo/models/hwars.pt")

if __name__ == "__main__":
    utils.logging_init_file()
    logging.info(f"Train started")
    main()
    logging.info(f"Train finished")
