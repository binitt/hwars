import logging
import json
from PIL import Image
import os
from datasets import load_dataset

from hwars import utils
from hwars.cfg import Cfg

def main():
    """https://huggingface.co/docs/datasets/image_dataset#imagefolder
    """
    create_dataset()
    upload_hf()

def upload_hf():
    dataset = load_dataset("imagefolder", data_dir="data/buttons", split="train")
    logging.info(f"Pushing to hub")
    dataset.push_to_hub("binitt/hwars-buttons", token=Cfg.HF_TOKEN)
    logging.info(f"Pushed to hub")

def create_dataset():
    from_file = r'data/buttons/hwars.json'
    with open(from_file, "r") as f:
        from_obj = json.load(f)
    
    to_file = r'data/buttons/metadata.jsonl'
    with open(to_file, "w") as f:
        i = 0
        for from_entry in from_obj['_via_img_metadata'].values():
            i += 1
            write_entry(f, from_entry)
    logging.info(f"Wrote {i} entries into {to_file}")

category_map = {
    "button": 0,
    "cross": 1,
}
image_id = 0
object_id = 0
def write_entry(f, from_entry):
    global image_id, object_id

    imfile = os.path.join('data/buttons', from_entry['filename'])
    with Image.open(imfile) as img:
        width, height = img.size

    to_entry = {
        "file_name": from_entry['filename'],
        "image_id": image_id,
        "width": width,
        "height": height,
        "objects": {
            "id": [],
            "area": [],
            "bbox": [],
            "category": []
        }
    }

    for r in from_entry['regions']:
        sa = r['shape_attributes']
        ra = r['region_attributes']
        if sa['name'] != 'rect':
            raise Exception(f'Only rect is supported but got {sa["name"]}')
        w,h = sa['width'], sa['height']
        to_entry["objects"]["id"].append(object_id)
        to_entry["objects"]["area"].append(w * h)
        to_entry["objects"]["bbox"].append([sa['x'], sa['y'], w, h]) 
        to_entry["objects"]["category"].append(category_map[ra['type']])
        object_id += 1

    f.write(json.dumps(to_entry) + "\n")

    image_id += 1

if __name__ == "__main__":
    utils.logging_init_stdout()
    logging.info("Converting to dataset started")
    main()
    logging.info("Converting to dataset finished")
