from datasets import load_dataset
import logging
from transformers import AutoImageProcessor
import albumentations
import numpy as np
from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer

from hwars import utils

checkpoint = "facebook/detr-resnet-50"
# checkpoint = "Ultralytics/YOLOv8" #nw
# checkpoint = "nickmuchi/yolos-small-finetuned-license-plate-detection"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model_name = "binitt/hwars-buttons-model"

categories = ["button"]
id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

def main():
    dataset = load_dataset("binitt/hwars-buttons", split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    dataset["train"] = dataset["train"].with_transform(transform_aug_ann)
 
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=model_name,
        per_device_train_batch_size=4,
        num_train_epochs=1000,
        save_steps=500,
        logging_steps=50,
        learning_rate=6e-5,
        weight_decay=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        # save_strategy="no",
        fp16=True,
        # bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset["train"],
        tokenizer=image_processor,
    )

    logging.info("Training started")
    trainer.train()
    logging.info("Training finished")
    # trainer.push_to_hub()
    trainer.save_model('data/model/buttons')    

def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations

# transforming a batch
def transform_aug_ann(examples):
    transform = albumentations.Compose(
        [
            # albumentations.Resize(640, 360),
            albumentations.Resize(960, 600),
            # albumentations.HorizontalFlip(p=1.0),
            albumentations.RandomBrightnessContrast(p=1.0),
            # albumentations.RandomCrop(height=240, width=240),
        ],
        bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
    )

    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

if __name__ == "__main__":
    utils.logging_init_file()
    logging.info(f"Train started")
    main()
    logging.info(f"Train finished")