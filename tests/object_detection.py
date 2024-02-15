# -*- coding: utf-8 -*-

from datasets import load_dataset

cppe5 = load_dataset("cppe-5")

import numpy as np
import os
from PIL import Image, ImageDraw

image = cppe5["train"][0]["image"]
annotations = cppe5["train"][0]["objects"]
draw = ImageDraw.Draw(image)

categories = cppe5["train"].features["objects"].feature["category"].names

id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

for i in range(len(annotations["id"])):
    box = annotations["bbox"][i - 1]
    class_idx = annotations["category"][i - 1]
    x, y, w, h = tuple(box)
    draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
    draw.text((x, y), id2label[class_idx], fill="white")

remove_idx = [590, 821, 822, 875, 876, 878, 879]
keep = [i for i in range(len(cppe5["train"])) if i not in remove_idx]
cppe5["train"] = cppe5["train"].select(keep)



from transformers import AutoImageProcessor

checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)



import albumentations
import numpy as np
import torch

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)

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


cppe5["train"] = cppe5["train"].with_transform(transform_aug_ann)

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

from transformers import AutoModelForObjectDetection

model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="detr-resnet-50_finetuned_cppe5",
    per_device_train_batch_size=1, #[B] was 8
    num_train_epochs=10,
    fp16=True,
    save_steps=200,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
    hub_token="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"
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=cppe5["train"],
    tokenizer=image_processor,
)

trainer.train()

"""If you have set `push_to_hub` to `True` in the `training_args`, the training checkpoints are pushed to the
Hugging Face Hub. Upon training completion, push the final model to the Hub as well by calling the [push_to_hub()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.push_to_hub) method.
"""

trainer.push_to_hub()
trainer.save_model("data/model/hwars.ckpt")


# import json


# # format annotations the same as for training, no need for data augmentation
# def val_formatted_anns(image_id, objects):
#     annotations = []
#     for i in range(0, len(objects["id"])):
#         new_ann = {
#             "id": objects["id"][i],
#             "category_id": objects["category"][i],
#             "iscrowd": 0,
#             "image_id": image_id,
#             "area": objects["area"][i],
#             "bbox": objects["bbox"][i],
#         }
#         annotations.append(new_ann)

#     return annotations


# # Save images and annotations into the files torchvision.datasets.CocoDetection expects
# def save_cppe5_annotation_file_images(cppe5):
#     output_json = {}
#     path_output_cppe5 = f"{os.getcwd()}/cppe5/"

#     if not os.path.exists(path_output_cppe5):
#         os.makedirs(path_output_cppe5)

#     path_anno = os.path.join(path_output_cppe5, "cppe5_ann.json")
#     categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
#     output_json["images"] = []
#     output_json["annotations"] = []
#     for example in cppe5:
#         ann = val_formatted_anns(example["image_id"], example["objects"])
#         output_json["images"].append(
#             {
#                 "id": example["image_id"],
#                 "width": example["image"].width,
#                 "height": example["image"].height,
#                 "file_name": f"{example['image_id']}.png",
#             }
#         )
#         output_json["annotations"].extend(ann)
#     output_json["categories"] = categories_json

#     with open(path_anno, "w") as file:
#         json.dump(output_json, file, ensure_ascii=False, indent=4)

#     for im, img_id in zip(cppe5["image"], cppe5["image_id"]):
#         path_img = os.path.join(path_output_cppe5, f"{img_id}.png")
#         im.save(path_img)

#     return path_output_cppe5, path_anno

# """Next, prepare an instance of a `CocoDetection` class that can be used with `cocoevaluator`."""

# import torchvision


# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self, img_folder, feature_extractor, ann_file):
#         super().__init__(img_folder, ann_file)
#         self.feature_extractor = feature_extractor

#     def __getitem__(self, idx):
#         # read in PIL image and target in COCO format
#         img, target = super(CocoDetection, self).__getitem__(idx)

#         # preprocess image and target: converting target to DETR format,
#         # resizing + normalization of both image and target)
#         image_id = self.ids[idx]
#         target = {"image_id": image_id, "annotations": target}
#         encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
#         pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
#         target = encoding["labels"][0]  # remove batch dimension

#         return {"pixel_values": pixel_values, "labels": target}


# im_processor = AutoImageProcessor.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")

# path_output_cppe5, path_anno = save_cppe5_annotation_file_images(cppe5["test"])
# test_ds_coco_format = CocoDetection(path_output_cppe5, im_processor, path_anno)

# """Finally, load the metrics and run the evaluation."""

# import evaluate
# from tqdm import tqdm

# model = AutoModelForObjectDetection.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")
# module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
# val_dataloader = torch.utils.data.DataLoader(
#     test_ds_coco_format, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn
# )

# with torch.no_grad():
#     for idx, batch in enumerate(tqdm(val_dataloader)):
#         pixel_values = batch["pixel_values"]
#         pixel_mask = batch["pixel_mask"]

#         labels = [
#             {k: v for k, v in t.items()} for t in batch["labels"]
#         ]  # these are in DETR format, resized + normalized

#         # forward pass
#         outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

#         orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
#         results = im_processor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api

#         module.add(prediction=results, reference=labels)
#         del batch

# results = module.compute()
# print(results)

# """These results can be further improved by adjusting the hyperparameters in [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments). Give it a go!

# ## Inference

# Now that you have finetuned a DETR model, evaluated it, and uploaded it to the Hugging Face Hub, you can use it for inference.
# The simplest way to try out your finetuned model for inference is to use it in a [Pipeline](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline). Instantiate a pipeline
# for object detection with your model, and pass an image to it:
# """

# from transformers import pipeline
# import requests

# url = "https://i.imgur.com/2lnWoly.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# obj_detector = pipeline("object-detection", model="MariaK/detr-resnet-50_finetuned_cppe5")
# obj_detector(image)

# """You can also manually replicate the results of the pipeline if you'd like:"""

# image_processor = AutoImageProcessor.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")
# model = AutoModelForObjectDetection.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")

# with torch.no_grad():
#     inputs = image_processor(images=image, return_tensors="pt")
#     outputs = model(**inputs)
#     target_sizes = torch.tensor([image.size[::-1]])
#     results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     print(
#         f"Detected {model.config.id2label[label.item()]} with confidence "
#         f"{round(score.item(), 3)} at location {box}"
#     )

# """Let's plot the result:"""

# draw = ImageDraw.Draw(image)

# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     x, y, x2, y2 = tuple(box)
#     draw.rectangle((x, y, x2, y2), outline="red", width=1)
#     draw.text((x, y), model.config.id2label[label.item()], fill="white")

# image