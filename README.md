# Yolo training

## venv

/cygdrive/c/Python/Python310/python.exe -m venv ./venv
(set ff=unix with vim for activate)
. venv/Scripts/activate
pip install -r requirements.txt

## train

### Optional
Clone this (for trial code): 
https://github.com/binitt/yolov11-try

### Main train
// setup datasets folder
vim "C:\Users\binit\AppData\Roaming\Ultralytics\settings.json"
```
  "datasets_dir": "C:\\git\\hwars\\data\\yolo",
```

mkdir -p logs
python -m hwars.buttons.yolo.create_model


## docker linux setup
dc up -d
start lxterminal
follow instructions from docker/readme.txt

## config for running
### hwars host
./pbuild.sh
// check binit-hwars ip addr, this is in utils.py that makes api call
./scripts/api.sh

### hwars container
pip install -r dist/requirements-client.txt 
./dist/pinstall.sh
./dist/run.sh


# DETR repo
#https://github.com/facebookresearch/detr
#https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb#scrollTo=PcxWAOzOYTEn

# finetune DETR
https://colab.research.google.com/github/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb#scrollTo=hMMXcsU8MCIa

# https://www.kaggle.com/code/shnakazawa/object-detection-with-pytorch-and-detr

# HF
https://huggingface.co/docs/transformers/tasks/object_detection

#Albumentations
https://albumentations.ai/docs/examples/pytorch_classification/

#Training/Tuning steps
## PREP Dataset
# create image png in data/buttons
# Open via and annotate from browser: 
## file:///C:/installed/via-2.0.12/via-2.0.12/via.html
## load data/buttons/hwars.json
## load all image files
## annotate and save project
## copy hwars.json back to data/buttons 
python -m hwars.dataset.converter

#Versioning 
git clone "https://huggingface.co/binitt/hwars-buttons-model.git"
git tag v1.0
git push origin v1.0

## Training in colab
hwars\hwars\buttons\hwars_buttons.ipynb

#RUNNING API
python -m hwars.api.flask_app
#Calling API
curl -F "file=@./data/buttons/ss-1-find.png" http://127.0.0.1:5000/buttons

#campaign
python -m hwars.jobs.play '
{
  "repeat": 300,
  "commands":[
    {"button":"Find", "index":-1},
    {"button":"Start"},
    {"button":"To battle"},
    {"button":"Return to", "timeout": 300}
  ]
}'

#arena
python -m hwars.jobs.play '
{
  "repeat": 5,
  "commands":[
    {"button":"Attack", "index":0},
    {"button":"To battle", "index":0, "sleep":60},
    {"button":"OK", "index":0}
  ]
}'


#grand arena
python -m hwars.jobs.play '
{
  "repeat": 5,
  "commands":[
{"button":"To battle", "index":0},
{"button":"Attack", "index":0},
{"button":"Next", "index":0},
{"button":"Next", "index":0},
{"button":"To battle", "index":0, "sleep":60 },
{"button":"Next battle", "index":0, "sleep":60},
{"button":"Next battle", "index":0, "optional": true, "sleep":60},
    {"button":"OK", "index":0}
  ]
}'



#test with api
python -m hwars.buttons.test_model_api
