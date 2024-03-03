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

#RUNNING API
python -m hwars.api.flask_app
#Calling API
curl -F "file=@./data/buttons/ss-1-find.png" http://127.0.0.1:5000/buttons

#sample run
python -m hwars.jobs.play '
{
  "repeat": 30,
  "commands":[
    {"button":"Find", "index":3},
    {"button":"Start"},
    {"button":"To battle"},
    {"button":"Return to the City", "timeout": 300}
  ]
}'

