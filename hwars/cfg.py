import yaml
import os
import logging

class Cfg:
    HF_TOKEN = None
    DEBUG_MODE = False

config_loaded = False
_cfg = "data/config.yaml"
def load_configs():
    global config_loaded, _cfg
    if not config_loaded:
        config_loaded = True
        if os.path.exists(_cfg):
            print(f"Loading configs from {_cfg}")
            with open(_cfg, 'r') as stream:
                configs = yaml.safe_load(stream)
            for k in configs:
                setattr(Cfg, k, configs[k])

load_configs()