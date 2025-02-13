import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.devices = ["cuda:0"]
    config.stop_on_success = False
    return config