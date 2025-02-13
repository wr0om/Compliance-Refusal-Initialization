import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()


    config.transfer = True
    config.logfile = ""

    config.progressive_goals = False
    config.stop_on_success = False
    config.tokenizer_kwargs = [{"use_fast": False}]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False}
        ]
    config.devices = ["cuda:0"]

    # TODO: comment this if not wanted eval set
    config.add_eval_set = True
    config.n_test_data = 25 # this is the eval set size!! (if add_eval_set is True)
    config.test_steps = 0
    config.eval_offset = 200 # so we don't evaluate the same data as the training/testing data (we test on maximum of until 125)


    return config
