import os
import json
import logging
from datetime import datetime


def load_json(file_name: str):
    if isinstance(file_name, str) and file_name.endswith("json"):
        with open(file_name, 'r') as file:
            data = json.load(file)
    else:
        raise ValueError("The file path you passed in is not a json file path.")
    
    return data

def init_logger(outputs_dir):
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    os.makedirs(os.path.join(outputs_dir, "logs"), exist_ok=True)
    log_path = os.path.join(outputs_dir, "logs", "{}.txt".format(current_time))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s || %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    