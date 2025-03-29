from models.model import Model

from datasets.hellaswag import HellaSwag
from datasets.boolq import BoolQ

from utils.logger_manager import LoggerManager
from utils.log_to_file import LogToFile

from huggingface_hub.hf_api import HfFolder
#HfFolder.save_token("HF-TOKEN") # If using protected models

# Example of how to run an evaluation
logger_manager = LoggerManager("log.txt")

model = Model("TinyLlama/TinyLlama-1.1B-Chat-v0.6", load_on_init=True)

dataset = BoolQ(load_on_init = True)

accuracy = dataset.evaluate_model(model, n_shot=2, logger_manager = logger_manager)

logger_manager.write(f"Accuracy: {accuracy}")
