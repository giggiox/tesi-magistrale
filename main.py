from models.phi2 import Phi2
from models.tiny_llama_small import TinyLLamaSmall

from datasets.squad_v2 import SquadV2
from datasets.hellaswag import HellaSwag
from datasets.boolq import BoolQ

from utils.logger_manager import LoggerManager
from utils.log_to_file import LogToFile

# Example of how to run an evaluation
logger_manager = LoggerManager("log.txt")
model = Phi2(load_on_init=True)
dataset = SquadV2(load_on_init=True)

accuracy = dataset.evaluate_model(model, n_shot=2, logger_manager=logger_manager)
logger_manager.write(f"Accuracy: {accuracy}")
