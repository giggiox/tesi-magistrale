# models/tiny_llama_small.py
from .model import Model

class TinyLLamaSmall(Model):
    def __init__(self, load_on_init=False):
        super().__init__("TinyLlama/TinyLlama-1.1B-Chat-v0.6", load_on_init)

    def clean_answer(self, answer, prompt):
        return answer