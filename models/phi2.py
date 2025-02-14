from models.model import Model

class Phi2(Model):
    def __init__(self, load_on_init=False):
        super().__init__("microsoft/phi-2", load_on_init)

    def clean_answer(self, answer, prompt):
        return answer