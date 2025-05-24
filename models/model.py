from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class Model:
    """
    examples of models:
    microsoft/phi-2
    TinyLlama/TinyLlama-1.1B-Chat-v0.6
    google/gemma-2-2b-it
    """
    
    def __init__(self, model_name, load_on_init = False):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipe = None
        if load_on_init:
            self.get_model()
            self.get_tokenizer()

    def get_model_name(self):
        return self.model_name

    def format_prompt(self, prompt):
        # By default, keep prompt unchanged, some subclasses may have to override this behaviour
        # for example deepseek may have to append the <think> tag at the end of the prompt
        return prompt
        

    def get_model(self):
        if not self.model:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        return self.model
        
    def get_tokenizer(self):
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        return self.tokenizer

    def get_pipeline(self):
        if not self.pipe:
            self.pipe = pipeline(
                "text-generation",
                model=self.get_model(),
                tokenizer=self.get_tokenizer(),
                max_new_tokens=256,
                temperature=0
            )
        return self.pipe
