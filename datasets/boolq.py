from datasets.dataset import Dataset
import re

class BoolQ(Dataset):
    """
link: https://huggingface.co/datasets/google/boolq
Example (cropped):
{
    "passage": "\"All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned...",
    "question": "does ethanol take more energy make that produces"
    "answer": false,
}
    
    """
    def __init__(self, load_on_init = False, dataset_fraction = 1, split = "validation"):
        super().__init__(dataset_fraction, split)
        self.dataset_name = "google/boolq"
        if load_on_init:
            self.get_dataset()
    
    def get_dataset(self):
        return super().get_dataset(self.dataset_name)

    def get_dataset_name(self):
        return self.dataset_name

    def format_prompt(self, example, use_cot):
        """
        Formats the example in this way:
        
        Answer the following true/false question. The last line of your response should be in the following format: 'Answer: true/false' (e.g. 'Answer: true').
        Passage: All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned...
        Question: does ethanol take more energy make that produces
        """
        ret = f"Answer the following true/false question. The last line of your response should be in the following format: 'Answer: true/false' (e.g. 'Answer: true').\nPassage: {example['passage']}\nQuestion: {example['question']}\n"
        if use_cot:
            ret += "Let's think step by step"
        return ret


    def get_true_answer(self, example):
        return f"Answer: {example['answer']}"

    def is_correct(self, model_answer, row):
        true_answer = self.get_true_answer(row)
        prediction = re.search("(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*(true|false)(?![a-zA-Z0-9])", model_answer)
        if prediction:
            true_false = re.search("(True|False)", prediction.group(0).lower(), re.IGNORECASE).group(0).lower()
            true_true_false = re.search("(True|False)", true_answer, re.IGNORECASE).group(0).lower()
            return true_false == true_true_false, False
        return False, True