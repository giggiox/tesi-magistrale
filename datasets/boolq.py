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

    def format_prompt(self, example):
        """
        Formats the example in this way:
        
        Passage: All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned...
        Question: does ethanol take more energy make that produces
        Answer with only True or False:
        """
        return f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer with only True or False:"

    def clean_answer(self, answer, prompt):
        return answer.split("\n")[0] # Take only first row of response, the other rows are usually the explaining

    def get_true_answer(self, example):
        return example["answer"]

    def is_correct(self, model_answer, row):
        """
        looks for a True of False (ignoring case sensitivity) in the model answer string.
        """
        true_answer = self.get_true_answer(row)
        prediction = re.search(r"(True|False)", model_answer, re.IGNORECASE)
        if prediction: # If there is a true or a false
            return str(prediction.group(0).lower()) == str(true_answer).lower() # https://stackoverflow.com/questions/15340582/python-extract-pattern-matches
        return False # In every other case, it is not the correct answer :(