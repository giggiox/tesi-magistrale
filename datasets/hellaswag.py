from datasets.dataset import Dataset
import re

class HellaSwag(Dataset):
    """
link: https://huggingface.co/datasets/Rowan/hellaswag
Example(cropped)
{
    "activity_label": "Removing ice from car",
    "ctx": "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then",
    "ctx_a": "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles.",
    "ctx_b": "then",
    "endings": "[\", the man adds wax to the windshield and cuts it.\", \", a person board a ski lift, while two men supporting the head of the per...",
    "ind": 4,
    "label": "3",
    "source_id": "activitynet~v_-1IBHYS3L-Y",
    "split": "train",
    "split_type": "indomain"
}

Note: 
1. The ctx and the endings may contain tags like [header], [title], [step], [substeps], etc. If we don't remove them, the LLM might mis-interpret the prompt.
2. label is from 0-3. We will pose the question to LLM as choose between 4 options indexed 1-4.

    """  
    
    def __init__(self, load_on_init = False, dataset_fraction = 1, split = "validation"):
        super().__init__(dataset_fraction, split)
        self.dataset_name = "hellaswag"
        if load_on_init:
            self.get_dataset()
        
    
    def get_dataset(self):
        return super().get_dataset(self.dataset_name)

    def get_dataset_name(self):
        return self.dataset_name

    def format_prompt(self,example):
        """
        The example is formatted as
        
        Context: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then
        Which of the following options is the most plausible continuation?
        1. The man adds wax to the windshield and cuts it.  
        2. A person boards a ski lift, while two men support the head of the person.  
        3. The woman walks away and the man starts removing the ice from the car.  
        4. The man and woman start dancing on the snowy ground. 
        Respond with only the number of the most plausible option: 
        """
        ctx = example['ctx']
        ctx = re.sub(r"\[.*?\]", "", ctx).strip() # Remove tags from context
        endings = example['endings']
        for i in range(len(endings)):
            endings[i] = re.sub(r"\[.*?\]", "", endings[i]).strip() # Remove tags from endings
        return f"Context: {ctx}\nWhich of the following options is the most plausible continuation?\n1. {endings[0]}\n2. {endings[1]}\n3. {endings[2]}\n4. {endings[3]}\nRespond with only the number of the most plausible option:"

    def clean_answer(self, answer, prompt):
        return answer.split("\n")[0] # Take only first row of response, the other rows are usually the explaining

    def get_true_answer(self, example):
        return str(int(example["label"]) + 1)

    def is_correct(self, model_answer, row): 
        """
        true_answer is already 1-indexed by get_true_answer. 
        true_answer \in [1,2,3,4]
        """
        true_answer = self.get_true_answer(row)
        return str(true_answer) in model_answer