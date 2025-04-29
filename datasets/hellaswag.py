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

    ANSWER = {
        0:'A',
        1:'B',
        2:'C',
        3:'D'
    }
    
    def __init__(self, load_on_init = False, dataset_fraction = 1, split = "validation"):
        super().__init__(dataset_fraction, split)
        self.dataset_name = "hellaswag"
        if load_on_init:
            self.get_dataset()
        
    
    def get_dataset(self):
        return super().get_dataset(self.dataset_name)

    def get_dataset_name(self):
        return self.dataset_name

    def format_prompt(self, example, use_cot):
        """
        The example is formatted as

        Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A')
        Context: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then
        Which of the following options is the most plausible continuation?
        A. The man adds wax to the windshield and cuts it.  
        B. A person boards a ski lift, while two men support the head of the person.  
        C. The woman walks away and the man starts removing the ice from the car.  
        D. The man and woman start dancing on the snowy ground. 
        """
        ctx = example['ctx']
        ctx = re.sub(r"\[.*?\]", "", ctx).strip() # Remove tags from context
        endings = example['endings']
        for i in range(len(endings)):
            endings[i] = re.sub(r"\[.*?\]", "", endings[i]).strip() # Remove tags from endings
        ret = f"Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').\n Context: {ctx}\nWhich of the following options is the most plausible continuation?\nA. {endings[0]}\nB. {endings[1]}\nC. {endings[2]}\nD. {endings[3]}\n"
        if use_cot:
            ret += "Let's think step by step"
        return ret

    def get_true_answer(self, example):
        return f"Answer: {self.ANSWER[int(example['label'])]}"

    def is_correct(self, model_answer, row): 
        true_answer = self.ANSWER[int(row['label'])].lower()
        prediction = re.search("(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])", model_answer)
        
        if prediction:
            predicition_label = re.search(r"Answer:(.*)", prediction.group(0).lower(), re.IGNORECASE).group(1).lower().strip()
            return predicition_label == true_answer, False
        return False, True