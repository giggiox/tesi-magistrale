from datasets.dataset import Dataset

class SquadV2(Dataset):
    """
link: https://huggingface.co/datasets/rajpurkar/squad_v2
Example(cropped)
{
    "answers": {
        "answer_start": [94, 87, 94, 94],
        "text": ["10th and 11th centuries", "in the 10th and 11th centuries", "10th and 11th centuries", "10th and 11th centuries"]
    },
    "context": "\"The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave thei...",
    "question": "When were the Normans in Normandy?",
    "title": "Normans"
}

Note: there can be no correct answer, i.e 
Example
{
    "context": "\"The "West Side" of Fresno, also often called "Southwest Fresno", is one of the oldest neighborhoods in the city. The neighborhood lies southwest of the 99 freeway (which divides it from Downtown Fresno), west of the 41 freeway and south of Nielsen Ave (or the newly constructed 180 Freeway), and extends to the city limits to the west and south. The neighborhood is traditionally considered to be the center of Fresno's African-American community. It is culturally diverse and also includes significant Mexican-American and Asian-American (principally Hmong or Laotian) populations.
    "question": "What is significant about the age of Downtown Fresno?"
    "answers": {
        "answer_start": []
        "text": []
    }
}
    """
    
    def __init__(self, load_on_init = False, dataset_fraction = 1, split = "validation"):
        super().__init__(dataset_fraction, split)
        self.dataset_name = "rajpurkar/squad_v2"
        if load_on_init:
            self.get_dataset()
    
    def get_dataset(self):
        return super().get_dataset(self.dataset_name)

    def get_dataset_name(self):
        return self.dataset_name

    def format_prompt(self, example):
        return f"Context: {example['context']} Question: {example['question']} Answer:"

    def clean_answer(self, answer, prompt):
        return answer.split("\n")[0] # Take only first row of response, the other rows are usually the explaining

    def get_true_answer(self, example):
        text = example["answers"]["text"]
        if len(text) == 0:
            return "No answer"
        return text[0]

    def is_correct(self, model_answer, row):
        # model_answer = re.sub(r'[^\w\s]','', model_answer)
        true_answer = row["answers"]["text"]    
        llm_no_answer = "No answer" in model_answer or model_answer == ""
        no_answer = len(true_answer) == 0
        return (no_answer and llm_no_answer) or (model_answer in true_answer)