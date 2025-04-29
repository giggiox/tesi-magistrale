import random
from datasets import load_dataset
from tqdm import tqdm
from utils.log_to_file import LogToFile

class Dataset():
    def __init__(self, dataset_fraction = 1, split="validation"):
        self.dataset = None
        self.dataset_fraction = dataset_fraction
        self.split = split
     

    def get_dataset(self, dataset_name):
        if self.dataset:
            return self.dataset
        if self.split:
            self.dataset = load_dataset(dataset_name, split=self.split)
        else:
            self.dataset = load_dataset(dataset_name)
        if self.dataset_fraction != None:
            num_samples = int(len(self.dataset) * self.dataset_fraction)
            self.dataset = self.dataset.shuffle().select(range(num_samples))
        return self.dataset


    def iteration_evaluate_model(self, model, row_idx, row, n_shot, use_cot, logger_manager = None):
        dataset = self.get_dataset()

        # Code for n-shot prompting
        dataset_keys = list(range(len(dataset)))
        dataset_keys_filtered = dataset_keys[:row_idx] + dataset_keys[row_idx + 1:]
        dataset_filtered = dataset.select(dataset_keys_filtered)
        prompt = ""
        for i in range(n_shot):
            shot_row = random.choice(dataset_filtered)
            prompt += self.format_prompt(shot_row, use_cot) + " " + str(self.get_true_answer(shot_row)) + "\n"

        # Building the prompt
        prompt = prompt + self.format_prompt(row, use_cot)

        # Ask model the prompt
        answer = model.get_pipeline()(prompt, return_full_text=False)[0]['generated_text']  


        is_llm_answer_correct, is_answer_rejected = self.is_correct(answer, row)

        if logger_manager:
            logger_manager.write(f"- Prompt:\n {prompt}\n")
            logger_manager.write(f"- Answer:\n {answer}\n")
            logger_manager.write(f"- True Answer: {self.get_true_answer(row)}\n")
            logger_manager.write(f"- Is LLM answer correct? : {is_llm_answer_correct}\n")

        return is_llm_answer_correct, is_answer_rejected
        
    
    def evaluate_model(self, model, n_shot = 0, use_cot = False, logger_manager = None):
        correct = 0
        rejected = 0
        pipe = model.get_pipeline()
        dataset = self.get_dataset()
        
        
        with LogToFile(logger_manager):
            logger_manager.write(f"Evaluating {model.get_model_name()} on {self.get_dataset_name()} with {n_shot}-shot\n")
            
        for idx, example in tqdm(enumerate(dataset),total=len(dataset), desc=f"Evaluating {model.get_model_name()} on {self.get_dataset_name()} with {n_shot}-shot"):
            with LogToFile(logger_manager):
                is_correct, is_answer_rejected = self.iteration_evaluate_model(model, idx, example, n_shot, use_cot, logger_manager)
                if is_correct:
                    correct += 1
                if is_answer_rejected:
                    rejected += 1
                    
        accuracy = correct / (len(dataset)) * 100
        with LogToFile(logger_manager):
            logger_manager.write(f"\nFinal accuracy: {accuracy}")
            logger_manager.write(f"\nNumber of rejected answers: {rejected}")
        return accuracy