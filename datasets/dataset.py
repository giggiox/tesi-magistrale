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


    def iteration_evaluate_model(self, model, row_idx, row, n_shot, logger_manager = None):
        dataset = self.get_dataset()


        # Code for n_shot prompting
        dataset_keys = list(range(len(dataset)))
        dataset_keys_filtered = dataset_keys[:row_idx] + dataset_keys[row_idx + 1:]
        dataset_filtered = dataset.select(dataset_keys_filtered)
        prompt = ""
        for i in range(n_shot):
            shot_row = random.choice(dataset_filtered)
            prompt += self.format_prompt(shot_row) + " " + str(self.get_true_answer(shot_row)) + "\n"
        
        prompt = prompt + self.format_prompt(row)
        # prompt = model.format_prompt(prompt)

        # Ask model the prompt
        answer = model.get_pipeline()(prompt, return_full_text=False)[0]['generated_text']  

        # Clean answer from the model
        cleaned_answer = model.clean_answer(answer, prompt) # Each model has a unique way to reply
        cleaned_answer = self.clean_answer(cleaned_answer, prompt) # Each dataset has a unique way to clean the answer
        true_answer = self.get_true_answer(row) # For logging purposes


        is_llm_answer_correct = self.is_correct(cleaned_answer, row)

        if logger_manager:
            logger_manager.write(f"- Prompt:\n {prompt}\n")
            logger_manager.write(f"- Answer:\n {answer}\n")
            logger_manager.write(f"- Cleaned Answer:\n {cleaned_answer}\n")
            logger_manager.write(f"- True Answer: {true_answer}\n")
            logger_manager.write(f"- Is LLM answer correct? : {is_llm_answer_correct}\n")

        return is_llm_answer_correct
        
    
    def evaluate_model(self, model, n_shot = 0, logger_manager = None):
        with LogToFile(logger_manager):
            logger_manager.write(f"Evaluating {model.get_model_name()} on {self.get_dataset_name()} with {n_shot}-shot\n")

        dataset = self.get_dataset()
        correct = 0    
        for idx, example in tqdm(enumerate(dataset),total=len(dataset), desc=f"Evaluating {model.get_model_name()} on {self.get_dataset_name()}"):
            with LogToFile(logger_manager):
                is_correct = self.iteration_evaluate_model(model, idx, example, n_shot, logger_manager)
                if is_correct:
                    correct += 1
        accuracy = correct / (len(dataset)) * 100

        with LogToFile(logger_manager):
            logger_manager.write(f"\nFinal accuracy: {accuracy}")

        return accuracy