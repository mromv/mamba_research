import torch
from typing import List
from tqdm import tqdm
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

def generate_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    prompt_template: str,
    batch_size=16,
    max_length=8192,
) -> List[int]:
    prompts = [prompt_template.format(sentence=example['sentence']) for example in dataset]
    
    answers = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating answers"):
        batch_prompts = prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True, 
            max_length=max_length, 
            padding_side='left'
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
            )
        
        batch_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for prompt, full_answer in zip(batch_prompts, batch_answers):
            answer_part = full_answer[len(prompt):].strip().lower()
            
            answer_word = answer_part.split()[0] if answer_part else ""
            if answer_word.startswith("yes"):
                answers.append(1)
            elif answer_word.startswith("no"):
                answers.append(0)
            else:
                answers.append(-1)
    
    return answers
    