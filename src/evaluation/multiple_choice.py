import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer


def calculate_unnormalized_score(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, continuation: str):
    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")
    continuation_tokens = tokenizer.encode(continuation, return_tensors="pt")
    
    input_ids = torch.cat([prompt_tokens, continuation_tokens], dim=-1).to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    unnormalized_score = 0.0
    for i in range(prompt_tokens.size(1), input_ids.size(1)):
        token_id = input_ids[0, i]
        unnormalized_score += log_probs[0, i-1, token_id].item()
    
    return unnormalized_score

def evaluate_multiple_choice(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, options: list[str], ):
    result = {}
    for option in options:
        temp_score = calculate_unnormalized_score(model, tokenizer, prompt, option)
        result[option] = temp_score
    return result, max(result, key=result.get)


def evaluate_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer, 
    dataset: Dataset,
    options: list[str]
):
    predictions = []
    for example in tqdm(dataset):
        _, predicted_answer = evaluate_multiple_choice(model, tokenizer, example['prompt'], options)
        out = options.index(predicted_answer)
        predictions.append(out)
    return predictions
