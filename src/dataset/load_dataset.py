from typing import List
from .tokenize_function import get_tokenize_function
from datasets import load_dataset
from transformers import PreTrainedTokenizer

def load_tokenized_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    text_template: str,
    choices: List[str],
    learnable_tokens: bool,
):
    dataset = load_dataset('glue', dataset_name)
    columns_to_remove = [
        col for col in dataset['train'].column_names
    ]
    
    tokenize_func = get_tokenize_function(tokenizer, text_template, choices, learnable_tokens)
    tokenized_dataset = dataset.map(tokenize_func)
    
    return tokenized_dataset, tokenized_dataset.remove_columns(columns_to_remove).remove_columns(['prompt'])
    