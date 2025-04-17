from .tokenize_function import get_tokenize_function
from datasets import load_dataset
from transformers import PreTrainedTokenizer

def load_tokenized_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    text_template: str,
    learnable_tokens: bool,
):
    dataset = load_dataset('glue', dataset_name)
    columns_to_remove = [
        col for col in dataset['train'].column_names
        if col not in ("label")
    ]
    
    tokenize_func = get_tokenize_function(tokenizer, text_template, learnable_tokens)
    
    return dataset.map(tokenize_func)\
                  .remove_columns(columns_to_remove)\
                  .rename_column("label", "labels")
    