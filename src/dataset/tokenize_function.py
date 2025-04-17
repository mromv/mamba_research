from typing import List
from transformers import PreTrainedTokenizer

def get_tokenize_function(
    tokenizer: PreTrainedTokenizer,
    template: str,
    choices: List[str],
    learnable_tokens: bool = True
):
    '''
    To insert the text and specidl tokens (if True) into the template, and then tokenize the text.
    Return the tokenization function.
    '''
    special_tokens = {'cls_token': '', 'sep_token': ''}
    if learnable_tokens:
        special_tokens['cls_token'] = '<|CLS|>'
        special_tokens['sep_token'] = '<|SEP|>'
        tokenizer.add_special_tokens(special_tokens)

    def tokenize_func(example):
        prompt = template.format(**example, **special_tokens)
        answer = choices[example['label']]
        label = f"{answer}{tokenizer.eos_token}"

        inputs_ids_prompt = tokenizer(prompt)['input_ids']
        input_ids_label = tokenizer(label)['input_ids']

        example['input_ids'] = inputs_ids_prompt + input_ids_label    
        example['labels'] = [-100] * (len(inputs_ids_prompt) - 1) + input_ids_label + [-100]  # [-100, ... -100, label_tok, eos_tok]
        
        return example

    return tokenize_func
