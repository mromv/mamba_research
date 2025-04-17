from transformers import PreTrainedTokenizer

def get_tokenize_function(
    tokenizer: PreTrainedTokenizer,
    template: str,
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
        example['text'] = template.format(**example, **special_tokens)
        tokenized_example = tokenizer(example['text'])
        tokenized_example['eos_pos'] = len(tokenized_example['input_ids']) - 1
        return tokenized_example

    return tokenize_func