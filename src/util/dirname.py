def peft_name(
    peft: dict
) -> str:
    method_parts = []
    if peft.get('lora'):
        method_parts.append('lora')

    else:
        if peft.get('learnable_tokens'):
            method_parts.append('tokens')
    
        if not peft.get('freeze_model'):
            method_parts.append('full_ft')
        
    return '_'.join(method_parts)
    
def get_output_dir_name(
    template: str,
    **kwargs
) -> str:
    peft_config = kwargs.get('peft')
    if peft_config:
        kwargs['peft'] = peft_name(peft_config)
        
    return template.format(**kwargs)
