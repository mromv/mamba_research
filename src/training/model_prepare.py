import logging
import torch

from hydra.utils import instantiate
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

log = logging.getLogger(__name__)


def setup_model_and_tokenizer(
    model_name: str,
    peft_config: dict
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    if peft_config.get('freeze_model'):
        model.freeze()
        
    elif peft_config.get('lora'):
        config = instantiate(peft_config['lora'])
        model = get_peft_model(model, config)
        log.info(model.print_trainable_parameters())
    
    return model, tokenizer
