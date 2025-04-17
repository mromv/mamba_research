import logging
import torch

from hydra.utils import instantiate
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

from src.model import MambaForSequenceClassification

log = logging.getLogger(__name__)


def setup_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    n_additional_tokens: int,
    peft_config: dict
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MambaForSequenceClassification(
        model_name=model_name,
        num_labels=num_labels,
        n_additional_tokens=n_additional_tokens,
    ).to(device)
    
    if peft_config.get('freeze_model'):
        model.freeze()
        
    elif peft_config.get('lora'):
        config = instantiate(peft_config['lora'])
        model = get_peft_model(model, config)
        log.info(model.print_trainable_parameters())
    
    model.mamba.gradient_checkpointing_enable()
    return model, tokenizer
