import os
import wandb
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from typing import Optional
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./conf/', config_name='config')
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)

    model_name = cfg['model_name'].split('/')[1]
    dataset_name = cfg['dataset']['dataset_name']

    output_dir = f"./outputs/{model_name}_{dataset_name}"
    model_dir = f"./models/{model_name}_{dataset_name}"
    
    # model
    model = AutoModelForCausalLM.from_pretrained(cfg['model_name'], use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

    new_tokens = ["<|task|>", "<|/task|>", "<|think|>", "<|/think|>"]
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))
    model.resize_token_embeddings(len(tokenizer))

    # dataset
    dataset = load_dataset('mromv/glue_reasoning', dataset_name)
    dataset = dataset.filter(lambda example: 
        (example['answer'] == 'yes' and example['label'] == 1) or 
        (example['answer'] == 'no' and example['label'] == 0)
    )

    def process_func(example):
        prompt = cfg['dataset']['doc_to_text'].format(**example)
        prompt = f"<|task|>{prompt}<|/task|>\n<|think|>{example['reasoning']}<|/think|>\n{example['answer']}{tokenizer.eos_token}"
        return {'text': prompt}

    dataset = dataset.map(process_func)

    # init wandb
    run = wandb.init(
        name=output_dir,
        project=cfg['wandb']['project']
    )
    
    training_args = SFTConfig(
        report_to="wandb",
        output_dir=output_dir,
        num_train_epochs=cfg['training']['epochs'],
        per_device_train_batch_size=cfg['training']['batch_size'],
        logging_dir='./logs',
        lr_scheduler_type=cfg['training']['lr_scheduler_type'],
        logging_steps=cfg['training']['logging_steps'],
        learning_rate=cfg['training']['lr'],
        eval_strategy=cfg['training']['eval_strategy'],
        weight_decay=cfg['training']['weight_decay'],
        max_grad_norm=cfg['training']['max_grad_norm'],
        warmup_steps=cfg['training']['warmup_steps'],
        optim=cfg['training']['optim'],
        packing=False,
    )

    model.train()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation']
    )
    trainer.train()
    
    trainer.save_model(model_dir)
    wandb.finish()

if __name__ == '__main__':
    main()

