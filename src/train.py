import os
import wandb
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from typing import Optional
from trl import SFTTrainer
from transformers import DataCollatorForSeq2Seq

from src.util.dirname import get_output_dir_name
from src.dataset import get_tokenize_function, load_tokenized_dataset
from src.training import setup_model_and_tokenizer, get_metrics, setup_training

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./conf/', config_name='config')
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)

    output_name = get_output_dir_name(
        cfg['output_dir_template'],
        dataset_name=cfg['dataset']['dataset_name'],
        peft=cfg['peft'],
        epochs=cfg['training']['epochs'],
        lr=cfg['training']['lr']
    )
    save_dir = os.path.join(cfg['save_dir'], output_name)

    # model
    model, tokenizer = setup_model_and_tokenizer(
        model_name=cfg['model_name'],
        peft_config=cfg['peft']
    )

    # dataset
    dataset = load_tokenized_dataset(
        dataset_name=cfg['dataset']['dataset_name'],
        tokenizer=tokenizer,
        text_template=cfg['dataset']['doc_to_text'],
        choices=cfg['dataset']['doc_to_choice'],
        learnable_tokens=cfg['peft'].get('learnable_tokens')
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        label_pad_token_id=-100
    )

    # Load the metric
    compute_metrics = get_metrics(
        dataset_name=cfg['dataset']['dataset_name']
    )

    # init wandb
    run = wandb.init(
        name=output_name,
        project=cfg['wandb']['project']
    )

    training_args = setup_training(save_dir, cfg['training'])
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
    )
    
    trainer.train()
    wandb.finish()

if __name__ == '__main__':
    main()

