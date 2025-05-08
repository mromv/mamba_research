import os
import wandb
import logging

import hydra
from typing import Optional
from omegaconf import DictConfig, OmegaConf

from evaluate import load 
from trl import SFTTrainer
from transformers import DataCollatorForSeq2Seq
from sklearn.metrics import classification_report

from src.util.dirname import get_output_dir_name
from src.dataset import get_tokenize_function, load_tokenized_dataset
from src.training import setup_model_and_tokenizer, setup_training
from src.evaluation.multiple_choice import evaluate_dataset
from src.evaluation.metrics import get_metrics

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
    checkpoints = os.path.join(cfg['checkpoints'], output_name)

    # model
    model, tokenizer = setup_model_and_tokenizer(
        model_name=cfg['model_name'],
        peft_config=cfg['peft']
    )

    # dataset
    dataset, tok_dataset = load_tokenized_dataset(
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

    # init wandb
    run = wandb.init(
        name=output_name,
        project=cfg['wandb']['project']
    )

    training_args = setup_training(checkpoints, cfg['training'])

    model.train()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=tok_dataset['train'],
        eval_dataset=tok_dataset['validation'],
        data_collator=data_collator,
    )
    
    trainer.train()
    # trainer.save_model(os.path.join(cfg['save_dir'], output_name))

    # evaluation
    model.eval()
    
    eval_dataset = dataset['validation']
    metric = load(cfg['dataset']['tag'], cfg['dataset']['dataset_name'])
    
    preds = evaluate_dataset(
        trainer.model,
        trainer.tokenizer,
        eval_dataset,
        cfg['dataset']['doc_to_choice'],
    )
    
    report = classification_report(eval_dataset['label'], preds)
    eval_metric = metric.compute(predictions=preds, references=eval_dataset['label'])
    
    log.info(cfg)
    log.info(report)
    log.info(eval_metric)
    
    wandb.finish()

if __name__ == '__main__':
    main()

