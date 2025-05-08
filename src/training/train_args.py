from transformers import TrainingArguments

def setup_training(output_dir: str, training_config: dict):
    return TrainingArguments(
        report_to='none',#'wandb',
        output_dir=output_dir,
        run_name=output_dir,
        num_train_epochs=training_config.get('epochs'),
        per_device_train_batch_size=training_config.get('batch_size'),
        per_device_eval_batch_size=training_config.get('batch_size'),
        learning_rate=training_config.get('lr'),
        weight_decay=training_config.get('weight_decay'),
        warmup_steps=training_config.get('warmup_steps'),
        optim=training_config.get('optim'),
        lr_scheduler_type=training_config.get('lr_scheduler_type'),
        logging_dir='./logs',
        eval_strategy='epoch',
        save_strategy='epoch',
        label_names=['labels'],
        save_safetensors=False,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        greater_is_better=False,
    )
