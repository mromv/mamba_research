from evaluate import load

def get_metrics(dataset_name: str):
    metric = load('glue', dataset_name)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    return compute_metrics
