import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer


class MambaForSequenceClassification(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        n_additional_tokens: int = 0
    ) -> None:
        super().__init__()
        self.mamba = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.num_labels = num_labels
        self.n_additional_tokens = n_additional_tokens
        
        if n_additional_tokens > 0:
            prev_embedding = self.mamba.backbone.embeddings
            self.mamba.backbone.embeddings = nn.Embedding(
                self.mamba.config.vocab_size + n_additional_tokens, self.mamba.config.d_model,
            )
            with torch.no_grad():
                self.mamba.backbone.embeddings.weight[:-n_additional_tokens] = prev_embedding.weight
                
        self.mamba.lm_head = nn.Linear(self.mamba.config.d_model, num_labels)

    def freeze(self) -> None:
        '''Freeze all layers except embedding and classification'''
        for name, parameter in self.mamba.named_parameters():
            if ('embeddings' not in name) and ('lm_head' not in name):
                parameter.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor,
        eos_pos: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Dict:
        # input_ids: (b, l), eos_pos: (b,), token_type_ids: (b,), labels: (b,)            
        output = self.mamba(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_logits = output.logits[range(len(eos_pos)), eos_pos, :]

        output = {'logits': cls_token_logits}

        if labels is not None:
            output['loss'] = F.cross_entropy(cls_token_logits, labels)

        return output


if __name__ == "__main__":
    model = MambaForSequenceClassification('state-spaces/mamba-130m-hf')
    