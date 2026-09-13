import torch
import torch.nn as nn

from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class GemmaModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.gemma = AutoModel.from_pretrained(
            "google/gemma-3-1b-it"
        )

        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        self.gemma = get_peft_model(
            self.gemma,
            lora_config
        )

        self.classifier = nn.Linear(
            self.gemma.config.hidden_size,
            2
        ).to(dtype=self.gemma.dtype)

    def forward(self, input_ids, attention_mask):

        outputs = self.gemma(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_idx = attention_mask.sum(dim=1) - 1

        hidden = outputs.last_hidden_state[
            torch.arange(
                input_ids.size(0),
                device=input_ids.device
            ),
            last_idx
        ]

        logits = self.classifier(hidden)

        return logits
