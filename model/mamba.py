import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

def pool_hidden_states(strategy, hidden_states, attention_mask=None):
    if strategy == "mean":
        if attention_mask is not None:
                # Apply attention mask for proper mean calculation
                expanded_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                sum_hidden = torch.sum(hidden_states * expanded_mask, dim=1)
                token_count = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)
                return sum_hidden / token_count
        else:
            return torch.mean(hidden_states, dim=1)
        
    elif strategy == "max":
        # Max pooling
        if attention_mask is not None:
            # Create mask for padding tokens (large negative value)
            mask = (1 - attention_mask).unsqueeze(-1).expand_as(hidden_states) * -10000.0
            return torch.max(hidden_states + mask, dim=1)[0]
        else:
            return torch.max(hidden_states, dim=1)[0]
        
    elif strategy == "first":
        return hidden_states[:, 0, :]
        
    elif strategy == "last":
        if attention_mask is not None:
            last_idx = torch.sum(attention_mask, dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, last_idx]
        else:
            return hidden_states[:, -1, :]
        
    else:
        raise ValueError(f"Unsupported pooling strategy: {strategy}")


class CustomMamba(nn.Module):
    def __init__(
        self,
        vocab_size=2442,
        dropout_prob: float = 0.1,
        strategy: str = "mean",
        hidden_size = 768,
        num_classes=1,
        n_layers=None
    ):
        super().__init__()
        #self.backbone = AutoModelForCausalLM.from_pretrained("StanfordShahLab/mamba-tiny-16384-clmbr")
        self.backbone = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        self.backbone.resize_token_embeddings(vocab_size)

        if not n_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = False

            for param in self.backbone.backbone.norm_f.parameters():
                param.requires_grad = True

            for i in range(-2, 0):  # Last two blocks
                for param in  self.backbone.backbone.layers[i].parameters():
                    param.requires_grad = True

        self.time_layer = nn.Linear(1, hidden_size)
        self.combine_layer = nn.Linear(hidden_size*2, hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self.strategy = strategy
    
    def forward(self, input_ids, times, attention_mask=None):
        token_embeddings = self.backbone.backbone.embeddings(input_ids)
        time_embeddings = self.time_layer(times.unsqueeze(-1))
        embs = token_embeddings + time_embeddings
                
        outputs = self.backbone.backbone(
            inputs_embeds = embs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        logits = pool_hidden_states(self.strategy, hidden_states, attention_mask)
        logits = self.classifier(logits)
        return logits