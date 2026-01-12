"""Data collator for batching with padding."""

import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any


class DiffusionCollator:
    """
    Collator for batching diffusion training data.
    
    Handles:
    - Padding token_ids and attention masks
    - Padding label tensors
    - Creating attention masks
    """
    def __init__(
        self,
        pad_token_id: int = 0,
        pad_label_idx: int = 0,
    ):
        """
        Args:
            pad_token_id: Token ID for padding
            pad_label_idx: Label index for padding (typically 0 for O/padding)
        """
        self.pad_token_id = pad_token_id
        self.pad_label_idx = pad_label_idx
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Expected batch items:
        - token_ids: List[int]
        - labels: torch.Tensor of shape [L] with label indices (0=O, 1=subject, 2=object, 3=predicate)
        """
        # Extract token_ids and labels
        token_ids_list = [torch.tensor(item["token_ids"], dtype=torch.long) for item in batch]
        labels_list = [item.get("labels") for item in batch]
        
        # Pad token_ids
        token_ids = pad_sequence(
            token_ids_list,
            batch_first=True,
            padding_value=self.pad_token_id,
        )  # [B, L]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (token_ids != self.pad_token_id).long()  # [B, L]
        
        # Pad labels
        max_len = token_ids.size(1)
        label_indices_list = []
        
        for i, labels in enumerate(labels_list):
            if labels is None:
                # If no labels provided, create a dummy sequence
                label_indices = torch.zeros(len(token_ids_list[i]), dtype=torch.long)
            elif isinstance(labels, torch.Tensor):
                # Labels are already a tensor
                label_indices = labels.clone()
            else:
                # Convert list to tensor if needed
                label_indices = torch.tensor(labels, dtype=torch.long)
            
            # Pad or truncate to max_len
            if len(label_indices) < max_len:
                padding = torch.full(
                    (max_len - len(label_indices),),
                    self.pad_label_idx,
                    dtype=torch.long,
                )
                label_indices = torch.cat([label_indices, padding])
            elif len(label_indices) > max_len:
                label_indices = label_indices[:max_len]
            
            label_indices_list.append(label_indices)
        
        label_indices = torch.stack(label_indices_list)  # [B, L]
        
        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "label_indices": label_indices,
        }
