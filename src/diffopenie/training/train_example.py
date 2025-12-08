"""Example training script for diffusion-based OpenIE model.

This script demonstrates how to set up and run training for the diffusion model.
"""

import torch
from torch.utils.data import DataLoader

from diffopenie.models.diffusion.denoiser import DiffusionSLDenoiser
from diffopenie.models.diffusion.scheduler import LinearScheduler
from diffopenie.models.label_mapper import LabelMapper
from diffopenie.models.encoder import BERTEncoder
from diffopenie.dataset import SequenceLSOEIDataset
from diffopenie.training.trainer import DiffusionTrainer
from diffopenie.training.collator import DiffusionCollator


def main():
    """Main training function."""
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    num_steps = 1000  # Diffusion steps
    
    # Model dimensions
    x_dim = 256  # Dimension of label embeddings (should match label_mapper embedding_dim)
    bert_dim = 768  # BERT hidden size
    
    # Label classes: 0=O/padding, 1=subject, 2=object, 3=predicate
    num_classes = 4
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = SequenceLSOEIDataset(split="train")
    print(f"Dataset loaded: {len(train_dataset)} examples")
    
    # Initialize models
    print("Initializing models...")
    
    # BERT encoder
    encoder = BERTEncoder(model_name="bert-base-uncased", freeze=True)
    
    # Label mapper
    label_mapper = LabelMapper(num_classes=num_classes, embedding_dim=x_dim)
    
    # Diffusion scheduler
    scheduler = LinearScheduler(num_steps=num_steps)
    
    # Denoiser
    denoiser = DiffusionSLDenoiser(
        x_dim=x_dim,
        bert_dim=bert_dim,
        num_steps=num_steps,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
    )
    
    # Create trainer
    trainer = DiffusionTrainer(
        denoiser=denoiser,
        scheduler=scheduler,
        label_mapper=label_mapper,
        encoder=encoder,
        device=device,
        learning_rate=learning_rate,
    )
    
    # Optional: Set learning rate scheduler
    # from torch.optim.lr_scheduler import CosineAnnealingLR
    # lr_scheduler = CosineAnnealingLR(trainer.optimizer, T_max=num_epochs)
    # trainer.set_lr_scheduler(lr_scheduler)
    
    # Create data collator
    collator = DiffusionCollator(
        pad_token_id=0,  # BERT pad token ID
        pad_label_idx=0,  # Padding label index (0 = O/padding)
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
    )
    
    # Create validation dataloader (optional)
    val_dataset = SequenceLSOEIDataset(split="validation")  # or "test" depending on your dataset
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
    )
    
    print("Training setup complete!")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Number of classes: {num_classes} (0=O, 1=subject, 2=object, 3=predicate)")
    print(f"Diffusion steps: {num_steps}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Start training with validation
    trainer.train(
        train_dataloader=train_dataloader,
        num_epochs=num_epochs,
        log_interval=100,
        save_path="./checkpoints",
        save_interval=1,
        val_dataloader=val_dataloader,  # Pass validation dataloader
        num_classes=num_classes,  # Number of classes for metrics
    )


if __name__ == "__main__":
    main()
