from pathlib import Path
import argparse

from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config
from diffopenie.training.base_trainer import BaseTrainer
from diffopenie.models.diffusion_model import DiffusionSequenceLabeler

def load_model(config: TrainingConfig, checkpoint_path: Path) -> DiffusionSequenceLabeler:
    model = config.model.create()
    trainer = config.trainer.create(model=model)
    trainer.load_checkpoint(checkpoint_path)
    return trainer.model


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=Path, required=True)
    argparser.add_argument("--checkpoint-path", type=Path, required=True)
    args = argparser.parse_args()

    config = load_config(TrainingConfig, args.config)
    model = load_model(config, args.checkpoint_path)
    sentence = "It was a notable influence on John Buchan and Ken Follett , who described it as `` an open-air adventure thriller about two young men who stumble upon a German armada preparing to invade England . ''"
    words = sentence.split()
    sub_span, obj_span, pred_span = model.get_triplets([words, words])[0]
    if sub_span is not None:
        print(f"Sub: {words[sub_span[0]:sub_span[1]+1]}")
    if obj_span is not None:
        print(f"Obj: {words[obj_span[0]:obj_span[1]+1]}")
    if pred_span is not None:
        print(f"Pred: {words[pred_span[0]:pred_span[1]+1]}")


if __name__ == "__main__":
    main()