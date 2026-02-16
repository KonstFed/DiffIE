"""CLI to run triplet generation on text using a trained model (config + checkpoint)."""

import argparse
from pathlib import Path

from diffopenie.evaluation.carb_eval import load_model, extract_span_text
from diffopenie.models.base_model import BaseTripletModel
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config


def sample_triplets(
    model: BaseTripletModel,
    sentences: list[str],
    num_samples_per_sentence: int = 1,
) -> list[list[tuple[str, str, str]]]:
    """
    Sample triplets per sentence (stochastic diffusion sampling when num_samples > 1).

    Returns:
        For each sentence, a list of (subject, predicate, object) text tuples.
    """
    results = []
    for sentence in sentences:
        words = sentence.split()
        sentence_triplets = []
        for _ in range(num_samples_per_sentence):
            triplets = model.get_triplets([words])
            (sub_span, obj_span, pred_span) = triplets[0]
            subject = extract_span_text(words, sub_span)
            predicate = extract_span_text(words, pred_span)
            object_ = extract_span_text(words, obj_span)
            sentence_triplets.append((subject, predicate, object_))
        results.append(sentence_triplets)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run triplet generation on text using config and checkpoint (like diversity_EDA)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config YAML (e.g. config/training_discrete.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (e.g. checkpoints/discrete/checkpoint_epoch_100.pt)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single sentence to run generation on (words separated by spaces)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to file with one sentence per line (alternative to --text)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of triplet samples per sentence for diversity (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write results (one line per triplet: subject ; predicate ; object)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. cuda, cpu). Default: from config (auto cuda/cpu)",
    )
    args = parser.parse_args()

    if not args.text and not args.input:
        parser.error("Provide at least one of --text or --input")

    sentences = []
    if args.text:
        sentences.append(args.text.strip())
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)

    print("Loading config and model...")
    config = load_config(TrainingConfig, args.config)
    if args.device is not None:
        config.trainer.device = args.device
    model = load_model(config, args.checkpoint)
    if args.device is not None:
        model = model.to(args.device)
        if hasattr(model, "scheduler") and hasattr(model.scheduler, "to"):
            model.scheduler.to(args.device)
    model.eval()

    print(f"Generating triplets for {len(sentences)} sentence(s), {args.num_samples} sample(s) each...")
    sampled = sample_triplets(model, sentences, num_samples_per_sentence=args.num_samples)

    lines = []
    for sent, triplets in zip(sentences, sampled):
        print(f"Sentence: {sent}")
        for i, (subj, pred, obj) in enumerate(triplets):
            line = f"  [{i + 1}] ({subj} ; {pred} ; {obj})"
            print(line)
            lines.append(f"{subj} ; {pred} ; {obj}")
        print()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Wrote {len(lines)} triplets to {args.output}")


if __name__ == "__main__":
    main()
