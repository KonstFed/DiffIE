from pathlib import Path
import argparse
from tqdm import tqdm

from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config
from diffopenie.models.diffusion_model import DiffusionSequenceLabeler


def load_model(
    config: TrainingConfig, checkpoint_path: Path
) -> DiffusionSequenceLabeler:
    model = config.model.create()
    trainer = config.trainer.create(model=model)
    trainer.load_checkpoint(checkpoint_path)
    return trainer.model


def extract_span_text(words: list[str], span: tuple[int, int] | None) -> str:
    """Extract text from a word span."""
    if span is None:
        return ""
    start, end = span
    return " ".join(words[start : end + 1])


def format_carb_tsv(
    sentence: str, predicate: str, subject: str, object_: str, prob: float = 1.0
) -> str:
    """
    Format a triplet in CARB tabbed format.

    According to CaRB README, tabbed format is:
    sent, prob, pred, arg1, arg2, ...
    """
    return f"{sentence}\t{prob}\t{predicate}\t{subject}\t{object_}"


def format_carb_txt(
    sentence: str, predicate: str, subject: str, object_: str
) -> str:
    """Format a triplet in CARB TXT format."""
    return f"1 ({subject} ; {predicate} ; {object_})"


def process_sentences(
    model: DiffusionSequenceLabeler,
    sentences: list[str],
    batch_size: int = 32,
) -> list[tuple[str, str, str, str]]:
    """
    Process sentences and extract triplets.

    Returns:
        List of (sentence, predicate, subject, object) tuples
    """
    results = []

    for i in tqdm(
        range(0, len(sentences), batch_size), desc="Processing sentences"
    ):
        batch_sentences = sentences[i : i + batch_size]
        batch_words = [sentence.split() for sentence in batch_sentences]

        # Get triplets for the batch
        triplets = model.get_triplets(batch_words)

        # Format results
        for sentence, words, (sub_span, obj_span, pred_span) in zip(
            batch_sentences, batch_words, triplets
        ):
            subject = extract_span_text(words, sub_span)
            object_ = extract_span_text(words, obj_span)
            predicate = extract_span_text(words, pred_span)

            # Only include if we have at least subject, predicate, and object
            if subject and predicate and object_:
                results.append((sentence, predicate, subject, object_))
            else:
                # Include empty extraction if no valid triplet found
                results.append((sentence, "", "", ""))

    return results


def main():
    argparser = argparse.ArgumentParser(
        description="Evaluate model on CARB benchmark and save predictions"
    )
    argparser.add_argument(
        "--config", type=Path, required=True, help="Path to training config"
    )
    argparser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    argparser.add_argument(
        "--input-sentences",
        type=Path,
        required=True,
        help="Path to input sentences file (one sentence per line)",
    )
    argparser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save output files",
    )
    argparser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing sentences",
    )
    args = argparser.parse_args()

    # Load model
    print("Loading model...")
    config = load_config(TrainingConfig, args.config)
    model = load_model(config, args.checkpoint_path)
    model.eval()

    # Read sentences
    print(f"Reading sentences from {args.input_sentences}...")
    with open(args.input_sentences, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(sentences)} sentences...")

    # Process sentences and get predictions
    predictions = process_sentences(model, sentences, batch_size=args.batch_size)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write TSV format (CaRB tabbed format: sent, prob, pred, arg1, arg2, ...)
    tsv_path = args.output_dir / "extractions.tsv"
    print(f"Writing TSV predictions to {tsv_path}...")
    with open(tsv_path, "w", encoding="utf-8") as f:
        for sentence, predicate, subject, object_ in predictions:
            if predicate and subject and object_:
                f.write(
                    format_carb_tsv(sentence, predicate, subject, object_, prob=1.0)
                    + "\n"
                )

    # Write TXT format
    txt_path = args.output_dir / "extractions.txt"
    print(f"Writing TXT predictions to {txt_path}...")
    with open(txt_path, "w", encoding="utf-8") as f:
        for sentence, predicate, subject, object_ in predictions:
            f.write(sentence + "\n")
            if predicate and subject and object_:
                f.write(
                    format_carb_txt(sentence, predicate, subject, object_) + "\n"
                )
            f.write("\n")

    # Print statistics
    valid_extractions = sum(1 for _, p, s, o in predictions if p and s and o)
    print("\nCompleted!")
    print(f"Total sentences: {len(sentences)}")
    print(f"Valid extractions: {valid_extractions}")
    print(f"Output files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
