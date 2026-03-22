import argparse
from pathlib import Path

from tqdm import tqdm

from diffopenie.evaluation.carb_metrics import (
    evaluate,
    load_gold_file,
    load_predicted_file,
)
from diffopenie.models.discrete.discrete_model import DiscreteModel
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config


def load_model(config: TrainingConfig, checkpoint_path: Path) -> DiscreteModel:
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


def format_carb_txt(sentence: str, predicate: str, subject: str, object_: str) -> str:
    """Format a triplet in CARB TXT format."""
    return f"1 ({subject} ; {predicate} ; {object_})"


SentenceExtractions = list[tuple[str, str, str, float]]  # (pred, subj, obj, prob)


def process_sentences(
    model: DiscreteModel,
    sentences: list[str],
) -> list[tuple[str, SentenceExtractions]]:
    """
    Process sentences and extract triplet distributions.

    Returns:
        List of (sentence, extractions) where extractions is a list of
        (predicate, subject, object, probability) tuples in descending
        probability order.
    """
    results = []

    for sentence in tqdm(sentences, desc="Processing sentences"):
        words = sentence.split()
        triplets, probs = model.get_carb_prediction(words)

        extractions: SentenceExtractions = []
        for (sub_span, obj_span, pred_span), prob in zip(triplets, probs):
            subject = extract_span_text(words, sub_span)
            object_ = extract_span_text(words, obj_span)
            predicate = extract_span_text(words, pred_span)
            if subject and predicate and object_:
                extractions.append((predicate, subject, object_, prob))

        results.append((sentence, extractions))

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
        "--gold",
        type=Path,
        default=None,
        help="Path to CaRB gold TSV file; if provided, in-house metrics are printed after evaluation",
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

    print(f"Processing {len(sentences)} sentences")

    # Process sentences and get triplet distributions
    predictions = process_sentences(model, sentences)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write TSV format (CaRB tabbed format: sent, prob, pred, arg1, arg2, ...)
    tsv_path = args.output_dir / "extractions.tsv"
    print(f"Writing TSV predictions to {tsv_path}...")
    with open(tsv_path, "w", encoding="utf-8") as f:
        for sentence, extractions in predictions:
            for predicate, subject, object_, prob in extractions:
                f.write(
                    format_carb_tsv(sentence, predicate, subject, object_, prob=prob)
                    + "\n"
                )

    # Write TXT format
    txt_path = args.output_dir / "extractions.txt"
    print(f"Writing TXT predictions to {txt_path}...")
    with open(txt_path, "w", encoding="utf-8") as f:
        for sentence, extractions in predictions:
            f.write(sentence + "\n")
            for predicate, subject, object_, _prob in extractions:
                f.write(format_carb_txt(sentence, predicate, subject, object_) + "\n")
            f.write("\n")

    # Print statistics
    total_extractions = sum(len(exts) for _, exts in predictions)
    sentences_with_extractions = sum(1 for _, exts in predictions if exts)
    print("\nCompleted!")
    print(f"Total sentences: {len(sentences)}")
    print(f"Sentences with extractions: {sentences_with_extractions}")
    print(f"Total extractions: {total_extractions}")
    print(f"Avg extractions per sentence: {total_extractions / len(sentences):.2f}")
    print(f"Output files saved to: {args.output_dir}")

    if args.gold:
        print(f"\nComputing in-house CaRB metrics against {args.gold}...")
        gold = load_gold_file(str(args.gold))
        predicted = load_predicted_file(str(tsv_path))
        result = evaluate(gold, predicted)
        print(f"  AUC:       {result.auc:.3f}")
        print(f"  Precision: {result.precision:.3f}")
        print(f"  Recall:    {result.recall:.3f}")
        print(f"  F1:        {result.f1:.3f}")


if __name__ == "__main__":
    main()
