import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


STOPWORDS = {
    "a",
    "an",
    "and",
    "of",
    "the",
    "to",
    "with",
    "in",
    "on",
    "for",
    "style",
    "photo",
    "portrait",
    "face",
    "human",
    "person",
    "realistic",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze synthetic image prompts with simple NLP features.")
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/metadata/synthetic_prompts.csv"),
        help="Synthetic prompt metadata CSV from generate_synthetic.py.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("results/predictions.csv"),
        help="Prediction CSV from evaluate_model.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/prompt_analysis"))
    parser.add_argument("--top-k", type=int, default=20, help="Number of top keywords to plot.")
    return parser.parse_args()


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def tokenize(text):
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def image_key(path):
    return Path(path).name


def load_predictions(path):
    if not path.exists():
        return {}

    rows = read_csv(path)
    predictions = {}
    for row in rows:
        if row.get("true_label") == "synthetic":
            predictions[image_key(row["image_path"])] = row
    return predictions


def enrich_rows(metadata_rows, predictions):
    enriched = []
    for row in metadata_rows:
        prompt = row["prompt"]
        tokens = tokenize(prompt)
        prediction = predictions.get(image_key(row["image_path"]), {})
        confidence = prediction.get("confidence")

        enriched.append(
            {
                **row,
                "prompt_word_count": len(prompt.split()),
                "prompt_keyword_count": len(tokens),
                "keywords": " ".join(tokens),
                "predicted_label": prediction.get("predicted_label", ""),
                "confidence": float(confidence) if confidence else None,
                "correct": prediction.get("correct", ""),
            }
        )
    return enriched


def write_enriched_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "label",
        "prompt_id",
        "prompt",
        "prompt_word_count",
        "prompt_keyword_count",
        "keywords",
        "negative_prompt",
        "seed",
        "width",
        "height",
        "steps",
        "guidance_scale",
        "model_id",
        "predicted_label",
        "confidence",
        "correct",
    ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_keyword_frequency(output_path, counter, top_k):
    top_items = counter.most_common(top_k)
    if not top_items:
        return

    keywords = [item[0] for item in top_items]
    counts = [item[1] for item in top_items]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=keywords, color="#3b82f6")
    plt.xlabel("Frequency")
    plt.ylabel("Keyword")
    plt.title("Top Prompt Keywords")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_prompt_lengths(output_path, rows):
    lengths = [row["prompt_word_count"] for row in rows]
    if not lengths:
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(lengths, bins=10, color="#10b981")
    plt.xlabel("Prompt word count")
    plt.ylabel("Number of images")
    plt.title("Prompt Length Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confidence_by_prompt(output_path, rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["confidence"] is not None:
            grouped[row["prompt_id"]].append(row["confidence"])

    if not grouped:
        return

    prompt_ids = sorted(grouped, key=lambda item: int(item))
    means = [sum(grouped[prompt_id]) / len(grouped[prompt_id]) for prompt_id in prompt_ids]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=prompt_ids, y=means, color="#f97316")
    plt.xlabel("Prompt ID")
    plt.ylabel("Average classifier confidence")
    plt.ylim(0, 1)
    plt.title("Average Confidence by Prompt ID")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_summary(rows, keyword_counter):
    confidences = [row["confidence"] for row in rows if row["confidence"] is not None]
    prompt_lengths = [row["prompt_word_count"] for row in rows]
    unique_prompts = len({row["prompt"] for row in rows})

    summary = {
        "num_synthetic_images": len(rows),
        "unique_prompts": unique_prompts,
        "average_prompt_word_count": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
        "min_prompt_word_count": min(prompt_lengths) if prompt_lengths else 0,
        "max_prompt_word_count": max(prompt_lengths) if prompt_lengths else 0,
        "top_keywords": keyword_counter.most_common(20),
        "prediction_rows_matched": len(confidences),
        "average_prediction_confidence": sum(confidences) / len(confidences) if confidences else None,
    }
    return summary


def write_report(path, summary):
    lines = [
        "Prompt Analysis Report",
        "======================",
        "",
        f"Synthetic images analyzed: {summary['num_synthetic_images']}",
        f"Unique prompts: {summary['unique_prompts']}",
        f"Average prompt word count: {summary['average_prompt_word_count']:.2f}",
        f"Min prompt word count: {summary['min_prompt_word_count']}",
        f"Max prompt word count: {summary['max_prompt_word_count']}",
        f"Prediction rows matched: {summary['prediction_rows_matched']}",
    ]

    avg_confidence = summary["average_prediction_confidence"]
    if avg_confidence is not None:
        lines.append(f"Average prediction confidence: {avg_confidence:.4f}")
    else:
        lines.append("Average prediction confidence: N/A")

    lines.extend(["", "Top keywords:"])
    for keyword, count in summary["top_keywords"]:
        lines.append(f"- {keyword}: {count}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = read_csv(args.metadata_path)
    predictions = load_predictions(args.predictions_path)
    enriched_rows = enrich_rows(metadata_rows, predictions)

    keyword_counter = Counter()
    for row in enriched_rows:
        keyword_counter.update(row["keywords"].split())

    summary = build_summary(enriched_rows, keyword_counter)

    write_enriched_csv(args.output_dir / "prompt_analysis.csv", enriched_rows)
    (args.output_dir / "prompt_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir / "prompt_analysis_report.txt", summary)
    plot_keyword_frequency(args.output_dir / "prompt_keyword_frequency.png", keyword_counter, args.top_k)
    plot_prompt_lengths(args.output_dir / "prompt_length_distribution.png", enriched_rows)
    plot_confidence_by_prompt(args.output_dir / "confidence_by_prompt_id.png", enriched_rows)

    print(f"Analyzed {len(enriched_rows)} synthetic prompt rows")
    print(f"Saved prompt analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
