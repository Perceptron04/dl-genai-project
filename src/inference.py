import argparse

import numpy as np
import pandas as pd
import torch

from utils import OPTS, probs_to_top3, apply_cleaning, get_device

DEVICE = get_device()
ELECTRA_WEIGHT = 0.25

# Prediction from a saved checkpoint
def predict_from_checkpoint(checkpoint: str, test_df, max_len: int = 256,
                            batch_size: int = 16) -> np.ndarray:
    """Run a saved multiple-choice model over the test set.

    Each row is expanded into five (question, option) pairs, exactly as in
    training, and the model emits one logit per option.
    """
    from transformers import AutoTokenizer, AutoModelForMultipleChoice

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMultipleChoice.from_pretrained(checkpoint).to(DEVICE).eval()

    all_probs = []
    for start in range(0, len(test_df), batch_size):
        chunk = test_df.iloc[start : start + batch_size]

        first, second = [], []
        for _, row in chunk.iterrows():
            first.extend([row["prompt_clean"]] * 5)
            second.extend([row[o] for o in OPTS])

        enc = tokenizer(first, second, truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")
        enc = {k: v.view(len(chunk), 5, -1).to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.vstack(all_probs)


# Submission
def build_submission(test_df, probs: np.ndarray, sample_df,
                     out_path: str = "submission.csv") -> pd.DataFrame:
    """Write the submission file and verify its format.

    A malformed file wastes a scoring attempt, so the format is asserted
    rather than assumed: column names must match the sample exactly, the
    row count must match the test set, and every prediction must contain
    exactly three space-separated letters.
    """
    top3 = probs_to_top3(probs)
    submission = pd.DataFrame(
        {"ID": test_df["id"].values, "Prediction": [" ".join(t) for t in top3]}
    )
    submission = submission[list(sample_df.columns)]

    assert list(submission.columns) == list(sample_df.columns), "column names do not match"
    assert len(submission) == len(test_df), "row count does not match"
    assert submission["Prediction"].str.split().map(len).eq(3).all(), "need 3 letters per row"

    submission.to_csv(out_path, index=False)
    print(f"wrote {out_path}  shape={submission.shape}")


    print("first-letter distribution:")
    print(submission["Prediction"].str[0].value_counts().sort_index())
    return submission


def main():
    parser = argparse.ArgumentParser(description="Build the final submission")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--deberta", help="path to saved DeBERTa test probabilities (.npy)")
    parser.add_argument("--electra", help="path to saved ELECTRA test probabilities (.npy)")
    parser.add_argument("--checkpoint", help="run a saved model directly instead")
    parser.add_argument("--weight", type=float, default=ELECTRA_WEIGHT,
                        help="weight given to ELECTRA in the blend")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    test_df = apply_cleaning(pd.read_csv(f"{args.data_dir}/test.csv"))
    sample_df = pd.read_csv(f"{args.data_dir}/sample_submission.csv")

    if args.checkpoint:
        print(f"predicting from checkpoint: {args.checkpoint}")
        probs = predict_from_checkpoint(args.checkpoint, test_df)

    elif args.deberta and args.electra:
        deberta_probs = np.load(args.deberta)
        electra_probs = np.load(args.electra)
        w = args.weight
        probs = (1 - w) * deberta_probs + w * electra_probs
        print(f"blending: {1 - w:.2f} * DeBERTa + {w:.2f} * ELECTRA")

    elif args.deberta:
        probs = np.load(args.deberta)
        print("using DeBERTa alone (no blend)")

    else:
        parser.error("provide --checkpoint, or --deberta (optionally with --electra)")

    build_submission(test_df, probs, sample_df, args.out)


if __name__ == "__main__":
    main()
