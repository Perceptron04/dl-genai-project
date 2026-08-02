import re
import random
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score

# The five answer options, in a fixed order. Index i corresponds to OPTS[i]
# everywhere in the project: in labels, in probability array columns, and
# when converting predictions back to letters.
OPTS = ["A", "B", "C", "D", "E"]

SEED = 42


# Reproducibility
def seed_everything(seed: int = SEED) -> None:
    """Fix every random number generator used in this project.

    A neural network draws randomness from several independent sources:
    weight initialisation, per-epoch data shuffling, and dropout. Each
    library keeps its own generator, so seeding one is not enough.
    """
    random.seed(seed)                 
    np.random.seed(seed)              
    torch.manual_seed(seed)           
    torch.cuda.manual_seed_all(seed)  


def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"



# Evaluation metric: MAP@3
def map_at_3(true_letters, pred_lists) -> float:
    """Mean Average Precision at 3.

    Each question has exactly one correct answer, so average precision
    reduces to the reciprocal of the rank at which the correct answer
    appears: 1.0 at rank 1, 0.5 at rank 2, 1/3 at rank 3, 0 otherwise.

    Args:
        true_letters: list of correct letters, e.g. ["B", "A", ...]
        pred_lists:   list of ranked top-3 guesses, e.g. [["B", "C", "A"], ...]
    """
    total = 0.0
    for gt, preds in zip(true_letters, pred_lists):
        for rank, p in enumerate(preds[:3]):
            if p == gt:
                total += 1.0 / (rank + 1)
                break                       
    return total / len(true_letters)


def probs_to_top3(probs: np.ndarray):
    """Convert an (N, 5) score array into ranked top-3 letter lists."""
    order = np.argsort(-probs, axis=1)[:, :3]  
    return [[OPTS[j] for j in row] for row in order]


def evaluate(true_letters, probs: np.ndarray) -> dict:
    """Return MAP@3, accuracy and macro F1 together.

    All three models report the same dictionary, which is what makes the
    Weights & Biases run comparison meaningful.
    """
    top3 = probs_to_top3(probs)
    y_true = np.array([OPTS.index(a) for a in true_letters])
    y_pred = probs.argmax(axis=1)
    return {
        "map@3": round(map_at_3(true_letters, top3), 4),
        "accuracy": round(float((y_pred == y_true).mean()), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


# Text preprocessing
_PREFIX = re.compile(
    r"^\s*(pick the best possible answer|choose the correct.*?|"
    r"select the (?:correct|best).*?|identify the.*?)\s*:\s*",
    re.I,
)
_SUFFIX = re.compile(
    r"\s*(among the listed options|from the following choices|"
    r"from the options|carefully)\.?\s*$",
    re.I,
)


def clean_text(s) -> str:
    """Strip templated wrappers and normalise whitespace."""
    s = str(s).strip()
    s = _PREFIX.sub("", s)
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def apply_cleaning(df, opts=OPTS):
    """Add a `prompt_clean` column and clean the option columns in place.

    The identical function is applied to train and test so that no drift
    can appear between what the model learns and what it sees at inference.
    The raw `prompt` column is never overwritten.
    """
    df["prompt_clean"] = df["prompt"].map(clean_text)
    for o in opts:
        df[o] = df[o].map(clean_text)
    return df


# Word-level tokenisation for the from-scratch model
def simple_tokenize(text) -> list:
    """Lowercase and split into alphanumeric tokens, discarding punctuation."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def build_vocab(df, columns, max_size: int = 20000) -> dict:
    """Build a word -> index vocabulary from the given text columns.

    Index 0 is reserved for <pad> so that sequences can be padded to a
    common length, and index 1 for <unk> so that unseen words do not
    raise a lookup error. Real words start at index 2.
    """
    counter = Counter()
    for col in columns:
        for text in df[col]:
            counter.update(simple_tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_size):
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab: dict, maxlen: int = 80) -> list:
    """Encode text into a fixed-length list of vocabulary indices."""
    ids = [vocab.get(w, 1) for w in simple_tokenize(text)]  
    ids = ids[:maxlen]                                      
    return ids + [0] * (maxlen - len(ids))                 


if __name__ == "__main__":
    t = ["A", "B"]
    p = [["A", "X", "Y"], ["X", "B", "Y"]]
    score = map_at_3(t, p)
    assert abs(score - 0.75) < 1e-9, f"expected 0.75, got {score}"
    print(f"map_at_3 self-test passed: {score}")

    demo = "Pick the best possible answer: What is entropy? among the listed options."
    print(f"before : {demo}")
    print(f"after  : {clean_text(demo)}")
