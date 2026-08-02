import argparse
import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model

from utils import (
    OPTS, SEED, seed_everything, get_device, evaluate,
    apply_cleaning, build_vocab, encode,
)

DEVICE = get_device()
MAX_LEN = 256



# Model 1 — from-scratch BiGRU dual encoder
class ScratchBiGRU(nn.Module):
    """A Siamese dual-encoder trained from random initialisation.

    The same encoder processes the prompt and each option. Each text is
    embedded, passed through a bidirectional GRU, and mean-pooled over
    real (non-padding) tokens. The prompt vector is concatenated with each
    option vector and scored by a small MLP, giving five scores per row.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 1),
        )

    def encode(self, x):
        """Mean-pool the BiGRU outputs over real tokens only."""
        mask = (x != 0).float().unsqueeze(-1)   
        out, _ = self.gru(self.embedding(x))
        summed = (out * mask).sum(1)
        return summed / mask.sum(1).clamp(min=1)

    def forward(self, prompt_ids, option_ids):
        B, C, L = option_ids.shape                    
        p_vec = self.encode(prompt_ids)                             
        o_vec = self.encode(option_ids.reshape(B * C, L)).reshape(B, C, -1)
        p_rep = p_vec.unsqueeze(1).expand(-1, C, -1)               
        joined = torch.cat([p_rep, o_vec], dim=-1)
        return self.scorer(self.dropout(joined)).squeeze(-1)       


# Diagnostic — prompt-blind model

class PromptBlindModel(nn.Module):
    """Scores each option from its own text alone.

    This model has no prompt argument in its forward pass at all, so it is
    architecturally incapable of reading the question. If it beats random
    guessing, the correct answers must be identifiable from their writing
    style — which is exactly what the artifact investigation found.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def encode(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        out, _ = self.gru(self.embedding(x))
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, option_ids):       
        B, C, L = option_ids.shape
        o_vec = self.encode(option_ids.reshape(B * C, L)).reshape(B, C, -1)
        return self.scorer(o_vec).squeeze(-1)


# Collator for multiple-choice transformers

@dataclass
class MCQCollator:
    """Packs a list of examples into a (batch, 5, seq_len) tensor.

    The dataset stores five tokenised sequences per row. The standard
    collator only understands (batch, seq_len), so the five choices are
    flattened, padded to a common length, and reshaped back into 3D.
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        labels = [f.pop("label") for f in features] if "label" in features[0] else None
        bs, nc = len(features), 5
        flat = [{k: v[i] for k, v in f.items()} for f in features for i in range(nc)]
        batch = self.tokenizer.pad(flat, padding=True, return_tensors="pt")
        batch = {k: v.view(bs, nc, -1) for k, v in batch.items()}
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def build_compute_metrics():
    """Return a compute_metrics function that reuses the shared evaluate()."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        letters = [OPTS[i] for i in labels]
        return evaluate(letters, probs)

    return compute_metrics


# Data preparation
def load_and_split(data_dir: str, test_size: float = 0.15):
    """Load the CSVs, clean the text, and make a stratified split."""
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")

    train = apply_cleaning(train)
    test = apply_cleaning(test)

    train["label"] = train["answer"].map({l: i for i, l in enumerate(OPTS)})

    # stratify keeps the A-E distribution identical in both halves
    train_df, val_df = train_test_split(
        train, test_size=test_size, random_state=SEED, stratify=train["answer"]
    )
    return train.reset_index(drop=True), test, train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_preprocess(tokenizer, max_len: int = MAX_LEN):
    """Build a preprocess function that expands one row into five pairs.

    Pair tokenisation produces [CLS] question [SEP] option [SEP], which is
    the same sentence-pair format the model was pretrained with.
    """

    def preprocess(example):
        first = [example["prompt_clean"]] * 5
        second = [example[o] for o in OPTS]
        tok = tokenizer(first, second, truncation=True, max_length=max_len)
        out = {k: v for k, v in tok.items()}
        if "label" in example:
            out["label"] = example["label"]
        return out

    return preprocess


def make_datasets(train_df, val_df, test_df, tokenizer):
    preprocess = make_preprocess(tokenizer)
    cols = ["prompt_clean"] + OPTS + ["label"]
    te_cols = ["prompt_clean"] + OPTS
    tr = Dataset.from_pandas(train_df[cols], preserve_index=False).map(preprocess, remove_columns=cols)
    va = Dataset.from_pandas(val_df[cols], preserve_index=False).map(preprocess, remove_columns=cols)
    te = Dataset.from_pandas(test_df[te_cols], preserve_index=False).map(preprocess, remove_columns=te_cols)
    return tr, va, te


# Model 1 training
def train_scratch(train_df, val_df, vocab, epochs: int = 12, batch: int = 64):
    """Train the from-scratch BiGRU and return its validation probabilities."""

    def make_tensors(df):
        P = torch.tensor([encode(t, vocab) for t in df["prompt_clean"]], dtype=torch.long)
        O = torch.stack(
            [torch.tensor([encode(t, vocab) for t in df[o]], dtype=torch.long) for o in OPTS],
            dim=1,
        )
        y = torch.tensor(df["label"].values, dtype=torch.long)
        return P.to(DEVICE), O.to(DEVICE), y.to(DEVICE)

    P_tr, O_tr, y_tr = make_tensors(train_df)
    P_va, O_va, _ = make_tensors(val_df)

    model = ScratchBiGRU(len(vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()                                    
        perm = torch.randperm(len(P_tr), device=DEVICE)  
        for i in range(0, len(P_tr), batch):
            idx = perm[i : i + batch]
            loss = loss_fn(model(P_tr[idx], O_tr[idx]), y_tr[idx])
            optimizer.zero_grad()   
            loss.backward()
            optimizer.step()

        model.eval()                                     
        with torch.no_grad():
            probs = torch.softmax(model(P_va, O_va), dim=1).cpu().numpy()
        print(f"epoch {epoch + 1:2d} | val {evaluate(val_df['answer'].tolist(), probs)}")

    return model, probs


# Transformer training with LoRA
def get_lora_config(model_name: str):
    """LoRA config. Layer names are model-specific.

    DeBERTa uses query_proj / key_proj / value_proj; BERT-style models such
    as ELECTRA and RoBERTa use query / key / value. Passing the wrong names
    attaches adapters to nothing and training silently does almost nothing.
    """
    if "deberta" in model_name:
        targets = ["query_proj", "key_proj", "value_proj"]
        save = ["classifier", "pooler"]
    else:
        targets = ["query", "key", "value"]
        save = ["classifier"]

    return LoraConfig(
        r=16,                    
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=targets,
        modules_to_save=save,    
        task_type="SEQ_CLS",
    )


def train_transformer(
    model_name: str,
    train_ds,
    val_ds,
    test_ds,
    tokenizer,
    val_answers,
    seeds=(42, 123, 777),
    epochs: int = 3,
    lr: float = 2e-4,           
    batch_size: int = 4,
    use_bf16: bool = False,
):
    """Train one transformer per seed and bag the predicted probabilities.

    Bagging averages away part of the random variation between runs, which
    comes from weight initialisation, data shuffling and dropout.
    """
    lora_config = get_lora_config(model_name)
    val_probs, test_probs = [], []

    for seed in seeds:
        print(f"\n{'=' * 55}\n{model_name}  |  seed {seed}\n{'=' * 55}")
        seed_everything(seed)

        model = get_peft_model(
            AutoModelForMultipleChoice.from_pretrained(model_name), lora_config
        )
        model.print_trainable_parameters()

        args = TrainingArguments(
            output_dir=f"out_{seed}",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            num_train_epochs=epochs,
            warmup_ratio=0.1,    
            max_grad_norm=1.0,      
            weight_decay=0.01,
            bf16=use_bf16,
            fp16=False,           
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            report_to=[],
            remove_unused_columns=False,
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=MCQCollator(tokenizer),
            compute_metrics=build_compute_metrics(),
            processing_class=tokenizer,
        )
        trainer.train()

        vp = torch.softmax(torch.tensor(trainer.predict(val_ds).predictions), dim=1).numpy()
        tp = torch.softmax(torch.tensor(trainer.predict(test_ds).predictions), dim=1).numpy()
        val_probs.append(vp)
        test_probs.append(tp)
        print(f"seed {seed} val -> {evaluate(val_answers, vp)}")

        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    bagged_val = np.mean(val_probs, axis=0)
    bagged_test = np.mean(test_probs, axis=0)
    print(f"\nbagged val -> {evaluate(val_answers, bagged_val)}")
    return bagged_val, bagged_test


# CLI
def main():
    parser = argparse.ArgumentParser(description="Train Smart MCQ Solver models")
    parser.add_argument("--model", default="deberta",
                        choices=["scratch", "deberta", "electra"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", default="probs")
    args = parser.parse_args()

    seed_everything(SEED)
    _, test_df, train_df, val_df = load_and_split(args.data_dir)
    val_answers = val_df["answer"].tolist()
    print(f"train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    if args.model == "scratch":
        vocab = build_vocab(train_df, ["prompt_clean"] + OPTS)
        print("vocabulary size:", len(vocab))
        _, val_probs = train_scratch(train_df, val_df, vocab)
        np.save(f"{args.out}_scratch_val.npy", val_probs)
        return

    model_name = ("microsoft/deberta-v3-base" if args.model == "deberta"
                  else "google/electra-base-discriminator")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tr, va, te = make_datasets(train_df, val_df, test_df, tokenizer)

    bagged_val, bagged_test = train_transformer(
        model_name, tr, va, te, tokenizer, val_answers,
        seeds=tuple(args.seeds), epochs=args.epochs,
        batch_size=8 if args.model == "electra" else 4,
        use_bf16=(args.model == "electra"),
    )

    np.save(f"{args.out}_{args.model}_val.npy", bagged_val)
    np.save(f"{args.out}_{args.model}_test.npy", bagged_test)
    print(f"saved {args.out}_{args.model}_val.npy and {args.out}_{args.model}_test.npy")


if __name__ == "__main__":
    main()
