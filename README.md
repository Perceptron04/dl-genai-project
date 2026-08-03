# Smart MCQ Solver Challenge

**Intro to Deep Learning & Generative AI — Diploma Project**
Sagar K Chaudhary | Roll No: 23f2002523 | BS in Data Science and Applications, IIT Madras

Ranking the top-3 answers for five-option multiple-choice questions, evaluated by MAP@3.

**Final public leaderboard score: 0.75644** (cutoff: 0.73)

---

## Problem

Each row contains a question (`prompt`) and five candidate answers (`A`–`E`), exactly one of
which is correct. The task is to output the three most likely answers in ranked order.

MAP@3 awards 1.0 if the correct answer is ranked first, 0.5 if second, 0.333 if third, and
0 if absent from the top three.

| | Train | Test |
|---|---|---|
| Rows | 2,000 | 500 |
| Columns | 8 (`id`, `prompt`, `A`–`E`, `answer`) | 7 (`answer` withheld) |
| Missing values | 0 | 0 |

---

## Models

Four models were built (the specification requires three), plus one diagnostic model.

| # | Model | Type | Validation MAP@3 |
|---|-------|------|------------------|
| 1 | BiGRU dual-encoder | built from scratch | 1.0000 (artifact — see below) |
| 2 | all-MiniLM-L6-v2 | pretrained, zero-shot | 0.4083 |
| 3 | DeBERTa-v3-base + LoRA | fine-tuned, primary scorer | 0.9983 |
| 4 | ELECTRA-base + LoRA | fine-tuned, added for diversity | 0.8900 |
| — | Prompt-blind model | diagnostic only | 1.0000 |

LoRA trains only **1,476,097 of 185,899,010 parameters (0.79%)**.

Architecture code for every model is documented in [`models`]

---

## Key finding: a dataset artifact

The from-scratch model — 644,737 parameters trained on 1,700 rows — reached a perfect
validation score. This was treated as an anomaly rather than a success.

A **prompt-blind model** was built with no prompt argument in its forward pass, making it
architecturally incapable of reading the question. It scored **100% accuracy** against a 20%
random baseline, proving the correct answers carry a writing-style signature identifying them
without the question. A diagnostic leaderboard submission scored **0.75311**, confirming the
artifact transfers to the hidden test set.

The finding was reported to and confirmed by the course TA, who advised that the final model
should not rely on this signal given a possible private leaderboard.

Two mitigation attempts both reduced artifact reliance **and** reduced the score, establishing
that the public test set rewards the artifact:

| Configuration | Agreement with prompt-blind | Public LB |
|---|---|---|
| Baseline (3-seed bagged DeBERTa) | 99.7% | 0.75187 |
| Option-order shuffling | 95.7% | 0.73773 |
| Label smoothing + cosine schedule | 96.0% | 0.74937 |

---

## Experiment log

Because validation is artifact-contaminated, every decision was verified on the leaderboard.

| Approach | Public LB | Outcome |
|---|---|---|
| Prompt-blind diagnostic (artifact only) | 0.75311 | excluded per TA guidance |
| DeBERTa-base + LoRA, single | 0.75062 | baseline |
| DeBERTa-base + LoRA, 3-seed bagged | 0.75187 | best single-family result |
| DeBERTa-large + LoRA | 0.74937 | 3× capacity, no gain |
| 5-fold cross-validation bagging | 0.74930 | all data used, no gain |
| Label smoothing + cosine | 0.74937 | agreement fell, score fell |
| Option-order shuffling | 0.73773 | mitigation reduced score |
| **DeBERTa + ELECTRA blend, w=0.25** | **0.75644** | **final submission** |
| DeBERTa + ELECTRA blend, w=0.40 | 0.75602 | same within noise |
| Three-way blend with RoBERTa | 0.74896 | third family did not help |

**On measurement resolution:** with 500 test rows, one row is worth 1/500 = 0.002 MAP@3.
Six single-model configurations spanning 644K to 400M parameters all landed within a 0.004
band — approximately two rows. Capacity was never the bottleneck.

The only change to exceed that band was **cross-family blending**. ELECTRA agrees with
DeBERTa on only **83%** of validation rows, unlike same-family methods where agreement exceeds
99%. Ensembling requires diversity, not multiplicity.

---
## Repo Structure

```text
dl-genai-project/
├── notebooks/
│   ├── miletsone-1.ipynb
│   ├── milestone-2.ipynb
│   ├── milestone-3.ipynb
│   ├── milestone-4.ipynb
│   ├── milestone-5.ipynb
│   ├── test_notebook.ipynb
│   └── final_notebook.ipynb        
├── src/
│   ├── utils.py                  
│   ├── train.py                   
│   └── inference.py            
├── models/
│   ├── Model 1 — BiGRU.ipynb
│   ├── Model 2 — MiniLM.ipynb
│   ├── Model 3 — DeBERTa-v3-base + LoRA.ipynb
│   └── Model 4 — ELECTRA-base + LoRA.ipynb                 
├── reports/
│   ├── milestone-1-report.pdf
│   ├── milestone-2-report.pdf
│   ├── milestone-3-report.pdf
│   ├── milestone-4-report.pdf
│   ├── milestone-5-report.pdf
│   └── final-report.pdf
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <repo-url>
cd dl-genai-project
pip install -r requirements.txt
```

Place the competition CSVs in `data/` (not committed — see `.gitignore`):
data/train.csv
data/test.csv
data/sample_submission.csv


> **Note for Kaggle:** the pre-installed `torchao` version conflicts with current `peft`.
> Run `pip uninstall -y torchao` and restart the kernel before importing. The notebook
> handles this in its first cell.

---

## Usage

Train the models:

```bash
python src/train.py --model deberta --seeds 42 123 777 --data-dir data
python src/train.py --model electra --seeds 42 123 --data-dir data
```

Build the final blended submission:

```bash
python src/inference.py \
    --deberta probs_deberta_test.npy \
    --electra probs_electra_test.npy \
    --weight 0.25 \
    --data-dir data
```

Or reproduce everything end to end by running `notebooks/final_notebook.ipynb` on a GPU
(approximately 50 minutes).

---

## Experiment tracking

All model runs are logged to Weights & Biases under project `23f2002523-t22026`, reporting
MAP@3, accuracy and macro F1 computed by a single shared function so that cross-run
comparison is valid.

---

## Reproducibility

Random seeds are fixed across Python, NumPy and PyTorch (CPU and CUDA). The MAP@3
implementation is unit-tested against a hand-computed case.

The winning configuration scored 0.75644; a later re-run of the identical notebook produced
0.75602. The difference of 0.0004 (~0.2 test rows) arises from GPU-level non-determinism
during retraining, and sits well inside the noise band described above.


