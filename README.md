# Medical LLM Paper Screening

A lightweight classifier for screening biomedical/clinical AI papers for
inclusion in a systematic review — and a comparison against an LLM
(K2-Think-V2) doing the same task.

**Bottom line:** a TF-IDF + logistic regression classifier — cheap, fast,
no GPU required — turned out to be statistically indistinguishable from
an LLM on this screening task (McNemar's test, p = 0.94), after a domain-
specific transformer embedding model (SPECTER2) was tried first and
decisively lost to the LLM (p < 0.0001). The investigation into *why*, and
what closed the gap, is the more interesting part — see [Findings](#findings)
below.

## Results

5-fold cross-validation on ~510 labeled papers (title + abstract → include/exclude):

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **TF-IDF + LogisticRegression** (ships as `classifier.pkl`) | **0.743** | **0.743** | **0.807** |
| SPECTER2 embeddings + LogisticRegression | 0.597 | 0.586 | 0.668 |
| K2-Think-V2 (LLM) | 0.747 | 0.78 (include) | — |

**Paired comparison against K2-Think-V2 (McNemar's test, n=506):**

| Comparison | p-value | Verdict |
|---|---|---|
| SPECTER2 vs. K2-Think-V2 | < 0.0001 | K2 significantly better |
| TF-IDF vs. K2-Think-V2 | 0.940 | Statistically indistinguishable |

K2 still holds a real recall edge on "include" (0.90 vs. TF-IDF's 0.77) —
it misses fewer true-relevant papers — but on overall accuracy, the gap
to a linear bag-of-words classifier is not statistically detectable at
this sample size.

## Findings

The path from "SPECTER2 loses badly to an LLM" to "TF-IDF ties an LLM" went
through a few wrong hypotheses first: **SPECTER2 losing to plain TF-IDF**
was the red flag that started the investigation, a **title-embedding
concatenation bug** turned out to be quietly hurting the embedding model,
and a **counterintuitive TF-IDF fix** (keeping stopwords instead of
removing them) closed most of the remaining gap to K2-Think-V2 — confirmed
with McNemar's paired test rather than eyeballed accuracy deltas.

<details>
<summary>Full investigation trail (7 steps)</summary>

1. **Started with SPECTER2** (a transformer pretrained specifically on
   scientific papers, with a classification adapter attached) — assumed
   it would beat a plain bag-of-words baseline. It didn't: TF-IDF beat it
   on every metric, and SPECTER2 lost to K2-Think-V2 decisively (p<0.0001).
2. **Tested and ruled out feature scaling** as the fix — adding
   `StandardScaler` before the classifier didn't help (it made things
   marginally worse), despite being the most likely-looking cause.
3. **Chased and ruled out a red herring** — a library warning suggested
   the adapter wasn't actually active during embedding. Verified directly
   (forward-pass output differs with the adapter on vs. off) — it was a
   stale log message, not a real bug.
4. **Found the real issue**: concatenating title + abstract embeddings
   (intended to give the model more structure) was actively hurting —
   titles are short, so their embedding is mostly noise. Abstract-only
   embeddings recovered some of the gap, but not enough.
5. **Tried combining TF-IDF + SPECTER2** (both early fusion via raw
   feature concatenation, and late fusion via a stacked meta-classifier).
   Neither beat TF-IDF alone — a clean negative result showing SPECTER2
   doesn't carry complementary signal for this task.
6. **Found a real, counterintuitive TF-IDF improvement**: *keeping*
   English stopwords (rather than removing them, the usual default) beat
   removing them by +2.4pt accuracy — short words like "not" or "without"
   apparently carry real screening signal here.
7. **Validated every comparison with McNemar's paired test**, not just
   eyeballed accuracy deltas — this is what caught that the SPECTER2-vs-K2
   gap was real (p<0.0001) while the TF-IDF-vs-K2 gap was not (p=0.94).

The full investigation, including the ablations, hyperparameter sweeps,
and error analysis of exactly which papers TF-IDF gets wrong that K2
doesn't, is preserved as documented methodology in `train.py`.

</details>

## Repo structure

```
train.py           # full training + evaluation pipeline (see docstring for the 6-step breakdown)
predict.py          # score new, unlabeled papers with the saved model
requirements.txt
```

Note: the labeled dataset (`answer.csv`) and K2-Think-V2's predictions
(`k2result.csv`) are not included in this repo, since they're the lab's
in-progress review data. The expected schema is documented below so you
can run this against your own labeled data.

## Data schema

`train.py` expects `answer.csv` with columns `pmid, title, abstract, Exclude`,
and (for the optional K2-Think-V2 comparison in step 6) `k2result.csv` with
`pmid, title, abstract, llm_exclude`. Full column semantics are documented
in `train.py`'s header comments.

## Usage

```bash
pip install -r requirements.txt

# Train (produces classifier.pkl + cv_predictions.csv)
python train.py

# Score new, unlabeled papers
python predict.py --input new_papers.csv --output predictions.csv
```

`new_papers.csv` needs columns: `pmid, title, abstract`. `predict.py` only
needs `pandas`, `scikit-learn`, and `joblib` — no GPU or transformer
dependencies, since the shipped model is TF-IDF-based.

## Tech stack

`scikit-learn` (TF-IDF, LogisticRegression, cross-validation) ·
`pandas` · `PyTorch` + HuggingFace `transformers`/`adapters` (SPECTER2
comparison only) · `statsmodels` (McNemar's test)

