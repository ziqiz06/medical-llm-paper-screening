"""
Lightweight include/exclude classifier for YLab paper filtering.

Pipeline:
  1. Load labeled papers (title, abstract, label), dedup by title
  2. Train a TF-IDF + logistic regression classifier via cross-validation.
     stop_words=None was picked via a hyperparameter sweep -- keeping
     stopwords beat removing them here (see comment at the Pipeline def).
     This is the model that ships (step 5): it beats SPECTER2 on every
     metric and is statistically indistinguishable from K2-Think-V2 on
     accuracy (step 6b, p=0.94), despite trailing on recall.
     2b picks a recall-biased decision threshold for screening use.
  3. Embed title and abstract separately using SPECTER2 (scientific-paper
     embedding model) with its "classification" adapter attached -- kept as
     a documented comparison (step 4), not because it's used in production
  4. Train the SPECTER2 classifier via cross-validation, reporting both
     pooled metrics and per-fold mean +/- std (more honest at n=500), plus
     its own recall-biased threshold, an error analysis (which papers are
     misclassified), and a cv_predictions.csv export -- SPECTER2 loses to
     K2-Think-V2 decisively (step 6a, p<0.0001), which is why TF-IDF ships
     instead
  5. Train + save the final TF-IDF model on all data (see predict.py to
     score new, unlabeled papers with it -- no torch/transformers needed)
  6. Compare against K2-Think-V2's predictions on the same papers (by pmid):
     6a. SPECTER2 classifier vs K2-Think-V2, with McNemar's test
     6b. TF-IDF classifier vs K2-Think-V2, with McNemar's test and the list
         of papers K2 got right that TF-IDF got wrong (the most useful
         place to look for a fixable pattern)

Install requirements first:
  pip install transformers adapters torch scikit-learn pandas statsmodels --break-system-packages
"""

import pandas as pd
import numpy as np
import torch
import joblib
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
from statsmodels.stats.contingency_tables import mcnemar

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
# CSV columns: pmid, title, abstract, Exclude (True/False). Exclude=True
# means the paper should be excluded, so label (1=include, 0=exclude) is the
# inverse of it. pmid is kept (not used for training) so predictions can be
# joined back to K2-Think-V2's output later (step 6).

DATA_PATH = "answer.csv"

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lstrip("﻿") for c in df.columns]  # strip stray BOM/whitespace
df = df[["pmid", "title", "abstract", "Exclude"]]  # drop empty trailing spreadsheet columns

df["label"] = (~df["Exclude"].astype(bool)).astype(int)  # include=1, exclude=0

# Basic sanity checks -- don't skip this, it catches most early bugs
print(f"Loaded {len(df)} papers")
print(f"Label distribution:\n{df['label'].value_counts()}")
assert df["title"].isna().sum() == 0, "Found missing titles"
n_missing_abstract = df["abstract"].isna().sum()
if n_missing_abstract:
    print(f"Filling {n_missing_abstract} row(s) with missing abstract (title-only)")
    df["abstract"] = df["abstract"].fillna("")

n_duplicates = df.duplicated(subset=["title"]).sum()
if n_duplicates:
    print(f"Dropping {n_duplicates} duplicate row(s) by title (avoids CV leakage)")
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

y = df["label"].astype(int).values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def run_cv(estimator, X, y, cv):
    """Fit `estimator` once per CV fold, collecting per-fold scores and pooled
    out-of-fold probabilities in a single pass -- avoids the double-fit you'd
    get from calling cross_validate (for scores) and cross_val_predict (for
    predictions) separately on the same data."""
    fold_scores = {"accuracy": [], "f1": [], "roc_auc": [], "average_precision": []}
    y_proba = np.empty(len(y))
    for train_idx, test_idx in cv.split(X, y):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        y_proba[test_idx] = proba

        fold_scores["accuracy"].append(accuracy_score(y[test_idx], pred))
        fold_scores["f1"].append(f1_score(y[test_idx], pred))
        fold_scores["roc_auc"].append(roc_auc_score(y[test_idx], proba))
        fold_scores["average_precision"].append(average_precision_score(y[test_idx], proba))

    fold_scores = {k: np.array(v) for k, v in fold_scores.items()}
    y_pred = (y_proba >= 0.5).astype(int)
    return fold_scores, y_proba, y_pred


# ---------------------------------------------------------------------------
# 2. BASELINE: TF-IDF + LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
# This tells you whether SPECTER2's domain-specific embeddings are actually
# earning their complexity, rather than assuming a fancier model = better.
# Wrapping TfidfVectorizer in the Pipeline means it's refit inside each CV
# fold (via run_cv's clone+fit), so there's no train/test leakage from
# vocabulary fitting.
#
# stop_words=None (i.e. NOT removing English stopwords) was picked via a
# hyperparameter sweep -- counterintuitively, keeping them beat removing
# them by +2.4pt accuracy / +1.9pt F1 (0.719->0.743, 0.727->0.746), likely
# because short words like "not"/"without" carry real screening signal here.
# Combining with min_df=2 or sublinear_tf on top of that made it worse
# again (tested), so this is deliberately the single change, not a bundle.
print("\n=== Baseline: TF-IDF + Logistic Regression ===")
baseline_texts = np.array((df["title"] + ". " + df["abstract"]).tolist(), dtype=object)

baseline_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words=None, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

baseline_scores, baseline_proba, baseline_pred = run_cv(baseline_pipeline, baseline_texts, y, cv)
print(f"Accuracy:  {baseline_scores['accuracy'].mean():.3f} +/- {baseline_scores['accuracy'].std():.3f}")
print(f"F1:        {baseline_scores['f1'].mean():.3f} +/- {baseline_scores['f1'].std():.3f}")
print(f"ROC-AUC:   {baseline_scores['roc_auc'].mean():.3f} +/- {baseline_scores['roc_auc'].std():.3f}")
print(f"PR-AUC:    {baseline_scores['average_precision'].mean():.3f} +/- {baseline_scores['average_precision'].std():.3f}")

print("\nBaseline confusion matrix:")
print(confusion_matrix(y, baseline_pred))

# ---------------------------------------------------------------------------
# 2b. PICK A RECALL-BIASED THRESHOLD FOR TF-IDF
# ---------------------------------------------------------------------------
# Same reasoning as SPECTER2's threshold selection below (step 4b): for
# screening, missing a true include is worse than over-flagging an exclude.
# TF-IDF is the model that actually ships (see step 5), so this is the
# threshold predict.py uses.
TARGET_RECALL = 0.95
tfidf_precision, tfidf_recall, tfidf_thresholds = precision_recall_curve(y, baseline_proba)
tfidf_above_target = tfidf_recall[:-1] >= TARGET_RECALL
if tfidf_above_target.any():
    tfidf_threshold = tfidf_thresholds[tfidf_above_target].max()
else:
    tfidf_threshold = 0.0
    print(f"\nWarning: no threshold reaches {TARGET_RECALL:.0%} recall -- using 0.0 (flags everything)")

tfidf_pred_recall = (baseline_proba >= tfidf_threshold).astype(int)
print(f"\n=== TF-IDF Performance (threshold={tfidf_threshold:.3f} for >= {TARGET_RECALL:.0%} recall) ===")
print(f"Accuracy: {accuracy_score(y, tfidf_pred_recall):.3f}")
print(classification_report(y, tfidf_pred_recall, target_names=["exclude", "include"]))

# Attach TF-IDF predictions to df (same row order as y) so they can be
# joined to K2-Think-V2's output by pmid in step 6.
df["tfidf_proba"] = baseline_proba
df["tfidf_pred"] = baseline_pred

# ---------------------------------------------------------------------------
# 3. GENERATE SPECTER2 EMBEDDINGS
# ---------------------------------------------------------------------------
# SPECTER2 is a scientific-paper embedding model (base model frozen, trained
# generally; small task-specific adapters specialize it). We attach the
# "classification" adapter since that best matches our include/exclude task.
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"\nLoading embedding model on {DEVICE} (first run downloads the model)...")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
embedder = AutoAdapterModel.from_pretrained("allenai/specter2_base")
embedder.load_adapter(
    "allenai/specter2_classification", source="hf", load_as="classification", set_active=True
)
embedder.to(DEVICE)
embedder.eval()

BATCH_SIZE = 16


def embed_texts(texts, desc=""):
    """Embed a list of strings with SPECTER2, returning one [CLS] vector per string."""
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt",
                return_token_type_ids=False, max_length=512,
            ).to(DEVICE)
            output = embedder(**inputs)
            embeddings.append(output.last_hidden_state[:, 0, :].cpu())  # [CLS] token
            print(f"  [{desc}] {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    return torch.cat(embeddings, dim=0)


# Embed title and abstract separately (rather than joining into one string) so
# the classifier gets each piece's own representation instead of one blended
# vector -- gives Logistic Regression more structured signal to work with.
print("Generating embeddings...")
title_embeddings = embed_texts(df["title"].tolist(), desc="title")
abstract_embeddings = embed_texts(df["abstract"].tolist(), desc="abstract")

X = torch.cat([title_embeddings, abstract_embeddings], dim=1).numpy()

# ---------------------------------------------------------------------------
# 4. TRAIN + EVALUATE SPECTER2 CLASSIFIER VIA CROSS-VALIDATION
# ---------------------------------------------------------------------------
# With only 500 examples, per-fold mean +/- std is more honest than a single
# pooled number -- it tells you how much performance actually varies by
# which 400 papers happened to be in the training fold.
#
# Unlike TfidfVectorizer (which L2-normalizes every row by default), raw
# SPECTER2 [CLS] embeddings have no built-in scale discipline -- without
# StandardScaler, LogisticRegression's L2 penalty hits differently-scaled
# dimensions unevenly. Refit inside each CV fold (via run_cv's clone+fit),
# same leakage-avoidance reasoning as the baseline's Pipeline.
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

print("\n=== SPECTER2 + Logistic Regression (5-fold CV) ===")
specter_scores, y_proba, y_pred_default = run_cv(clf, X, y, cv)
print(f"Accuracy:  {specter_scores['accuracy'].mean():.3f} +/- {specter_scores['accuracy'].std():.3f}")
print(f"F1:        {specter_scores['f1'].mean():.3f} +/- {specter_scores['f1'].std():.3f}")
print(f"ROC-AUC:   {specter_scores['roc_auc'].mean():.3f} +/- {specter_scores['roc_auc'].std():.3f}")
print(f"PR-AUC:    {specter_scores['average_precision'].mean():.3f} +/- {specter_scores['average_precision'].std():.3f}")

print("\n--- vs. TF-IDF baseline ---")
print(f"F1 delta:      {specter_scores['f1'].mean() - baseline_scores['f1'].mean():+.3f}")
print(f"ROC-AUC delta: {specter_scores['roc_auc'].mean() - baseline_scores['roc_auc'].mean():+.3f}")

# y_proba/y_pred_default above are the pooled out-of-fold predictions run_cv
# already collected while scoring -- used below for the confusion matrix,
# threshold selection, and the K2-Think-V2 comparison.
print("\nPooled confusion matrix (default 0.5 threshold):")
print(confusion_matrix(y, y_pred_default))
print(classification_report(y, y_pred_default, target_names=["exclude", "include"]))

# ---------------------------------------------------------------------------
# 4b. PICK A RECALL-BIASED THRESHOLD
# ---------------------------------------------------------------------------
# For screening, missing a paper that should be included (false negative) is
# usually worse than sending an extra paper to human review (false positive).
# Find the loosest threshold that still hits our target recall on "include".
# (TARGET_RECALL is already set in step 2b, reused here for consistency.)
precision, recall, thresholds = precision_recall_curve(y, y_proba)
above_target = recall[:-1] >= TARGET_RECALL  # last point has no threshold, drop it
if above_target.any():
    threshold = thresholds[above_target].max()
else:
    threshold = 0.0
    print(f"\nWarning: no threshold reaches {TARGET_RECALL:.0%} recall -- using 0.0 (flags everything)")

y_pred_recall = (y_proba >= threshold).astype(int)
print(f"\n=== SPECTER2 Performance (threshold={threshold:.3f} for >= {TARGET_RECALL:.0%} recall) ===")
print(f"Accuracy: {accuracy_score(y, y_pred_recall):.3f}")
print(classification_report(y, y_pred_recall, target_names=["exclude", "include"]))
print("Confusion matrix:")
print(confusion_matrix(y, y_pred_recall))

# Attach predictions to df (same row order as X/y) so they can be joined to
# K2-Think-V2's output by pmid in step 6.
df["y_proba"] = y_proba
df["pred_default"] = y_pred_default
df["pred_recall"] = y_pred_recall

# ---------------------------------------------------------------------------
# 4c. ERROR ANALYSIS -- which papers does the model get wrong?
# ---------------------------------------------------------------------------
# Aggregate metrics don't say WHICH papers are misclassified. Cheap way to
# catch genuine model weakness vs. mislabeled rows in answer.csv.
df["error_type"] = np.select(
    [
        (df["pred_recall"] == 1) & (df["label"] == 0),
        (df["pred_recall"] == 0) & (df["label"] == 1),
    ],
    ["false_positive", "false_negative"],
    default="correct",
)

false_positives = df[df["error_type"] == "false_positive"].sort_values("y_proba", ascending=False)
false_negatives = df[df["error_type"] == "false_negative"].sort_values("y_proba")
print(f"\n=== Error Analysis (threshold={threshold:.3f}) ===")
print(f"False positives (predicted include, actually exclude): {len(false_positives)}")
if len(false_positives):
    print(false_positives[["pmid", "title", "y_proba"]].head(10).to_string(index=False))
print(f"\nFalse negatives (predicted exclude, actually include): {len(false_negatives)}")
if len(false_negatives):
    print(false_negatives[["pmid", "title", "y_proba"]].head(10).to_string(index=False))

CV_PREDICTIONS_PATH = "cv_predictions.csv"
df.to_csv(CV_PREDICTIONS_PATH, index=False)
print(f"\nSaved out-of-fold predictions + error labels to {CV_PREDICTIONS_PATH}")

# ---------------------------------------------------------------------------
# 5. TRAIN + SAVE THE FINAL MODEL ON ALL DATA (for actual use going forward)
# ---------------------------------------------------------------------------
# TF-IDF ships as the production model, not SPECTER2 -- it wins on every CV
# metric (step 2 vs step 4) and is statistically indistinguishable from
# K2-Think-V2 (step 6b, p=0.94), while SPECTER2 loses to K2 decisively
# (step 6a, p<0.0001). The SPECTER2 run above is kept as documented
# methodology (what was tried, why it didn't help), not as a candidate for
# production. TF-IDF is also far cheaper to ship: predict.py needs no
# torch/transformers/GPU, just the fitted vectorizer + classifier below.
final_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words=None, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
final_pipeline.fit(baseline_texts, y)
print("\nFinal TF-IDF model trained on all labeled data -- ready to classify new papers.")

MODEL_OUT_PATH = "classifier.pkl"
joblib.dump(
    {
        "classifier": final_pipeline,
        "threshold": tfidf_threshold,  # use this instead of 0.5 when predicting -- see step 2b
        "text_layout": "title + '. ' + abstract",  # how predict.py must build input text
    },
    MODEL_OUT_PATH,
)
print(f"Saved trained classifier + metadata to {MODEL_OUT_PATH}")

# Example: classifying a new paper
# new_text = ["Some New Paper Title" + ". " + "Some new abstract text..."]
# new_proba = final_pipeline.predict_proba(new_text)[:, 1]
# prediction = int(new_proba[0] >= tfidf_threshold)
# print(f"Predicted label: {'include' if prediction == 1 else 'exclude'} (P={new_proba[0]:.3f})")

# ---------------------------------------------------------------------------
# 6. COMPARE AGAINST K2-THINK-V2 PREDICTIONS
# ---------------------------------------------------------------------------
# k2result.csv holds K2-Think-V2's own include/exclude call on (mostly) the
# same papers, keyed by pmid. llm_exclude follows the same True/False
# convention as our own Exclude column, so it's flipped into k2_pred the
# same way label was derived in step 1.
K2_PATH = "k2result.csv"

k2_df = pd.read_csv(K2_PATH)
k2_df.columns = [c.strip().lstrip("﻿") for c in k2_df.columns]
k2_df["k2_pred"] = (~k2_df["llm_exclude"].astype(bool)).astype(int)  # include=1, exclude=0
k2_df = k2_df[["pmid", "k2_pred"]]

merged = df.merge(k2_df, on="pmid", how="inner")
print(f"\n=== K2-Think-V2 Comparison ===")
print(f"Matched {len(merged)}/{len(df)} papers by pmid ({K2_PATH} has {len(k2_df)} rows)")

print("\nK2-Think-V2 vs ground truth:")
print(classification_report(merged["label"], merged["k2_pred"], target_names=["exclude", "include"]))
print("Confusion matrix:")
print(confusion_matrix(merged["label"], merged["k2_pred"]))

# --- 6a. SPECTER2 vs K2-Think-V2 -----------------------------------------
# Use pred_default (0.5 threshold), not pred_recall, for this comparison.
# pred_recall was deliberately tuned to an extreme, recall-maximizing
# operating point (TARGET_RECALL=0.95) -- comparing that against K2's own
# balanced decision would stack the deck against our classifier rather than
# asking "which model is actually better." pred_default is the fair,
# apples-to-apples operating point for both agreement and McNemar's test.
agreement = (merged["pred_default"] == merged["k2_pred"]).mean()
print(f"\nSPECTER2 classifier (default 0.5 threshold) vs K2-Think-V2 agreement: {agreement:.1%}")

disagreements = merged[merged["pred_default"] != merged["k2_pred"]]
print(f"Papers where they disagree: {len(disagreements)}")
if len(disagreements):
    print(disagreements[["pmid", "title", "label", "pred_default", "k2_pred"]].head(10).to_string(index=False))

# --- McNemar's test: is our classifier's error rate significantly
# different from K2-Think-V2's, on the SAME papers? ---------------------
# This matters because raw accuracy comparison ignores that both models are
# being tested on identical items -- McNemar's test uses only the papers
# where the two models *disagree*, checking whether one is wrong more often
# than the other in those disagreement cases.
our_correct = (merged["pred_default"] == merged["label"])
k2_correct = (merged["k2_pred"] == merged["label"])

# 2x2 contingency table of (our model correct/incorrect) x (K2 correct/incorrect)
both_correct = ((our_correct) & (k2_correct)).sum()
only_ours_correct = ((our_correct) & (~k2_correct)).sum()
only_k2_correct = ((~our_correct) & (k2_correct)).sum()
both_incorrect = ((~our_correct) & (~k2_correct)).sum()

table = [[both_correct, only_ours_correct], [only_k2_correct, both_incorrect]]
result = mcnemar(table, exact=True)

print(f"\n=== McNemar's Test: SPECTER2 classifier vs. K2-Think-V2 ===")
print(f"Both correct: {both_correct} | Only ours correct: {only_ours_correct} | "
      f"Only K2 correct: {only_k2_correct} | Both incorrect: {both_incorrect}")
print(f"p-value: {result.pvalue:.4f}")
if result.pvalue < 0.05:
    winner = "our classifier" if only_ours_correct > only_k2_correct else "K2-Think-V2"
    print(f"Statistically significant difference (p<0.05) -- {winner} performs better on this sample.")
else:
    print("No statistically significant difference in error rates at p<0.05.")

# --- 6b. TF-IDF vs K2-Think-V2 --------------------------------------------
# TF-IDF beat SPECTER2 in every metric above (see step 2 vs step 4), so this
# is the more relevant "how close are we to K2" comparison in practice.
tfidf_agreement = (merged["tfidf_pred"] == merged["k2_pred"]).mean()
print(f"\nTF-IDF classifier (default 0.5 threshold) vs K2-Think-V2 agreement: {tfidf_agreement:.1%}")

tfidf_disagreements = merged[merged["tfidf_pred"] != merged["k2_pred"]]
print(f"Papers where they disagree: {len(tfidf_disagreements)}")

tfidf_correct = (merged["tfidf_pred"] == merged["label"])
k2_correct_b = (merged["k2_pred"] == merged["label"])

# The papers that matter most for "closing the gap": K2 got these right and
# TF-IDF got them wrong. Sorted by tfidf_proba descending -- these are the
# ones TF-IDF was most CONFIDENTLY wrong about, the most informative to
# read first when looking for a fixable pattern.
only_k2_right = merged[(~tfidf_correct) & (k2_correct_b)].sort_values("tfidf_proba", ascending=False)
print(f"\nPapers where K2 was right and TF-IDF was wrong: {len(only_k2_right)}")
if len(only_k2_right):
    print(only_k2_right[["pmid", "title", "label", "tfidf_proba", "tfidf_pred", "k2_pred"]]
          .head(15).to_string(index=False))

both_correct_b = ((tfidf_correct) & (k2_correct_b)).sum()
only_tfidf_correct = ((tfidf_correct) & (~k2_correct_b)).sum()
only_k2_correct_b = ((~tfidf_correct) & (k2_correct_b)).sum()
both_incorrect_b = ((~tfidf_correct) & (~k2_correct_b)).sum()

table_b = [[both_correct_b, only_tfidf_correct], [only_k2_correct_b, both_incorrect_b]]
result_b = mcnemar(table_b, exact=True)

print(f"\n=== McNemar's Test: TF-IDF classifier vs. K2-Think-V2 ===")
print(f"Both correct: {both_correct_b} | Only TF-IDF correct: {only_tfidf_correct} | "
      f"Only K2 correct: {only_k2_correct_b} | Both incorrect: {both_incorrect_b}")
print(f"p-value: {result_b.pvalue:.4f}")
if result_b.pvalue < 0.05:
    winner_b = "TF-IDF" if only_tfidf_correct > only_k2_correct_b else "K2-Think-V2"
    print(f"Statistically significant difference (p<0.05) -- {winner_b} performs better on this sample.")
else:
    print("No statistically significant difference in error rates at p<0.05 -- "
          "the accuracy gap could plausibly be noise at this sample size.")