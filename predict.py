"""
Score new, unlabeled papers using the classifier trained in train.py.

Usage:
  python predict.py --input new_papers.csv --output predictions.csv

Input CSV must have columns: pmid, title, abstract
Output CSV adds: p_include, prediction (include/exclude), used_threshold

Note: the shipped model is TF-IDF + LogisticRegression (see train.py's
docstring for why), so this script has no torch/transformers/GPU dependency
-- it just needs the same scikit-learn stack train.py uses.
"""

import argparse
import pandas as pd
import joblib

MODEL_PATH = "classifier.pkl"


def main():
    parser = argparse.ArgumentParser(description="Score new papers with the trained classifier")
    parser.add_argument("--input", required=True, help="CSV with columns: pmid, title, abstract")
    parser.add_argument("--output", default="predictions.csv", help="Where to write scored results")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to saved classifier.pkl")
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Load the trained model bundle
    # -----------------------------------------------------------------
    print(f"Loading model from {args.model}...")
    bundle = joblib.load(args.model)
    clf = bundle["classifier"]
    threshold = bundle["threshold"]
    print(f"Using threshold={threshold:.3f}")

    # -----------------------------------------------------------------
    # Load new papers
    # -----------------------------------------------------------------
    df = pd.read_csv(args.input)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    required_cols = {"pmid", "title", "abstract"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV is missing required column(s): {missing_cols}")

    n_before = len(df)
    df["abstract"] = df["abstract"].fillna("")
    df = df.dropna(subset=["title"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} row(s) with missing title")

    n_duplicates = df.duplicated(subset=["title"]).sum()
    if n_duplicates:
        print(f"Note: {n_duplicates} duplicate title(s) in input -- not dropping, "
              f"since you may want predictions for all pmids, but check these manually")

    print(f"Scoring {len(df)} new papers...")

    # -----------------------------------------------------------------
    # Predict using the saved recall-biased threshold (not the default 0.5)
    # -----------------------------------------------------------------
    # Must match train.py's text_layout exactly -- a mismatch here silently
    # produces garbage predictions.
    texts = (df["title"] + ". " + df["abstract"]).tolist()
    p_include = clf.predict_proba(texts)[:, 1]
    prediction = (p_include >= threshold).astype(int)

    df["p_include"] = p_include
    df["prediction"] = pd.Series(prediction).map({1: "include", 0: "exclude"})
    df["used_threshold"] = threshold

    df.to_csv(args.output, index=False)
    print(f"\nSaved predictions to {args.output}")
    print(f"Predicted include: {(prediction == 1).sum()} | Predicted exclude: {(prediction == 0).sum()}")

    # Flag borderline cases -- papers near the threshold are the ones most
    # worth a quick manual glance, since the model is least confident there
    borderline = df[(df["p_include"] >= threshold - 0.1) & (df["p_include"] <= threshold + 0.1)]
    if len(borderline):
        print(f"\n{len(borderline)} paper(s) within +/-0.1 of the threshold (worth a manual spot-check):")
        print(borderline[["pmid", "title", "p_include"]].sort_values("p_include").to_string(index=False))


if __name__ == "__main__":
    main()
