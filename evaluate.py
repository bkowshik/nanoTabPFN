"""Evaluate a pretrained checkpoint across train-set sizes.

The pretraining prior contains tables of at most 150 rows, so evaluating with
the full train split (2702 rows for phoneme) asks the model to extrapolate far
beyond its pretraining regime. This script subsamples the train side to
increasing sizes, fitting the baselines on the same subsample, to show where
nanoTabPFN is relatively strongest along that curve.

Usage: python evaluate.py [checkpoint.pt]
"""
import sys

import numpy as np
import torch
from baseline import models as baseline_models
from model import NanoTabPFNClassifier, NanoTabPFNModel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from train import datasets, get_default_device

TRAIN_SIZES = [150, 500, 1000, None]  # None = the full train split


def score(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    prob = classifier.predict_proba(X_test)
    pred = prob.argmax(axis=1)
    if prob.shape[1] == 2:
        prob = prob[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, prob, multi_class="ovr")),
        "acc": float(accuracy_score(y_test, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_test, pred)),
    }


if __name__ == "__main__":
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
    device = get_default_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = NanoTabPFNModel(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    classifiers = {"nanoTabPFN": NanoTabPFNClassifier(model, device), **baseline_models}
    rng = np.random.default_rng(0)
    for X_train, X_test, y_train, y_test in datasets:
        for size in TRAIN_SIZES:
            if size is None or size >= len(X_train):
                X_sub, y_sub = X_train, y_train
            else:
                idx = rng.choice(len(X_train), size=size, replace=False)
                X_sub, y_sub = X_train[idx], y_train[idx]
            print(f"\ntrain rows = {len(X_sub)} (test rows = {len(X_test)})")
            for name, classifier in classifiers.items():
                scores = score(classifier, X_sub, X_test, y_sub, y_test)
                score_str = " | ".join(f"{k} {v:7.4f}" for k, v in scores.items())
                print(f"  {name:22s} {score_str}")
