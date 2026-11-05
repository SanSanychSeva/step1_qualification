#!/usr/bin/env python3
"""Deliberately naive starter baseline."""
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
train = pd.read_csv(ROOT / "train.csv")
test = pd.read_csv(ROOT / "test.csv")

features = [c for c in test.columns if c != "row_id"]
cat_features = ["entity_id", "group_id"] + [c for c in features if c.startswith("cat_")]

for frame in (train, test):
    for col in cat_features:
        frame[col] = frame[col].astype(str)

train_idx, valid_idx = train_test_split(
    range(len(train)),
    test_size=0.25,
    random_state=42,
    stratify=train["target"],
)

model = CatBoostClassifier(
    iterations=140,
    depth=7,
    learning_rate=0.11,
    loss_function="Logloss",
    eval_metric="AUC",
    l2_leaf_reg=6,
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)
model.fit(
    train.loc[train_idx, features],
    train.loc[train_idx, "target"],
    cat_features=[features.index(c) for c in cat_features],
)
valid_prediction = model.predict_proba(train.loc[valid_idx, features])[:, 1]
print(f"Random-split ROC-AUC: {roc_auc_score(train.loc[valid_idx, 'target'], valid_prediction):.6f}")

model.fit(
    train[features],
    train["target"],
    cat_features=[features.index(c) for c in cat_features],
)
test_prediction = model.predict_proba(test[features])[:, 1]
pd.DataFrame({"row_id": test["row_id"], "prediction": test_prediction}).to_csv(
    ROOT / "submission.csv", index=False
)
print("Saved submission.csv")
