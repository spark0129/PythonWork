"""Elastic Net rating-prediction baseline for the Netflix Prize Kaggle data.

Example (run from the repository root):
python outputs/netflix_elastic_net.py --ratings data/Netflix_Dataset_Rating.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def canonicalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept common variants such as User_ID, userId, or user id."""
    cleaned = {c: "".join(ch for ch in c.lower() if ch.isalnum()) for c in frame.columns}
    inverse = {v: k for k, v in cleaned.items()}
    aliases = {
        "userid": "user_id", "customerid": "user_id",
        "movieid": "movie_id", "rating": "rating", "date": "date",
    }
    rename = {inverse[source]: target for source, target in aliases.items() if source in inverse}
    frame = frame.rename(columns=rename)
    required = {"user_id", "movie_id", "rating", "date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Ratings file must contain {sorted(required)}; missing {sorted(missing)}.")
    return frame[list(required)]


def reservoir_sample(path: Path, n: int, seed: int) -> pd.DataFrame:
    """Uniform sample without loading the full (very large) CSV into memory."""
    rng = np.random.default_rng(seed)
    kept = pd.DataFrame()
    for chunk in pd.read_csv(path, chunksize=250_000):
        chunk = canonicalise_columns(chunk).dropna()
        chunk["_priority"] = rng.random(len(chunk))
        kept = pd.concat([kept, chunk], ignore_index=True).nsmallest(n, "_priority")
    return kept.drop(columns="_priority").reset_index(drop=True)


def prepare(frame: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna().query("1 <= rating <= 5").copy()
    # Repeat once because removing infrequent movies can make a user infrequent, and vice versa.
    for key in ("user_id", "movie_id", "user_id", "movie_id"):
        frame = frame[frame.groupby(key)[key].transform("size") >= min_interactions]
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["day_of_week"] = frame["date"].dt.dayofweek
    return frame.drop(columns="date")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True, help="Path to Netflix_Dataset_Rating.csv")
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/netflix_elasticnet_results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ratings = prepare(reservoir_sample(args.ratings, args.sample_size, args.seed), args.min_interactions)
    if len(ratings) < 1_000:
        raise ValueError("Too few usable ratings; increase --sample-size or lower --min-interactions.")

    train, test = train_test_split(ratings, test_size=0.2, random_state=args.seed)
    features = ["user_id", "movie_id", "year", "month", "day_of_week"]
    X_train, X_test = train[features], test[features]
    y_train, y_test = train["rating"], test["rating"]

    preprocessor = ColumnTransformer([
        ("ids", OneHotEncoder(handle_unknown="ignore"), ["user_id", "movie_id"]),
        ("time", StandardScaler(), ["year", "month", "day_of_week"]),
    ])
    cv = KFold(n_splits=10, shuffle=True, random_state=args.seed)
    model = Pipeline([
        ("features", preprocessor),
        ("elastic_net", ElasticNetCV(
            l1_ratio=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
            alphas=np.logspace(-4, 0, 50), cv=cv, max_iter=20_000,
            n_jobs=-1, random_state=args.seed,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = np.clip(model.predict(X_test), 1, 5)
    baseline = np.repeat(y_train.mean(), len(y_test))
    estimator = model.named_steps["elastic_net"]
    report = {
        "sample_after_filtering": int(len(ratings)),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "elastic_net_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "elastic_net_mae": float(mean_absolute_error(y_test, predictions)),
        "global_mean_rmse": float(np.sqrt(mean_squared_error(y_test, baseline))),
        "alpha": float(estimator.alpha_), "l1_ratio": float(estimator.l1_ratio_),
        "nonzero_coefficients": int(np.count_nonzero(np.abs(estimator.coef_) > 1e-10)),
        "total_expanded_features": int(len(estimator.coef_)),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    chosen_ratio_index = int(np.flatnonzero(np.isclose(estimator.l1_ratio, estimator.l1_ratio_))[0])
    mean_mse = estimator.mse_path_[chosen_ratio_index].mean(axis=-1)
    plt.figure(figsize=(7, 4.5))
    plt.semilogx(estimator.alphas_, mean_mse, marker="o", ms=2)
    plt.axvline(estimator.alpha_, color="crimson", ls="--", label=f"chosen alpha={estimator.alpha_:.4g}")
    plt.gca().invert_xaxis(); plt.xlabel("alpha (larger = stronger penalty)")
    plt.ylabel("10-fold CV MSE"); plt.title("Elastic Net cross-validation curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(args.output_dir / "cv_curve.png", dpi=180)


if __name__ == "__main__":
    main()
