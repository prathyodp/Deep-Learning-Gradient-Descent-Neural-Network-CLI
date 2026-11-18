from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def make_regression_csv(path="data/raw/regression.csv", samples=500, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(samples, 3))
    y = 3.2 * X[:, 0] - 1.7 * X[:, 1] + 0.8 * X[:, 2] + 2.5
    y += rng.normal(0, noise, size=samples)
    df = pd.DataFrame(X, columns=["feature_1", "feature_2", "feature_3"])
    df["y"] = y
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

def load_data(path=None, target="y", samples=500, noise=0.15, seed=42, test_size=0.2):
    if path is None:
        path = make_regression_csv(samples=samples, noise=noise, seed=seed)
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")
    X = df.drop(columns=[target]).select_dtypes(include="number").to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32).reshape(-1, 1)
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test, df
