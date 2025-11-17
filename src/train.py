# src/train.py

import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


def train_model(data_dir="data/processed", output_dir="models"):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))["SalePrice"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Modèle entraîné et sauvegardé → {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement du modèle")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="models")
    args = parser.parse_args()
    train_model(args.data_dir, args.output_dir)
