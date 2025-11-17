# src/eval.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(data_dir="data/processed", model_path="models/model.pkl"):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))["SalePrice"]
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))["SalePrice"]

    model = joblib.load(model_path)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"Train RMSE : {np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}")
    print(f"Test RMSE  : {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
    print(f"Train R2   : {r2_score(y_train, y_train_pred):.3f}")
    print(f"Test R2    : {r2_score(y_test, y_test_pred):.3f}")

    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title("Prédictions vs Réelles (LinearRegression)")
    os.makedirs("reports/figures", exist_ok=True)
    fig_path = "reports/figures/predictions_vs_actual.png"
    plt.savefig(fig_path)
    plt.show()
    print(f"Graphique sauvegardé → {fig_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Évaluation du modèle")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_path", type=str, default="models/model.pkl")
    args = parser.parse_args()
    evaluate_model(args.data_dir, args.model_path)
