# src/data_process.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def pd4_stratified_split(X, y, test_size=0.2, random_state=42, bins=10):
    y_bins = pd.cut(y, bins=bins, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y_bins
    )
    return X_train, X_test, y_train, y_test

def prepare_data(input_path="data/train.csv", output_dir="data/processed"):
    df = pd.read_csv(input_path)
    print("Dimensions initiales :", df.shape)

    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric = df_numeric.drop(columns=["Id"], errors="ignore")   # bonne pratique
    df_numeric = df_numeric.dropna()
    print("Dimensions après nettoyage :", df_numeric.shape)

    y = df_numeric["SalePrice"]
    X = df_numeric.drop(columns=["SalePrice"])

    X_train, X_test, y_train, y_test = pd4_stratified_split(X, y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_frame().to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_frame().to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print(f"Données préparées sauvegardées dans {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Préparation des données")
    parser.add_argument("--input_path", type=str, default="data/train.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()
    prepare_data(args.input_path, args.output_dir)