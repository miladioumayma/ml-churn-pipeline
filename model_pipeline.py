import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname", "Geography"]
TARGET = "Exited"
FEATURE_COLUMNS = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def prepare_data(csv_path, test_size=0.2, random_state=1):
    """Charge le CSV, nettoie les données et retourne (x_train, x_test, y_train, y_test)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    if TARGET not in df.columns:
        raise ValueError(f"La colonne cible '{TARGET}' est absente du fichier.")

    n_missing = int(df.isna().sum().sum())
    n_dup = int(df.duplicated().sum())
    print(
        f"[prepare_data] {df.shape[0]} lignes, {df.shape[1]} colonnes | "
        f"valeurs manquantes : {n_missing} | doublons : {n_dup}"
    )
    df = df.drop_duplicates().dropna()

    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])  # Female=0, Male=1
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    print(f"[prepare_data] train : {x_train.shape} | test : {x_test.shape}")
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier et retourne le modèle."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    print(f"[train_model] RandomForest entraîné ({n_estimators} arbres).")
    return model


def evaluate_model(model, x_test, y_test):
    """Évalue le modèle et retourne un dict (accuracy, matrice de confusion, rapport)."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"[evaluate_model] Accuracy : {accuracy * 100:.2f} %")
    print("Matrice de confusion :\n", matrix)
    print("Rapport de classification :\n", report)
    return {"accuracy": accuracy, "confusion_matrix": matrix, "report": report}


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, path)
    print(f"[save_model] Modèle sauvegardé dans '{path}'.")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modèle introuvable : {path}. Lancez d'abord l'entraînement."
        )
    model = joblib.load(path)
    print(f"[load_model] Modèle chargé depuis '{path}'.")
    return model


def predict_customer(model, features):
    """Prédit le churn d'un client. `features` suit l'ordre de FEATURE_COLUMNS."""
    if len(features) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"{len(FEATURE_COLUMNS)} valeurs attendues : {FEATURE_COLUMNS}"
        )
    df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(df)[0])
    print(
        f"[predict_customer] Prédiction : {prediction} "
        f"({'client perdu' if prediction else 'client conservé'})"
    )
    return prediction
