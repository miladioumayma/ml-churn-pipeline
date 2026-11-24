import argparse
import subprocess  # nosec B404
import sys
from pathlib import Path

from prefect import flow, task

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

PROJECT_DIR = Path(__file__).resolve().parent

PROJECT_FILES = [
    str(PROJECT_DIR / filename)
    for filename in [
        "model_pipeline.py",
        "main.py",
        "pipeline_perfect.py",
    ]
    if (PROJECT_DIR / filename).exists()
]

REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"
TESTS_DIR = PROJECT_DIR / "tests"


# ==============================
# TASKS D'AUTOMATISATION DU CODE
# ==============================


@task(name="Installation des dépendances")
def install_dependencies_task():
    print("[task] Installation des dépendances...")

    subprocess.check_call(  # nosec B603
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS_FILE),
        ]
    )


@task(name="Formatage du code")
def format_code_task():
    print("[task] Formatage du code avec Black...")

    subprocess.check_call(  # nosec B603
        [
            sys.executable,
            "-m",
            "black",
            *PROJECT_FILES,
        ]
    )


@task(name="Qualité du code")
def code_quality_task():
    print("[task] Vérification de la qualité avec Ruff...")

    subprocess.check_call(  # nosec B603
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *PROJECT_FILES,
        ]
    )


@task(name="Sécurité du code")
def code_security_task():
    print("[task] Vérification de sécurité avec Bandit...")

    subprocess.check_call(  # nosec B603
        [
            sys.executable,
            "-m",
            "bandit",
            "-q",
            *PROJECT_FILES,
        ]
    )


@task(name="Tests unitaires")
def unit_tests_task():
    print("[task] Exécution des tests avec Pytest...")

    subprocess.check_call(  # nosec B603
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            str(TESTS_DIR),
        ]
    )


# ==============================
# TASKS MACHINE LEARNING
# ==============================

CSV_PATH = PROJECT_DIR / "Churn_Modelling.csv"


@task(name="Préparation des données")
def prepare_data_task(csv_path=CSV_PATH):
    print("[task] Préparation des données...")
    return prepare_data(str(csv_path))


@task(name="Entraînement du modèle")
def train_model_task(x_train, y_train):
    print("[task] Entraînement du modèle...")

    return train_model(x_train, y_train)


@task(name="Sauvegarde du modèle")
def save_model_task(model):
    print("[task] Sauvegarde du modèle...")

    save_model(model)


@task(name="Chargement du modèle")
def load_model_task():
    print("[task] Chargement du modèle...")

    return load_model()


@task(name="Évaluation du modèle")
def evaluate_model_task(model, x_test, y_test):
    print("[task] Évaluation du modèle...")

    return evaluate_model(model, x_test, y_test)


# ==============================
# FLOW : VERIFICATION DU CODE
# ==============================


@flow(name="code")
def code_flow():
    print("\n========== CODE QUALITY PIPELINE ==========\n")

    format_code_task()
    code_quality_task()
    code_security_task()
    unit_tests_task()

    print("\n========== CODE PIPELINE TERMINE ==========\n")


# ==============================
# FLOW : PIPELINE COMPLET
# ==============================


@flow(name="all")
def all_flow():
    print("\n========== PIPELINE COMPLET ==========\n")

    install_dependencies_task()

    code_flow()

    x_train, x_test, y_train, y_test = prepare_data_task()

    model = train_model_task(x_train, y_train)

    save_model_task(model)

    evaluate_model_task(model, x_test, y_test)

    print("\n========== PIPELINE COMPLET TERMINE ==========\n")


# ==============================
# FLOW : ENTRAINEMENT
# ==============================


@flow(name="train")
def train_flow():
    print("\n========== TRAINING PIPELINE ==========\n")

    x_train, _, y_train, _ = prepare_data_task()

    model = train_model_task(x_train, y_train)

    save_model_task(model)

    print("\n========== TRAINING TERMINE ==========\n")


# ==============================
# FLOW : EVALUATION
# ==============================


@flow(name="evaluate")
def evaluate_flow():
    print("\n========== EVALUATION PIPELINE ==========\n")

    _, x_test, _, y_test = prepare_data_task()

    model = load_model_task()

    evaluate_model_task(model, x_test, y_test)

    print("\n========== EVALUATION TERMINEE ==========\n")


# ==============================
# EXECUTION DU PIPELINE
# ==============================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Customer Churn ML Pipeline avec Prefect"
    )

    parser.add_argument(
        "--flow",
        type=str,
        choices=["all", "train", "entrainement", "evaluate", "code"],
        default="code",
        help="Flow à exécuter",
    )

    args = parser.parse_args()

    if args.flow == "all":
        all_flow()

    elif args.flow in ["train", "entrainement"]:
        train_flow()

    elif args.flow == "evaluate":
        evaluate_flow()

    elif args.flow == "code":
        code_flow()
