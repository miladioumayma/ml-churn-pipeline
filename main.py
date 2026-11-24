import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    predict_customer,
    prepare_data,
    save_model,
    train_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn")
    parser.add_argument(
        "--prepare", action="store_true", help="Charger et prétraiter les données"
    )
    parser.add_argument(
        "--train", action="store_true", help="Entraîner et sauvegarder le modèle"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Évaluer le modèle sauvegardé"
    )
    parser.add_argument(
        "--predict",
        nargs=9,
        type=float,
        metavar="V",
        help="CreditScore Gender Age Tenure Balance "
        "NumOfProducts HasCrCard IsActiveMember "
        "EstimatedSalary",
    )
    parser.add_argument("--all", action="store_true", help="Exécuter tout le pipeline")
    parser.add_argument("--data", default=DATA_PATH, help="Chemin du CSV")
    parser.add_argument("--model", default=MODEL_PATH, help="Chemin du modèle")
    return parser.parse_args()


def main():
    args = parse_args()

    if not any([args.prepare, args.train, args.evaluate, args.predict, args.all]):
        print("Aucune action demandée. Utilisez --help.")
        return

    if args.prepare or args.all:
        prepare_data(args.data)

    if args.train or args.all:
        x_train, x_test, y_train, y_test = prepare_data(args.data)
        model = train_model(x_train, y_train)
        save_model(model, args.model)

    if args.evaluate or args.all:
        _, x_test, _, y_test = prepare_data(args.data)
        model = load_model(args.model)
        evaluate_model(model, x_test, y_test)

    if args.predict:
        model = load_model(args.model)
        predict_customer(model, args.predict)


if __name__ == "__main__":
    main()
