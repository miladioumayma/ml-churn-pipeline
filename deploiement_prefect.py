
from prefect import serve
from pipeline_perfect import all_flow, code_flow, evaluate_flow, train_flow

if __name__ == "__main__":
    code_deployment = code_flow.to_deployment(
        name="ml-pipeline-code",
        cron="0 7 * * *",
        tags=["qualite", "quotidien"],
        description="Contrôle qualité du code (Black + Ruff) une fois par jour.",
    )

    train_deployment = train_flow.to_deployment(
        name="ml-pipeline-train",
        cron="0 8 * * *",
        tags=["entrainement", "quotidien"],
        description="Préparation, entraînement et sauvegarde du modèle.",
    )

    all_deployment = all_flow.to_deployment(
        name="ml-pipeline-all",
        cron="0 9 * * *",
        tags=["complet", "quotidien"],
        description="Pipeline complet : qualité, entraînement, évaluation.",
    )

    evaluate_deployment = evaluate_flow.to_deployment(
        name="ml-pipeline-evaluate",
        tags=["evaluation", "manuel"],
        description="Chargement et évaluation du modèle sauvegardé.",
    )

    print("Déploiements enregistrés. Ce processus doit rester actif (Ctrl+C pour arrêter).")

    serve(
        code_deployment,
        train_deployment,
        all_deployment,
        evaluate_deployment,
    )

