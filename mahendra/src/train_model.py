import cookiecutter
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd


def train_loan_model_with_autolog(path="../data/processed/loan_preprocessed.csv", 
                                  use_xgboost=1):

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("LoanModelExperiment11")
    mlflow.autolog()  

    data = pd.read_csv(path)

    X = data.drop("loan_status", axis=1)
    y = data["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if use_xgboost == 1:
        model = LogisticRegression(max_iter=500)
        model_name = "LogisticRegressionModel"
    elif use_xgboost == 2:
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
        )
        model_name = "XGBoostModel"
    else:
        raise ValueError("use_xgboost must be 1 (Logistic) or 2 (XGBoost)")

    with mlflow.start_run() as run:

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"Training Done using {model_name}!")
        print("Accuracy:", accuracy)

        mlflow.log_metric("accuracy", accuracy)

        run_id = run.info.run_id

        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="LoanApprovalModel11"
        )

    return model

model = train_loan_model_with_autolog(use_xgboost=1)
