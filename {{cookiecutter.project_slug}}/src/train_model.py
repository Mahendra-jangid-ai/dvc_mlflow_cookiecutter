import cookiecutter
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd
import psutil, time, threading
from mlflow.system_metrics import enable_system_metrics_logging, set_system_metrics_sampling_interval


def log_system_metrics_background(interval=5):
    def logger():
        while True:
            mlflow.log_metric("system/cpu_percent", psutil.cpu_percent())
            mlflow.log_metric("system/memory_percent", psutil.virtual_memory().percent)
            time.sleep(interval)

    t = threading.Thread(target=logger, daemon=True)
    t.start()



def train_loan_model_with_autolog(path="data/processed/loan_preprocessed.csv", use_xgboost=1):

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("LoanModelExperiment11")

    enable_system_metrics_logging()           
    set_system_metrics_sampling_interval(5)

    mlflow.autolog()

    with mlflow.start_span("data_loading"):
        data = pd.read_csv(path)

    with mlflow.start_span("train_test_split"):
        X = data.drop("loan_status", axis=1)
        y = data["loan_status"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    if use_xgboost == 1:
        model = LogisticRegression(max_iter=500)
        model_name = "LogisticRegressionModel"
    else:
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
        )
        model_name = "XGBoostModel"

    with mlflow.start_run(log_system_metrics=True) as run:

        log_system_metrics_background()

        with mlflow.start_span("model_training"):
            model.fit(X_train, y_train)

        with mlflow.start_span("evaluation"):
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            print(f"Training Done using {model_name} — Accuracy: {accuracy}")

        with mlflow.start_span("model_registration"):
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name="LoanApprovalModel11"
            )

    return model



model = train_loan_model_with_autolog(use_xgboost={{cookiecutter.use_xgboost}})
