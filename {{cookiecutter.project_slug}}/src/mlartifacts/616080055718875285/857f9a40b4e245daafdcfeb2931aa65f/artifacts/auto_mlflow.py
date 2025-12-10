import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd


def train_loan_model_with_autolog(path="../data/processed/loan.csv"):

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("LoanModelExperiment11")

    mlflow.autolog()  

    data = pd.read_csv(path)

    X_raw = data.drop("loan_status", axis=1)
    y_raw = data["loan_status"]

    y = LabelEncoder().fit_transform(y_raw)

    categorical_cols = ['gender', 'occupation', 'marital_status']
    numerical_cols = ['age', 'income', 'credit_score']
    ordinal_cols = ["education_level"]

    df_ohe = pd.get_dummies(X_raw[categorical_cols], drop_first=True)

    ord_encoder = OrdinalEncoder(categories=[
        ['High School', "Associate's", "Bachelor's", "Master's", "PhD", "Doctoral"]
    ])
    df_ordinal = ord_encoder.fit_transform(X_raw[ordinal_cols])
    df_ordinal = pd.DataFrame(df_ordinal, columns=ordinal_cols, index=X_raw.index)

    X = pd.concat([X_raw[numerical_cols], df_ohe, df_ordinal], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = LogisticRegression()
        model.fit(X_train, y_train)

        print("Training Done!")
        print("Accuracy:", model.score(X_test, y_test))
        run_id = mlflow.active_run().info.run_id
        mlflow.log_artifact(__file__)
        mlflow.register_model(
        model_uri=f"runs:/a927e38dfbb4412eb6d333a58e5bed49/model",
        name="LoanApprovalModel11"
)


    return model, scaler, ord_encoder  


model, scaler, encoder = train_loan_model_with_autolog()
