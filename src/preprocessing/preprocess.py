import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
import pandas as pd


def preprocess(path="data/raw/loan.csv",
               output_path="data/processed/loan_preprocessed.csv"):

    data = pd.read_csv(path)

    X_raw = data.drop("loan_status", axis=1)
    y_raw = data["loan_status"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

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

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    X_scaled_df["loan_status"] = y

    X_scaled_df.to_csv(output_path, index=False)

    return X_scaled_df, output_path

preprocess()