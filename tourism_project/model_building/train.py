import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Bypass MLflow v3 file store restriction
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import mlflow.sklearn

def train_model():
    X_train, X_test = pd.read_csv("Xtrain.csv"), pd.read_csv("Xtest.csv")
    y_train, y_test = pd.read_csv("ytrain.csv").values.ravel(), pd.read_csv("ytest.csv").values.ravel()
    
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])

    models = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"classifier__n_estimators": [50, 100], "classifier__max_depth": [5, 10]}
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            "params": {"classifier__n_estimators": [50, 100], "classifier__learning_rate": [0.05, 0.1], "classifier__max_depth": [3, 5]}
        }
    }

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("wellness_package_prediction")

    best_overall_score = 0
    best_overall_pipeline = None
    best_algo_name = ""

    for model_name, config in models.items():
        full_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', config["model"])])
        
        with mlflow.start_run(run_name=f"{model_name}_tuning"):
            grid_search = GridSearchCV(full_pipeline, config["params"], cv=3, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            y_pred = best_pipeline.predict(X_test)
            
            acc, f1, roc_auc = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": roc_auc})
            
            # Explicitly pass skops_trusted_types to prevent numpy serialization blocking
            mlflow.sklearn.log_model(
                best_pipeline, 
                artifact_path="model",
                skops_trusted_types=["numpy.dtype", "numpy.int64", "numpy.float64"]
            )
            
            print(f"{model_name} F1: {f1:.4f}")
            if f1 > best_overall_score:
                best_overall_score, best_overall_pipeline, best_algo_name = f1, best_pipeline, model_name

    print(f"Winner: {best_algo_name}")
    os.makedirs("tourism_project/deployment", exist_ok=True)
    joblib.dump(best_overall_pipeline, "tourism_project/deployment/best_model.joblib")

if __name__ == "__main__":
    train_model()
