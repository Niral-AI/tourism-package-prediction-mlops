import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import mlflow
import mlflow.xgboost

def train_model():
    X_train = pd.read_csv("Xtrain.csv")
    X_test = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").values.ravel()
    y_test = pd.read_csv("ytest.csv").values.ravel()
    
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1]
    }
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("wellness_package_prediction")
    
    with mlflow.start_run():
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": roc_auc})
        
        print("\n================== BEST MODEL METRICS ==================")
        print(classification_report(y_test, y_pred))
        print("========================================================")
        
        os.makedirs("tourism_project/deployment", exist_ok=True)
        joblib.dump(best_model, "tourism_project/deployment/best_model.joblib")
        print("Model saved to tourism_project/deployment/best_model.joblib")

if __name__ == "__main__":
    train_model()
