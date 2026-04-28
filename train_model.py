#s16798

"""
train_model.py

Run this ONCE before submitting or running the app.

    cd s16798python train_model.py

Reads  : car_price_dataset .csv     (same folder as this file — note the space)
Saves  : best_model.pkl             (same folder — loaded by the Streamlit app)
Saves  : model_diagnostics.pkl      (same folder — charts & SHAP values for app)
Saves  : model_report.txt           (same folder — human-readable summary)
"""
#Imports and setup
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import shap

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, make_scorer

warnings.filterwarnings("ignore")

#Paths 
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH        = os.path.join(BASE_DIR, "car_price_dataset .csv")
MODEL_PATH       = os.path.join(BASE_DIR, "best_model.pkl")
DIAGNOSTICS_PATH = os.path.join(BASE_DIR, "model_diagnostics.pkl")
REPORT_PATH      = os.path.join(BASE_DIR, "model_report.txt")

#Province mapping (matches data_loader.py exactly)
TOWN_TO_PROVINCE = {
    "Colombo":"Western","Gampaha":"Western","Negombo":"Western",
    "Kalutara":"Western","Panadura":"Western","Moratuwa":"Western",
    "Dehiwala-Mount-Lavinia":"Western","Maharagama":"Western",
    "Kotte":"Western","Wattala":"Western","Ja-Ela":"Western",
    "Kelaniya":"Western","Kadawatha":"Western","Nugegoda":"Western",
    "Piliyandala":"Western","Boralesgamuwa":"Western",
    "Kandy":"Central","Matale":"Central","Nuwara-Eliya":"Central",
    "Gampola":"Central","Nawalapitiya":"Central","Hatton":"Central",
    "Galle":"Southern","Matara":"Southern","Hambantota":"Southern",
    "Weligama":"Southern","Tangalle":"Southern","Hikkaduwa":"Southern",
    "Ambalangoda":"Southern","Jaffna":"Northern","Vavuniya":"Northern",
    "Kilinochchi":"Northern","Mullaitivu":"Northern",
    "Batticaloa":"Eastern","Trincomalee":"Eastern","Ampara":"Eastern",
    "Kalmunai":"Eastern","Kurunegala":"North Western","Puttalam":"North Western",
    "Kuliyapitiya":"North Western","Chilaw":"North Western",
    "Anuradapura":"North Central","Polonnaruwa":"North Central",
    "Badulla":"Uva","Bandarawela":"Uva","Haputale":"Uva","Welimada":"Uva",
    "Ratnapura":"Sabaragamuwa","Kegalle":"Sabaragamuwa","Balangoda":"Sabaragamuwa",
}

#Features 
CATEGORICAL_FEATURES = ["Fuel Type", "Province"]
NUMERICAL_FEATURES   = [
    # Age (2026 - YOM) is EXCLUDED from modeling:
    #   perfect multicollinearity with Millage(KM), correlation = 1.0
    #   Age is calculated in the app for display purposes only
    "Engine (cc)", "Millage(KM)",
    "Gear_bin", "Leasing_bin",
    "AIR CONDITION_bin", "POWER STEERING_bin", "POWER MIRROR_bin", "POWER WINDOW_bin",
    "Condition_bin",
]
ALL_FEATURES  = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
THRESHOLD     = 20   # minimum listings to keep a category; rarer - grouped as "Other"


#Data loading & preparation (mirrors data_loader.py exactly)
def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop unnamed index column if present
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Drop nulls and duplicate rows
    df = df.dropna().drop_duplicates().copy()

    # Parse date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Age — computed for display only, NOT used in modeling
    df["Age"] = 2026 - df["YOM"]

    # Binary features — "Available" encoding (matches data_loader.py)
    for col in ["AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW"]:
        df[col + "_bin"] = (df[col] == "Available").astype(int)

    # Leasing, Gear, Condition binaries
    df["Leasing_bin"]   = (df["Leasing"] != "No Leasing").astype(int)
    df["Gear_bin"]      = (df["Gear"] == "Automatic").astype(int)
    df["Condition_bin"] = (df["Condition"] == "NEW").astype(int)

    # Standardise text columns
    for col in ["Brand", "Model", "Town"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    df = df[(df["Brand"] != "") & (df["Model"] != "")]

    # Province mapping — Town to Province
    df["Province"] = df["Town"].map(TOWN_TO_PROVINCE).fillna("Other")

    # Group rare categories into "Other" (threshold = 20 listings)
    for col in ["Brand", "Model", "Fuel Type"]:
        counts      = df[col].value_counts()
        rare        = counts[counts < THRESHOLD].index
        df[col]     = df[col].replace(rare, "Other")

    return df


#Main training steps
def train():
    print(f"Loading data: {DATA_PATH}")
    df = load_and_prepare(DATA_PATH)
    print(f"Rows after cleaning: {len(df)}")

    X = df[ALL_FEATURES]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train rows : {len(X_train)}  |  Test rows : {len(X_test)}")

    #Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
        ("num", Pipeline([("scaler", StandardScaler())]),                        NUMERICAL_FEATURES),
    ])

    cv         = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    #Model configs with hyperparameter grids
    models_config = {
        "Linear Regression": {
            "model": LinearRegression(),
            "params": None,
        },
        "Ridge": {
            "model": Ridge(),
            "params": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
        },
        "Lasso": {
            "model": Lasso(max_iter=2000),
            "params": {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
        },
        "Elastic Net": {
            "model": ElasticNet(max_iter=2000),
            "params": {
                "model__alpha":    [0.0001, 0.001, 0.01, 0.1, 1.0],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "model__max_depth":         [5, 10, 15, 20, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf":  [1, 2, 4],
            },
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "model__n_estimators":      [50, 100, 200],
                "model__max_depth":         [10, 20, None],
                "model__min_samples_split": [2, 5, 10],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "model__n_estimators":  [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__max_depth":     [3, 5, 7],
            },
        },
    }

    #Train & tune all models
    results     = {}
    best_params = {}

    for name, config in models_config.items():
        print(f"\nTraining : {name} ...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model",        config["model"]),
        ])

        if config["params"] is not None:
            gs = GridSearchCV(
                pipeline, config["params"],
                cv=cv, scoring=mae_scorer, n_jobs=-1, verbose=0,
            )
            gs.fit(X_train, y_train)
            best_pipeline     = gs.best_estimator_
            best_params[name] = gs.best_params_
            print(f"  Best params  : {gs.best_params_}")
        else:
            best_pipeline = pipeline
            best_pipeline.fit(X_train, y_train)
            best_params[name] = {}

        y_tr = best_pipeline.predict(X_train)
        y_te = best_pipeline.predict(X_test)

        results[name] = {
            "pipeline":   best_pipeline,
            "Train_RMSE": np.sqrt(mean_squared_error(y_train, y_tr)),
            "Train_R2":   r2_score(y_train, y_tr),
            "Train_MAE":  mean_absolute_error(y_train, y_tr),
            "Test_RMSE":  np.sqrt(mean_squared_error(y_test,  y_te)),
            "Test_R2":    r2_score(y_test,  y_te),
            "Test_MAE":   mean_absolute_error(y_test,  y_te),
        }
        print(f"  Train  RMSE: {results[name]['Train_RMSE']:.4f}  R²: {results[name]['Train_R2']:.4f}")
        print(f"  Test   RMSE: {results[name]['Test_RMSE']:.4f}  R²: {results[name]['Test_R2']:.4f}")

    #Select best model by lowest Test RMSE
    best_name     = min(results, key=lambda x: results[x]["Test_RMSE"])
    best_pipeline = results[best_name]["pipeline"]

    print(f"\n{'='*55}")
    print(f"  Best model : {best_name}")
    print(f"  Test RMSE  : {results[best_name]['Test_RMSE']:.4f}")
    print(f"  Test R²    : {results[best_name]['Test_R2']:.4f}")
    print(f"{'='*55}")

    #Test-set predictions & residuals
    y_test_pred = best_pipeline.predict(X_test)
    residuals   = np.array(y_test) - y_test_pred

    #SHAP feature importance
    print("\nComputing SHAP values ...")

    preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
    inner_model         = best_pipeline.named_steps["model"]
    X_test_transformed  = preprocessor_fitted.transform(X_test)

    # Reconstruct feature names after one-hot encoding
    cat_names = (
        preprocessor_fitted
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
        .tolist()
    )
    all_feature_names = cat_names + NUMERICAL_FEATURES

    try:
        if hasattr(inner_model, "feature_importances_"):
            # Tree-based models (RandomForest, GradientBoosting, DecisionTree)
            explainer   = shap.TreeExplainer(inner_model)
            shap_values = explainer.shap_values(X_test_transformed)
        else:
            # Linear models (LinearRegression, Ridge, Lasso, ElasticNet)
            X_dense     = X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed
            explainer   = shap.LinearExplainer(inner_model, X_dense)
            shap_values = explainer.shap_values(X_dense)

        shap_importance = pd.DataFrame({
            "Feature":    all_feature_names,
            "Importance": np.abs(shap_values).mean(axis=0),
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        print("  SHAP values computed successfully.")

    except Exception as e:
        print(f"  SHAP failed ({e}) — using built-in importances as fallback.")
        if hasattr(inner_model, "feature_importances_"):
            imp = inner_model.feature_importances_
        elif hasattr(inner_model, "coef_"):
            imp = np.abs(inner_model.coef_)
        else:
            imp = np.ones(len(all_feature_names))

        shap_importance = pd.DataFrame({
            "Feature":    all_feature_names,
            "Importance": imp,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    #Save best model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"\n best_model.pkl saved - {MODEL_PATH}")

    #Save diagnostics (used by Streamlit app page)
    diagnostics = {
        "best_model_name": best_name,
        "y_test":          np.array(y_test),
        "y_test_pred":     y_test_pred,
        "residuals":       residuals,
        "shap_importance": shap_importance,
        "all_results": {
            name: {k: v for k, v in res.items() if k != "pipeline"}
            for name, res in results.items()
        },
        "metrics": {
            "Train_RMSE": results[best_name]["Train_RMSE"],
            "Train_R2":   results[best_name]["Train_R2"],
            "Train_MAE":  results[best_name]["Train_MAE"],
            "Test_RMSE":  results[best_name]["Test_RMSE"],
            "Test_R2":    results[best_name]["Test_R2"],
            "Test_MAE":   results[best_name]["Test_MAE"],
        },
    }
    with open(DIAGNOSTICS_PATH, "wb") as f:
        pickle.dump(diagnostics, f)
    print(f"model_diagnostics.pkl saved - {DIAGNOSTICS_PATH}")

    #Save human-readable report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("  CAR PRICE MODEL REPORT\n")
        f.write("=" * 55 + "\n\n")

        f.write("Predictors used:\n")
        for col in ALL_FEATURES:
            f.write(f"  - {col}\n")
        f.write("\nNote: Age (2025 - YOM) excluded -- perfect multicollinearity\n")
        f.write("      with Millage(KM) (r = 1.0). Shown in UI for display only.\n\n")

        f.write("Rare category threshold: < 20 listings -> grouped as 'Other'\n")
        f.write("  Applied to: Brand, Model, Fuel Type\n\n")

        f.write("Model Performance (Train vs Test):\n")
        for name, res in results.items():
            marker = " ← BEST" if name == best_name else ""
            f.write(
                f"  {name}{marker}\n"
                f"    Train  RMSE: {res['Train_RMSE']:.4f}  R²: {res['Train_R2']:.4f}  MAE: {res['Train_MAE']:.4f}\n"
                f"    Test   RMSE: {res['Test_RMSE']:.4f}  R²: {res['Test_R2']:.4f}  MAE: {res['Test_MAE']:.4f}\n\n"
            )

        f.write("Best Hyperparameters (GridSearchCV):\n")
        for name, params in best_params.items():
            if params:
                for param, val in params.items():
                    f.write(f"  {name} — {param.replace('model__','')}: {val}\n")

    print(f"model_report.txt      saved- {REPORT_PATH}")
    print("\nAll done. You can now run: streamlit run app.py")


if __name__ == "__main__":
    train()