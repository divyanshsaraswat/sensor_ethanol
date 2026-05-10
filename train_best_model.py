import pandas as pd
import numpy as np
import joblib
import itertools
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import mean_squared_error, r2_score

def train_and_save():
    # Load the dataset
    try:
        df = pd.read_csv('fit.csv')
    except FileNotFoundError:
        print("Error: fit.csv not found.")
        return

    y = df['C (v/v)%']
    all_features = ['R', 'Cond (?S/cm)', 'pH']
    
    # 1. First, find the overall best model for the 3-sensor fusion
    # This is used for the UI's benchmark table
    X_fusion = df[all_features]
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fusion, y, test_size=0.2, random_state=42)
    
    fusion_pipelines = {
        "Polynomial_d2": Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        "SVR": Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.5))
        ]),
        "Wide_MLP": Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=2000, random_state=42, alpha=0.01))
        ]),
        "RandomForest": Pipeline([
            ('model', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42))
        ])
    }

    fusion_results = []
    best_fusion_rmse = float('inf')
    best_fusion_name = ""
    best_fusion_model = None

    print("--- Evaluating Fusion Architectures ---")
    for name, pipe in fusion_pipelines.items():
        pipe.fit(X_train_f, y_train_f)
        y_pred = pipe.predict(X_test_f)
        rmse = np.sqrt(mean_squared_error(y_test_f, y_pred))
        r2 = r2_score(y_test_f, y_pred)
        fusion_results.append({"Model": name, "RMSE": rmse, "R2 Score": r2})
        print(f"{name}: RMSE={rmse:.4f}")
        if rmse < best_fusion_rmse:
            best_fusion_rmse = rmse
            best_fusion_name = name
            best_fusion_model = pipe

    # Save Fusion Metadata
    joblib.dump(best_fusion_model, 'best_model.joblib')
    joblib.dump(all_features, 'features.joblib')
    joblib.dump({'name': best_fusion_name, 'rmse': best_fusion_rmse}, 'model_metadata.joblib')
    joblib.dump(fusion_results, 'model_summary.joblib')

    # 2. Train the Full Power Set of Expert Models (7 combinations)
    print("\n--- Training Model Registry (7 Experts) ---")
    model_registry = {}
    
    # Generate all non-empty combinations
    for r in range(1, len(all_features) + 1):
        for subset in itertools.combinations(all_features, r):
            subset = list(subset)
            # Create a unique key for the registry (sorted feature names joined by comma)
            registry_key = ",".join(sorted(subset))
            
            print(f"Training expert for: {subset}")
            X_sub = df[subset]
            X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y, test_size=0.2, random_state=42)
            
            # Use Polynomial_d2 as the default expert architecture
            expert_pipe = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])
            
            expert_pipe.fit(X_tr, y_tr)
            y_pred = expert_pipe.predict(X_te)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            
            # Save individual joblib for each expert
            clean_name = "_".join([s.split(' ')[0].replace('(', '').replace(')', '') for s in subset])
            filename = f"expert_{clean_name}.joblib"
            joblib.dump(expert_pipe, filename)
            
            model_registry[registry_key] = {
                "features": subset,
                "rmse": rmse,
                "filename": filename
            }
            print(f"  -> RMSE: {rmse:.4f} (Saved as {filename})")

    # Save the registry map
    joblib.dump(model_registry, 'model_registry.joblib')
    print("\nRegistry complete. All 7 combinations are now supported.")

if __name__ == "__main__":
    train_and_save()
