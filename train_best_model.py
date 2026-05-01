import pandas as pd
import numpy as np
import joblib
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

    # Define target and features
    y = df['C (v/v)%']
    X_multi = df[['R', 'Cond (?S/cm)', 'pH']]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)

    # Define pipelines
    pipelines = {
        "Linear": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        "Polynomial_d2": Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        "SVR": Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.5))
        ]),
        "RandomForest": Pipeline([
            ('model', RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42))
        ]),
        "Wide_MLP": Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(hidden_layer_sizes=(128,), activation='relu', solver='adam', 
                                   max_iter=2000, random_state=42, alpha=0.01))
        ]),
        "Nystroem_Ridge": Pipeline([
            ('scaler', StandardScaler()),
            ('nystroem', Nystroem(kernel='rbf', gamma=0.1, n_components=100, random_state=42)),
            ('model', Ridge(alpha=1.0))
        ])
    }

    best_rmse = float('inf')
    best_model = None
    best_name = ""
    results = []
    trained_models = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model: {name}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        results.append({
            "Model": name,
            "RMSE": rmse,
            "R2 Score": r2
        })
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline
            best_name = name

    if best_model:
        print(f"\nBest Model: {best_name} with RMSE: {best_rmse:.4f}")
        joblib.dump(best_model, 'best_model.joblib')
        print(f"Saved best model to 'best_model.joblib'")
        
        # Save feature names, metadata, and full summary for Gradio
        joblib.dump(['R', 'Cond (?S/cm)', 'pH'], 'features.joblib')
        joblib.dump({'name': best_name, 'rmse': best_rmse}, 'model_metadata.joblib')
        joblib.dump(results, 'model_summary.joblib')

if __name__ == "__main__":
    train_and_save()
