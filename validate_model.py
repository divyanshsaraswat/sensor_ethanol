import joblib
import pandas as pd
import numpy as np

def test_model():
    print("--- Model Validation Script ---")
    
    try:
        model = joblib.load('best_model.joblib')
        features = joblib.load('features.joblib')
        metadata = joblib.load('model_metadata.joblib')
        print(f"Loaded Model: {metadata['name']} (RMSE: {metadata['rmse']:.4f})")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Define test cases within training distribution: [RI, Cond, pH]
    test_cases = [
        {"name": "Pure Water / Near 0%", "data": [1.331, 36.0, 6.95]},
        {"name": "Approx 10%", "data": [1.334, 32.8, 7.05]},
        {"name": "Approx 20%", "data": [1.337, 29.4, 7.15]},
        {"name": "Approx 30%", "data": [1.340, 25.1, 7.35]},
        {"name": "Approx 40%", "data": [1.343, 19.6, 7.55]},
        {"name": "Maximum Trained (45%)", "data": [1.345, 16.4, 7.70]},
        {"name": "Slightly Beyond Range (Extrapolation)", "data": [1.348, 15.0, 7.80]},
    ]

    results = []
    for case in test_cases:
        df = pd.DataFrame([case['data']], columns=features)
        prediction = model.predict(df)[0]
        results.append({
            "Scenario": case['name'],
            "RI": case['data'][0],
            "Cond": case['data'][1],
            "pH": case['data'][2],
            "Predicted %": f"{prediction:.2f}%"
        })

    results_df = pd.DataFrame(results)
    print("\nTest Results:")
    print(results_df.to_string(index=False))

    # Check for unrealistic predictions
    predictions = [float(r['Predicted %'][:-1]) for r in results]
    if any(p < 0 or p > 100 for p in predictions):
        print("\nWARNING: Some predictions are outside the realistic 0-100% range!")
    else:
        print("\nSUCCESS: All test predictions are within realistic bounds (0-100%).")

if __name__ == "__main__":
    test_model()
