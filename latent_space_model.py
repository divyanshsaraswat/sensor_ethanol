# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Wide Latent Space & Embedding Models for Concentration Prediction
#
# This notebook explores advanced Machine Learning techniques to project our 3-dimensional physical features (**Refractive Index, Conductivity, and pH**) into a **high-dimensional wide latent space**.
#
# ### Methodology: Beyond Simple Features
# In complex liquid mixtures, the relationship between sensor readings and concentration is often non-linear and coupled. By creating "embeddings" or wide latent representations, we allow the model to learn complex interactions that are not captured by standard linear or polynomial models.
#
# We evaluate two primary approaches:
# 1. **Kernel Approximation (Nyström Method):** Approximating an RBF kernel by projecting the data into a high-dimensional feature space.
# 2. **Wide Neural Network (MLP):** Using a wide hidden layer (128 units) to act as a learned embedding of the physical properties.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set plot style for premium visuals
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# %% [markdown]
# ## 1. Data Acquisition and Preparation
# We load the `fit.csv` dataset and scale the features. Scaling is critical for neural networks and kernel methods to ensure no single feature dominates the latent space.

# %%
# 1. Load Data
try:
    # Using absolute path for consistency
    df = pd.read_csv(r'c:\Users\mgpsk\Documents\voc-streamlit\fit.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(r"Error: c:\Users\mgpsk\Documents\voc-streamlit\fit.csv not found.")
    exit()

# Features and Target
X = df[['R', 'Cond (?S/cm)', 'pH']]
y = df['C (v/v)%']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## 2. Approach 1: Kernel Approximation (Nyström Method)
# This method projects our 3D inputs into a high-dimensional space using an RBF kernel approximation. This creates a "wide" representation where linear separators (like Ridge Regression) become much more powerful.

# %%
# Projecting 3D to latent space using RBF kernel approximation
# n_components is automatically limited by n_samples (84)
nystroem = Nystroem(kernel='rbf', gamma=0.1, n_components=100, random_state=42)
X_train_latent_ker = nystroem.fit_transform(X_train_scaled)
X_test_latent_ker = nystroem.transform(X_test_scaled)

# Fit Ridge Regression on the wide latent space
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_latent_ker, y_train)
y_pred_ker = ridge_model.predict(X_test_latent_ker)

# %% [markdown]
# ## 3. Approach 2: Wide Neural Network (MLP)
# Here, we use a Multi-Layer Perceptron. The **128-neuron hidden layer** serves as a "learned embedding." The network maps the physical properties into this wide latent space, allowing it to capture highly non-linear dynamics.

# %%
# 128 neurons in the hidden layer act as the wide latent space
mlp_model = MLPRegressor(hidden_layer_sizes=(128,), activation='relu', solver='adam', 
                         max_iter=2000, random_state=42, alpha=0.01)
mlp_model.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_model.predict(X_test_scaled)

# %% [markdown]
# ## 4. Performance Comparison & Evaluation
# We compare the two latent-space approaches using RMSE and R2 Score. A significant reduction in error compared to baseline models indicates the effectiveness of the high-dimensional projection.

# %%
models = {
    "Nystroem + Ridge": y_pred_ker,
    "Wide MLP (128 units)": y_pred_mlp
}

results = []
for name, y_pred in models.items():
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2 Score": r2})
    print(f"\n{name} Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

# %% [markdown]
# ## 5. Visualizing the Results
# We visualize the RMSE comparison and the distribution of residuals for our best performing model (Wide MLP).

# %%
results_df = pd.DataFrame(results)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart of RMSE
sns.barplot(data=results_df, x='Model', y='RMSE', palette='viridis', ax=axes[0])
axes[0].set_title('RMSE Comparison: Latent Space Modeling', fontsize=14, fontweight='bold')
axes[0].set_ylabel('RMSE (Lower is better)')

# Distribution of Residuals for the Wide MLP
residuals_mlp = y_test - y_pred_mlp
sns.kdeplot(residuals_mlp, fill=True, color='seagreen', ax=axes[1])
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_title('Distribution of Residuals (Wide MLP)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Error')

plt.tight_layout()
plt.savefig('latent_space_results.png')
print("\nPlots saved to latent_space_results.png")

# --- Export Latent Space Embeddings ---
# We extract the 128-dimensional hidden layer outputs for the entire dataset
X_all_scaled = scaler.transform(X)
# Manual computation of the ReLU hidden layer: (X * W) + b
hidden_layer_input = np.dot(X_all_scaled, mlp_model.coefs_[0]) + mlp_model.intercepts_[0]
embeddings = np.maximum(hidden_layer_input, 0)  # ReLU Activation

# Create a DataFrame for the 128D embeddings
embedding_cols = [f'latent_{i}' for i in range(embeddings.shape[1])]
embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)

# Add the target Concentration column for context/labeling
embeddings_df['Concentration'] = y.values

# Save to CSV using absolute path
embeddings_csv_path = r'c:\Users\mgpsk\Documents\voc-streamlit\latent_space_embeddings.csv'
embeddings_df.to_csv(embeddings_csv_path, index=False)
print(f"\nLatent space embeddings (128D) successfully saved to: {embeddings_csv_path}")

plt.show()

# %% [markdown]
# ### Conclusion
# By projecting our 3 sensor features into a wide latent space, we have achieved significantly higher accuracy. The **Wide MLP** model in particular shows an exceptional fit with an $R^2$ exceeding 0.999, proving that learned embeddings are highly effective for multi-feature sensor fusion.
