"""
Product Demand Prediction - predict product demand (Quantity)
using the Superstore Sales dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("PRODUCT DEMAND PREDICTION PROJECT")
print("=" * 60)


# LOAD DATASET
print("\n1. LOADING DATASET...")
print("-" * 40)

# Load the dataset (comma-delimited CSV with latin-1 encoding)
df = pd.read_csv("Sample - Superstore.csv", encoding="latin-1")

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head())


# BASIC DATA CLEANING
print("\n2. DATA CLEANING...")
print("-" * 40)

# Check for missing values
print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

# Remove duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape[0]} rows")

# EDA
print("\n3. EXPLORATORY DATA ANALYSIS...")
print("-" * 40)

# Create output directory for plots
os.makedirs("plots", exist_ok=True)

# Plot 1: Distribution of Quantity (Target Variable)
plt.figure(figsize=(10, 6))
plt.hist(df['Quantity'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
plt.title('Distribution of Product Demand (Quantity)', fontsize=14, fontweight='bold')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('plots/quantity_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/quantity_distribution.png")

# Plot 2: Sales vs Quantity
plt.figure(figsize=(10, 6))
plt.scatter(df['Sales'], df['Quantity'], alpha=0.5, color='coral')
plt.title('Sales vs Quantity', fontsize=14, fontweight='bold')
plt.xlabel('Sales ($)')
plt.ylabel('Quantity')
plt.grid(True, alpha=0.3)
plt.savefig('plots/sales_vs_quantity.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/sales_vs_quantity.png")

# Plot 3: Discount vs Quantity
plt.figure(figsize=(10, 6))
plt.scatter(df['Discount'], df['Quantity'], alpha=0.5, color='green')
plt.title('Discount vs Quantity', fontsize=14, fontweight='bold')
plt.xlabel('Discount')
plt.ylabel('Quantity')
plt.grid(True, alpha=0.3)
plt.savefig('plots/discount_vs_quantity.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/discount_vs_quantity.png")

# Plot 4: Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f')
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/correlation_heatmap.png")

print("\nAll plots saved in 'plots/' folder!")


# FEATURE ENGINEERING
print("\n4. FEATURE ENGINEERING...")
print("-" * 40)

# Create new feature: price_per_unit
df['price_per_unit'] = df['Sales'] / df['Quantity']

print("Created new feature: price_per_unit = Sales / Quantity")
print(f"\nPrice per unit statistics:")
print(df['price_per_unit'].describe())

# SELECT FEATURES
print("\n5. SELECTING FEATURES...")
print("-" * 40)

# Features (X) and Target (y)
features = ['Sales', 'Discount', 'Profit', 'price_per_unit']
X = df[features]
y = df['Quantity']

print(f"Features selected: {features}")
print(f"Target variable: Quantity")
print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# TRAIN/TEST SPLIT
print("\n6. TRAIN/TEST SPLIT...")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# TRAIN SIMPLE MODELS
print("\n7. TRAINING MODELS...")
print("-" * 40)

# Model 1: Linear Regression
print("\nTraining Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Model 2: Random Forest Regressor
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# COMPARE MODEL PERFORMANCE
print("\n8. MODEL PERFORMANCE COMPARISON...")
print("-" * 40)

# Calculate metrics for Linear Regression
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

# Calculate metrics for Random Forest
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

# Display results
print("\n" + "=" * 50)
print(f"{'Model':<25} {'MAE':<12} {'R2 Score':<12}")
print("=" * 50)
print(f"{'Linear Regression':<25} {lr_mae:<12.4f} {lr_r2:<12.4f}")
print(f"{'Random Forest':<25} {rf_mae:<12.4f} {rf_r2:<12.4f}")
print("=" * 50)

# Determine best model
if rf_r2 > lr_r2:
    best_model = rf_model
    best_model_name = "Random Forest"
    print(f"\nBest Model: {best_model_name} (higher R2 score)")
else:
    best_model = lr_model
    best_model_name = "Linear Regression"
    print(f"\nBest Model: {best_model_name} (higher R2 score)")


#SAVE THE BEST MODEL
print("\n9. SAVING THE BEST MODEL...")
print("-" * 40)

joblib.dump(best_model, 'model.pkl')
print(f"✓ Model saved as: model.pkl")

# feature names for reference
joblib.dump(features, 'feature_names.pkl')
print(f"✓ Feature names saved as: feature_names.pkl")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nFiles created:")
print("  - model.pkl (trained model)")
print("  - feature_names.pkl (feature list)")
print("  - plots/ folder (visualizations)")
print("\nNext step: Run 'streamlit run streamlit_app.py' to use the prediction app!")
