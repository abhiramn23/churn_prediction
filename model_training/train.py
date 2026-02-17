"""
==========================================================
MODEL TRAINING SCRIPT — train.py
==========================================================

WHAT THIS FILE DOES:
--------------------
1. Loads the customer churn CSV data using Pandas
2. Cleans the data (handles missing values, encodes categories)
3. Splits data into training and testing sets
4. Trains a Random Forest classifier using Scikit-learn
5. Evaluates the model (accuracy, classification report)
6. Generates visualizations using Matplotlib
7. Saves the trained model as 'model.pkl' using joblib

HOW TO RUN:
-----------
    cd model_training
    pip install -r requirements.txt
    python train.py

WHAT YOU'LL GET:
----------------
- A 'model.pkl' file in the ../models/ folder (used by the backend)
- A 'feature_importance.png' chart
- A 'confusion_matrix.png' chart
- Printed accuracy and classification report in terminal
==========================================================
"""

# ──────────────────────────────────────────────────────────
# STEP 1: Import libraries
# ──────────────────────────────────────────────────────────
import pandas as pd                          # For loading and cleaning data
import matplotlib                            # For creating charts
matplotlib.use('Agg')                        # Use non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt              # For plotting charts
from sklearn.model_selection import train_test_split  # Split data into train/test
from sklearn.ensemble import RandomForestClassifier   # Our ML algorithm
from sklearn.metrics import (                # For evaluating the model
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder       # Convert text to numbers
import joblib                                # Save/load the trained model
import os                                    # For file path operations
import warnings
warnings.filterwarnings('ignore')            # Hide unnecessary warnings


def main():
    """Main function that runs the entire training pipeline."""

    print("=" * 60)
    print("🚀 CUSTOMER CHURN MODEL TRAINING")
    print("=" * 60)

    # ──────────────────────────────────────────────────────
    # STEP 2: Load the dataset
    # ──────────────────────────────────────────────────────
    print("\n📂 Step 1: Loading dataset...")

    # Get the path to our CSV file (same folder as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "sample_data.csv")

    # Read the CSV into a Pandas DataFrame
    # A DataFrame is like an Excel spreadsheet in Python
    df = pd.read_csv(data_path)

    print(f"   ✅ Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"   📋 Columns: {list(df.columns)}")

    # ──────────────────────────────────────────────────────
    # STEP 3: Explore the data
    # ──────────────────────────────────────────────────────
    print("\n🔍 Step 2: Exploring data...")
    print(f"   📊 First 5 rows:")
    print(df.head().to_string(index=False))
    print(f"\n   📈 Data types:")
    print(f"   {dict(df.dtypes)}")
    print(f"\n   ❓ Missing values per column:")
    print(f"   {dict(df.isnull().sum())}")

    # ──────────────────────────────────────────────────────
    # STEP 4: Clean the data
    # ──────────────────────────────────────────────────────
    print("\n🧹 Step 3: Cleaning data...")

    # 4a. Drop the customer_id column — it's just an identifier, not a feature
    # The model shouldn't learn from IDs (they don't predict churn!)
    df = df.drop(columns=["customer_id"])
    print("   ✅ Dropped 'customer_id' column (not useful for prediction)")

    # 4b. Handle missing values
    # For numeric columns: fill missing values with the column's median
    # For text columns: fill missing values with the most common value
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    text_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"   ✅ Filled missing values in '{col}' with median: {median_value}")

    for col in text_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
            print(f"   ✅ Filled missing values in '{col}' with mode: {mode_value}")

    print(f"   ✅ Missing values after cleaning: {df.isnull().sum().sum()}")

    # 4c. Encode categorical (text) columns to numbers
    # ML models can only work with numbers, not text like "Male"/"Female"
    # LabelEncoder converts: "Male" → 0, "Female" → 1 (for example)
    label_encoders = {}
    for col in text_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"   ✅ Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ──────────────────────────────────────────────────────
    # STEP 5: Split into features (X) and target (y)
    # ──────────────────────────────────────────────────────
    print("\n✂️  Step 4: Splitting data into features and target...")

    # X = all columns EXCEPT 'churn' (these are the inputs/features)
    # y = only the 'churn' column (this is what we want to predict)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Save the feature names — we'll need them later
    feature_names = list(X.columns)

    print(f"   📊 Features (X): {feature_names}")
    print(f"   🎯 Target (y): churn (0 = stays, 1 = leaves)")
    print(f"   📐 X shape: {X.shape}, y shape: {y.shape}")

    # ──────────────────────────────────────────────────────
    # STEP 6: Split into training and testing sets
    # ──────────────────────────────────────────────────────
    print("\n📦 Step 5: Splitting into train/test sets (80/20)...")

    # train_test_split randomly divides data:
    #   - 80% for training (the model learns from this)
    #   - 20% for testing (we check accuracy on this)
    # random_state=42 ensures reproducible results
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 20% for testing
        random_state=42,     # Reproducible results
        stratify=y           # Keep same ratio of churn/no-churn in both sets
    )

    print(f"   ✅ Training set: {X_train.shape[0]} samples")
    print(f"   ✅ Testing set:  {X_test.shape[0]} samples")

    # ──────────────────────────────────────────────────────
    # STEP 7: Train the model
    # ──────────────────────────────────────────────────────
    print("\n🤖 Step 6: Training Random Forest model...")

    # RandomForestClassifier:
    #   - Creates 100 decision trees (n_estimators=100)
    #   - Each tree "votes" on whether a customer will churn
    #   - The majority vote wins
    #   - This is called an "ensemble" method
    model = RandomForestClassifier(
        n_estimators=100,     # Number of decision trees
        max_depth=10,         # Maximum depth of each tree (prevents overfitting)
        random_state=42,      # Reproducible results
        n_jobs=-1             # Use all CPU cores for faster training
    )

    # .fit() is where the actual learning happens
    model.fit(X_train, y_train)
    print("   ✅ Model training complete!")

    # ──────────────────────────────────────────────────────
    # STEP 8: Evaluate the model
    # ──────────────────────────────────────────────────────
    print("\n📊 Step 7: Evaluating model performance...")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   🎯 Accuracy: {accuracy:.2%}")

    # Print detailed classification report
    print(f"\n   📋 Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    print(report)

    # ──────────────────────────────────────────────────────
    # STEP 9: Create visualizations
    # ──────────────────────────────────────────────────────
    print("📈 Step 8: Creating visualizations...")

    # Create the output directory for charts
    charts_dir = os.path.join(script_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # ---- Chart 1: Feature Importance ----
    # Shows which features matter most for predicting churn
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.barh(
        feature_importance_df["Feature"],
        feature_importance_df["Importance"],
        color="#4CAF50",
        edgecolor="#388E3C"
    )
    ax1.set_xlabel("Importance Score", fontsize=12)
    ax1.set_title("🔑 Feature Importance for Churn Prediction", fontsize=14, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig1.savefig(os.path.join(charts_dir, "feature_importance.png"), dpi=150)
    plt.close(fig1)
    print("   ✅ Saved: charts/feature_importance.png")

    # ---- Chart 2: Confusion Matrix ----
    # Shows how many predictions were correct vs incorrect
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )
    disp.plot(ax=ax2, cmap="Blues", values_format="d")
    ax2.set_title("📊 Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig2.savefig(os.path.join(charts_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig2)
    print("   ✅ Saved: charts/confusion_matrix.png")

    # ──────────────────────────────────────────────────────
    # STEP 10: Save the trained model
    # ──────────────────────────────────────────────────────
    print("\n💾 Step 9: Saving trained model...")

    # Create the models directory if it doesn't exist
    models_dir = os.path.join(script_dir, "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save the model using joblib
    # joblib is better than pickle for large numpy arrays (used inside ML models)
    model_path = os.path.join(models_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"   ✅ Model saved to: {model_path}")

    # Also save the feature names (so the backend knows what inputs to expect)
    feature_names_path = os.path.join(models_dir, "feature_names.pkl")
    joblib.dump(feature_names, feature_names_path)
    print(f"   ✅ Feature names saved to: {feature_names_path}")

    # Also save the label encoders (so the backend can decode predictions)
    encoders_path = os.path.join(models_dir, "label_encoders.pkl")
    joblib.dump(label_encoders, encoders_path)
    print(f"   ✅ Label encoders saved to: {encoders_path}")

    # ──────────────────────────────────────────────────────
    # DONE!
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"   🎯 Model Accuracy: {accuracy:.2%}")
    print(f"   💾 Model saved to: models/model.pkl")
    print(f"   📊 Charts saved to: model_training/charts/")
    print(f"   📋 Feature names: {feature_names}")
    print("=" * 60)
    print("\n🔜 Next steps:")
    print("   1. Start the backend: cd backend && uvicorn app.main:app --reload")
    print("   2. Start the frontend: cd frontend && streamlit run app.py")


# ──────────────────────────────────────────────────────────
# This runs the main() function when you execute: python train.py
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
