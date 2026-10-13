from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "german_credit_clean.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["target"])

# Good = 0
# Bad  = 1
y = df["target"].map({
    "Good": 0,
    "Bad": 1
})


# ============================================================
# 4. FEATURE TYPES
# ============================================================

numerical_features = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "present_residence_since",
    "age",
    "existing_credits",
    "people_liable_for_maintenance"
]

categorical_features = [
    "checking_account_status",
    "credit_history",
    "purpose",
    "savings_account_bonds",
    "present_employment_since",
    "personal_status_sex",
    "other_debtors_guarantors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# 6. ONE-HOT ENCODER
# ============================================================

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    # Compatibility with older scikit-learn versions
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


# ============================================================
# 7. PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            encoder,
            categorical_features
        )
    ]
)


# ============================================================
# 8. MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 9. COST MATRIX
# ============================================================

# Rows    = Actual
# Columns = Predicted
#
#              Predicted
#              Good   Bad
#
# Actual Good    0     1
# Actual Bad     5     0
#
# Predicting Bad customer as Good = COST 5

COST_MATRIX = {
    (0, 0): 0,   # Good -> Good
    (0, 1): 1,   # Good -> Bad
    (1, 0): 5,   # Bad  -> Good
    (1, 1): 0    # Bad  -> Bad
}


# ============================================================
# 10. COST FUNCTION
# ============================================================

def calculate_cost(y_true, y_pred):

    total_cost = 0

    for actual, predicted in zip(y_true, y_pred):
        total_cost += COST_MATRIX[(actual, predicted)]

    return total_cost


# ============================================================
# 11. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

print("\n" + "=" * 70)
print("MODEL TRAINING AND EVALUATION")
print("=" * 70)


for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probability for ROC-AUC
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    # Cost
    total_cost = calculate_cost(
        y_test,
        y_pred
    )

    average_cost = total_cost / len(y_test)

    # Store
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "Total_Cost": total_cost,
        "Average_Cost": average_cost
    })

    trained_models[model_name] = pipeline

    # Display
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"ROC-AUC        : {roc_auc:.4f}")
    print(f"Total Cost     : {total_cost}")
    print(f"Average Cost   : {average_cost:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


# ============================================================
# 12. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 13. SELECT BEST MODEL
# ============================================================

# Primary criterion:
# Lowest cost according to the German Credit cost matrix.

best_row = results_df.loc[
    results_df["Total_Cost"].idxmin()
]

best_model_name = best_row["Model"]

best_model = trained_models[best_model_name]


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

best_model_path = MODEL_DIR / "best_model.pkl"

joblib.dump(
    best_model,
    best_model_path
)


# ============================================================
# 15. SAVE INDIVIDUAL MODELS
# ============================================================

for model_name, pipeline in trained_models.items():

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".pkl"
    )

    joblib.dump(
        pipeline,
        MODEL_DIR / filename
    )


# ============================================================
# 16. SAVE RESULTS
# ============================================================

results_path = MODEL_DIR / "model_results.csv"

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(f"\nBest model based on lowest cost: {best_model_name}")

print(f"\nBest model saved at:")
print(best_model_path)

print(f"\nModel results saved at:")
print(results_path)

print("\nGenerated model files:")

for file in MODEL_DIR.glob("*.pkl"):
    print(f"  - {file.name}")

print("\nDone!")