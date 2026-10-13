from pathlib import Path
import pandas as pd

# =========================================================
# CREDIT SCORING MODEL - DATASET PREPARATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "german.data"

# =========================================================
# FEATURE NAMES
# =========================================================

FEATURE_NAMES = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account_bonds",
    "present_employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors_guarantors",
    "present_residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable_for_maintenance",
    "telephone",
    "foreign_worker",
    "target"
]

# Numerical features
NUMERICAL_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "present_residence_since",
    "age",
    "existing_credits",
    "people_liable_for_maintenance"
]

# Categorical features
CATEGORICAL_FEATURES = [
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

# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 70)
print("CREDIT SCORING MODEL - DATASET PREPARATION")
print("=" * 70)

print("\nDataset:")
print(DATASET_PATH)

if not DATASET_PATH.exists():
    print("\n❌ Dataset not found!")
    raise SystemExit

df = pd.read_csv(
    DATASET_PATH,
    sep=r"\s+",
    header=None,
    engine="python"
)

print("\n✅ Dataset loaded successfully!")

# =========================================================
# ASSIGN COLUMN NAMES
# =========================================================

df.columns = FEATURE_NAMES

# =========================================================
# TARGET MAPPING
# =========================================================

df["target"] = df["target"].map({
    1: "Good",
    2: "Bad"
})

# =========================================================
# BASIC INFORMATION
# =========================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# =========================================================
# FEATURE INFORMATION
# =========================================================

print("\n" + "=" * 70)
print("NUMERICAL FEATURES")
print("=" * 70)

for feature in NUMERICAL_FEATURES:
    print("-", feature)

print("\n" + "=" * 70)
print("CATEGORICAL FEATURES")
print("=" * 70)

for feature in CATEGORICAL_FEATURES:
    print("-", feature)

# =========================================================
# TARGET DISTRIBUTION
# =========================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(df["target"].value_counts())

print("\nTarget percentage:")

print(
    (df["target"].value_counts(normalize=True) * 100)
    .round(2)
)

# =========================================================
# MISSING VALUES
# =========================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print("Total missing values:", df.isnull().sum().sum())

# =========================================================
# DUPLICATES
# =========================================================

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

print("Duplicate rows:", df.duplicated().sum())

# =========================================================
# SAMPLE DATA
# =========================================================

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(df.head().to_string())

# =========================================================
# DATA TYPES
# =========================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)

# =========================================================
# SAVE CLEAN DATASET
# =========================================================

OUTPUT_PATH = BASE_DIR / "dataset" / "german_credit_clean.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(f"✅ Clean dataset saved at:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("✅ DATASET PREPARATION COMPLETED")
print("=" * 70)