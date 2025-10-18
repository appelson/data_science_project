# ============================================================
#  Libraries
# ============================================================
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# --- Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
print("Libraries imported successfully.")

# ============================================================
#  Configuration
# ============================================================
FILE_PATH = "PLS_FY2022 PUD_CSV/PLS_FY22_AE_pud22i.csv"
TARGET_VARIABLE = 'log_visits_per_capita'  # Continuous variable to derive binary target
THRESHOLD = 3.0                            # Binary cutoff for "good" vs "poor"
RANDOM_STATE = 123

# --- Columns to Keep & Rename ---
COLUMNS_MAP = {
    # --- Identifiers & Demographics ---
    'VISITS': 'visits', 'POPU_LSA': 'population_lsa',
    'CNTYPOP': 'county_population',
    'GEOCODE': 'geocode', 'C_RELATN': 'interlibrary_relation_code',
    'C_LEGBAS': 'legal_basis_code', 'C_ADMIN': 'admin_structure_code',
    'C_FSCS': 'fscs_definition_code', 'RSTATUS': 'impute_status',

    # --- Collections ---
    'BKVOL': 'print_volumes', 'EBOOK': 'ebook_volumes', 'AUDIO_PH': 'audio_physical',
    'AUDIO_DL': 'audio_digital', 'VIDEO_PH': 'video_physical', 'VIDEO_DL': 'video_digital',
    'ELECCOLL': 'total_e_collection', "TOTPHYS": "tot_physical", "OTHPHYS": "other_physical",

    # --- Staffing ---
    'MASTER': 'mls_librarians_fte', 'LIBRARIA': 'total_staff_fte', 'OTHPAID': 'other_paid_fte',

    # --- Service Metrics ---
    # 'HRS_OPEN': 'hours_open_yearly',
    # 'TOTPRO': 'total_programs',

    # ---- Extra ------
    "CENTLIB": "num_central_lib",
    "BRANLIB": "num_lib_branches",
    "ODFINE": "overdue_policy",
    "OBEREG": "beac_code",
    "LSAGEORATIO": "lsa_to_aligned",
    "LSAGEOTYPE": "geo_type",
    "LOCALE_ADD": "locale_code",
    "MICROF": "metro",
    "BKMOB": "num_bookmobiles"
}

# ============================================================
#  Helper Functions
# ============================================================

def load_and_clean_data(file_path: str, columns_map: dict):
    """Loads, renames, and performs initial cleaning of the library data.
    Returns cleaned DataFrame plus row counts before and after cleaning.
    """
    print("Loading data...")
    try:
        df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
    except FileNotFoundError:
        print(f"\nERROR: Data file not found at '{file_path}'. Please check the path.")
        return pd.DataFrame(), None, None

    # --- Rename relevant columns ---
    existing_cols = [col for col in columns_map.keys() if col in df.columns]
    df_clean = df[existing_cols].rename(columns=columns_map)

    # --- Count before cleaning ---
    n_before = df_clean.shape[0]

    # --- Replace coded missing values with NaN ---
    df_clean.replace([-1, -3, -4, -9], np.nan, inplace=True)

    # --- Keep only valid/imputed entries (RSTATUS 1 or 3) ---
    df_clean = df_clean[df_clean['impute_status'].isin([1, 3])]

    # --- Drop missing/invalid visit or population values ---
    df_clean = df_clean.dropna(subset=['visits', 'population_lsa'])
    df_clean = df_clean[(df_clean['visits'] > 0) & (df_clean['population_lsa'] > 0)]

    # --- Count after cleaning ---
    n_after = df_clean.shape[0]

    # --- Print cleaning summary ---
    print(f"Rows before cleaning: {n_before:,}")
    print(f"Rows after cleaning (RSTATUS 1 or 3, valid visits/pop): {n_after:,}")
    print(f"Removed {n_before - n_after:,} rows ({100*(n_before - n_after)/n_before:.1f}%)")
    print(f"Cleaned data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns\n")

    return df_clean, n_before, n_after


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary target and engineered features."""
    print("Engineering features...")
    df_eng = pd.DataFrame()

    # --- Continuous target for reference ---
    df_eng[TARGET_VARIABLE] = np.log(df['visits'] / df['population_lsa'])

    # --- Binary target ---
    df_eng['visits_per_capita'] = np.exp(df_eng[TARGET_VARIABLE])
    df_eng['target_binary'] = (df_eng['visits_per_capita'] > THRESHOLD).astype(int)

    # --- Per capita log features ---
    per_capita_vars = [
        'print_volumes', 'ebook_volumes', 'audio_physical', 'audio_digital',
        'video_physical', 'video_digital', 'total_e_collection', 'tot_physical',
        'other_physical', 'mls_librarians_fte', 'total_staff_fte', 'other_paid_fte',
        'hours_open_yearly', 'total_programs', 'county_population'
    ]
    for col in per_capita_vars:
        if col in df.columns:
            df_eng[f'log_{col}_per_capita'] = np.log1p(df[col] / df['population_lsa'])

    # --- Counts ---
    count_vars = ['num_central_lib', 'num_lib_branches', 'num_bookmobiles']
    for col in count_vars:
        if col in df.columns:
            df_eng[f'log_{col}'] = df[col]

    # --- Categorical ---
    categorical_features = [
        'geocode', 'interlibrary_relation_code', 'legal_basis_code',
        'admin_structure_code', 'fscs_definition_code', 'overdue_policy',
        'beac_code', 'geo_type', 'locale_code', 'metro'
    ]
    for col in categorical_features:
        if col in df.columns:
            df_eng[col] = df[col].astype('category')

    df_eng.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_eng.dropna(subset=['target_binary'], inplace=True)

    print(f"Feature engineering complete. Model data shape: {df_eng.shape}\n")
    return df_eng


# ============================================================
#  Data Preparation
# ============================================================
df_raw, n_before, n_after = load_and_clean_data(FILE_PATH, COLUMNS_MAP)

if df_raw.empty:
    print("Exiting script because data could not be loaded.")
else:
    print(f"✅ Loaded and cleaned data: {n_before:,} → {n_after:,} rows\n")

    df_model = engineer_features(df_raw)

    X = df_model.drop(columns=[TARGET_VARIABLE, 'visits_per_capita', 'target_binary'])
    y = df_model['target_binary']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=RANDOM_STATE
    )

    print("Dataset split:")
    print(f"  Training set size:    {X_train.shape[0]} ({X_train.shape[0]/len(df_model):.0%})")
    print(f"  Validation set size:  {X_val.shape[0]} ({X_val.shape[0]/len(df_model):.0%})")
    print(f"  Test set size:        {X_test.shape[0]} ({X_test.shape[0]/len(df_model):.0%})\n")

    # ============================================================
    #  Preprocessing Pipeline
    # ============================================================
    print("Defining preprocessing pipeline...")
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='category').columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # ============================================================
    #  Model Training
    # ============================================================
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    param_grids = {
        'Logistic Regression': { 'model__C': np.logspace(-3, 3, 10) },
        'Random Forest': {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [5, 10, None]
        },
        'Gradient Boosting': {
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5]
        }
    }

    results = []
    best_estimators = {}

    for name, model in models.items():
        start_time = time.time()
        print(f"--- Training {name} ---")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grids[name],
            n_iter=10,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0
        )
        search.fit(X_train, y_train)
        best_estimators[name] = search.best_estimator_

        y_pred_val = search.predict(X_val)
        y_proba_val = search.predict_proba(X_val)[:, 1]

        results.append({
            'Model': name,
            'Val Accuracy': accuracy_score(y_val, y_pred_val),
            'Val F1': f1_score(y_val, y_pred_val),
            'Val ROC AUC': roc_auc_score(y_val, y_proba_val),
            'Best Params': search.best_params_
        })
        print(f"Finished training {name} in {time.time() - start_time:.2f} seconds.")

    # ============================================================
    #  Final Evaluation
    # ============================================================
    results_df = pd.DataFrame(results).set_index('Model').sort_values('Val ROC AUC', ascending=False)
    print("\n--- Model Comparison (based on validation set) ---")
    print(results_df.round(4))

    # ============================================================
    #  Confusion Matrices
    # ============================================================
    print("\nGenerating confusion matrices for each model...")
    for name, pipeline in best_estimators.items():
        y_pred_test = pipeline.predict(X_test)
        y_proba_test = pipeline.predict_proba(X_test)[:, 1]

        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
        test_auc = roc_auc_score(y_test, y_proba_test)

        print(f"\n{name} — Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f} | Test ROC AUC: {test_auc:.4f}")
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test)).plot(cmap="Blues")
        plt.title(f"{name} - Confusion Matrix")
        plt.show()

    print("\nBinary classification complete! ✅")


print(n_before, n_after)
