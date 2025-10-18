
# ============================================================
#  Libraries
# ============================================================
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# --- Models ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

print("Libraries imported successfully.")

# ============================================================
#  Configuration
# ============================================================
# --- File Paths & Data ---
FILE_PATH = "PLS_FY2022 PUD_CSV/PLS_FY22_AE_pud22i.csv"
TARGET_VARIABLE = 'log_visits_per_capita' # TARGET: Log of visits per capita
RANDOM_STATE = 123

# --- Columns to Keep & Rename ---
# Using the comprehensive list of variables you provided
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

    # # --- Funding & Expenditures ---
    # 'LOCGVT': 'revenue_local', 'STGVT': 'revenue_state', 'FEDGVT': 'revenue_federal',
    # 'OTHINCM': 'revenue_other', 'STAFFEXP': 'exp_staff', 'TOTEXPCO': 'exp_collection',
    # 'CAP_REV': 'capital_revenue',

    # --- Service Metrics ---
    'HRS_OPEN': 'hours_open_yearly', 
    'TOTPRO': 'total_programs',
    
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
def load_and_clean_data(file_path: str, columns_map: dict) -> pd.DataFrame:
    """Loads, renames, and performs initial cleaning of the library data."""
    print("Loading data...")
    try:
        df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
    except FileNotFoundError:
        print(f"\nERROR: Data file not found at '{file_path}'. Please check the path.")
        return pd.DataFrame()

    existing_cols = [col for col in columns_map.keys() if col in df.columns]
    df_clean = df[existing_cols].rename(columns=columns_map)

    missing_codes = [-1, -3, -4, -9]
    df_clean.replace(missing_codes, np.nan, inplace=True)

    df_clean = df_clean[df_clean['impute_status'].isin([1, 3])]
    df_clean = df_clean.dropna(subset=['visits', 'population_lsa'])
    df_clean = df_clean[(df_clean['visits'] > 0) & (df_clean['population_lsa'] > 0)]

    print(f"Cleaned data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    return df_clean

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates target variable and other predictive features."""
    print("Engineering features...")
    df_eng = pd.DataFrame()

    df_eng[TARGET_VARIABLE] = np.log(df['visits'] / df['population_lsa'])

    per_capita_vars = [
        'print_volumes', 'ebook_volumes', 'audio_physical', 'audio_digital',
        'video_physical', 'video_digital', 'total_e_collection', 'tot_physical',
        'other_physical', 'mls_librarians_fte', 'total_staff_fte', 'other_paid_fte',
        'hours_open_yearly', 'total_programs', 'county_population'
    ]
    for col in per_capita_vars:
        if col in df.columns:
            df_eng[f'log_{col}_per_capita'] = np.log1p(df[col] / df['population_lsa'])

    count_vars = ['num_central_lib', 'num_lib_branches', 'num_bookmobiles']
    for col in count_vars:
        if col in df.columns:
            df_eng[f'log_{col}'] = df[col]

    if 'lsa_to_aligned' in df.columns:
        df_eng['lsa_to_aligned'] = df['lsa_to_aligned']
    
    categorical_features = [
        'geocode', 'interlibrary_relation_code', 'legal_basis_code',
        'admin_structure_code', 'fscs_definition_code', 'overdue_policy',
        'beac_code', 'geo_type', 'locale_code', 'metro'
    ]
    for col in categorical_features:
        if col in df.columns:
            df_eng[col] = df[col].astype('category')

    df_eng.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_eng.dropna(subset=[TARGET_VARIABLE], inplace=True)

    print(f"Feature engineering complete. Model data shape: {df_eng.shape}\n")
    return df_eng

# ============================================================
#  Data Preparation
# ============================================================
df_raw = load_and_clean_data(FILE_PATH, COLUMNS_MAP)
df_model = engineer_features(df_raw)

# --- Train, Validate, Test Split ---
X = df_model.drop(TARGET_VARIABLE, axis=1)
y = df_model[TARGET_VARIABLE]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE)

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

# Define transformers for numeric and categorical data
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

# Create the preprocessor
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
# --- Define the models to be trained ---
models = {
    'Random Forest': RandomForestRegressor(random_state=RANDOM_STATE),
    'SVR': SVR(),
    'MLP Regressor': MLPRegressor(random_state=RANDOM_STATE, max_iter=1000, early_stopping=True)
}

# --- Define hyperparameter search spaces for each model ---
param_grids = {
    'Random Forest': {
        'model__n_estimators': [100, 300, 500],
        'model__max_depth': [10, 20, None],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 1.0]
    },
    'SVR': {
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': ['scale', 'auto'],
        'model__kernel': ['rbf', 'poly', 'linear']
    },
    'MLP Regressor': {
        'model__hidden_layer_sizes': [(50, 50), (100,), (100, 50, 25)],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': [0.0001, 0.001, 0.01], # L2 penalty
        'model__learning_rate': ['constant', 'adaptive']
    }
}

# --- Training loop ---
results = []
best_estimators = {}

for name, model in models.items():
    start_time = time.time()
    print(f"--- Training {name} ---")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    # Use RandomizedSearchCV to find the best hyperparameters
    search = RandomizedSearchCV(
        pipeline,
        param_grids[name],
        n_iter=20,  # Number of parameter settings that are sampled
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    search.fit(X_train, y_train)
    best_estimators[name] = search.best_estimator_

    # Evaluate on training and validation sets
    y_pred_train = search.predict(X_train)
    y_pred_val = search.predict(X_val)

    results.append({
        'Model': name,
        'Train R²': r2_score(y_train, y_pred_train),
        'Val R²': r2_score(y_val, y_pred_val),
        'Val RMSE': np.sqrt(mean_squared_error(y_val, y_pred_val)),
        'Best Params': search.best_params_
    })
    print(f"Finished training {name} in {time.time() - start_time:.2f} seconds.\n")

# ============================================================
#  Final Evaluation & Plotting
# ============================================================
results_df = pd.DataFrame(results).set_index('Model').sort_values('Val R²', ascending=False)
best_model_name = results_df.index[0]
final_model_pipeline = best_estimators[best_model_name]

print(f"\n🏆 Best performing model on validation set: {best_model_name}")
print(f"   Best hyperparameters: {results_df.loc[best_model_name, 'Best Params']}")

# --- Refit the best model on the combined training and validation data ---
final_model_pipeline.fit(X_train_val, y_train_val)

# --- Evaluate the final model on the unseen test set ---
y_pred_test = final_model_pipeline.predict(X_test)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n--- Model Comparison (based on validation set) ---")
print(results_df[['Train R²', 'Val R²', 'Val RMSE']].round(4))
print(f"\n--- Final Performance of {best_model_name} on Unseen Test Data ---")
print(f"  Test R²:  {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}")

# --- Plotting ---
print("\nGenerating plots for the best model...")
sns.set_theme(style="whitegrid", palette="viridis")

# Plot 1: Actual vs. Predicted
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6, s=50, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2, label='Ideal Fit')
plt.title(f'Actual vs. Predicted - {best_model_name} (Test Set)', fontsize=16, fontweight='bold')
plt.xlabel('Actual ' + TARGET_VARIABLE, fontsize=12)
plt.ylabel('Predicted ' + TARGET_VARIABLE, fontsize=12)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Plot 2: Feature Importance
try:
    ohe_cols = final_model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(ohe_cols)
    model_step = final_model_pipeline.named_steps['model']
    
    importances = None
    if hasattr(model_step, 'feature_importances_'): # For RandomForest
        importances = model_step.feature_importances_
    elif hasattr(model_step, 'coef_'): # For SVR with linear kernel
        importances = model_step.coef_[0]
    else:
        # SVR (rbf, poly) and MLPRegressor don't have direct importance attributes
        print(f"Note: Feature importance plot is not available for '{best_model_name}' with its current parameters.")

    if importances is not None:
        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
        feature_importance_df['abs_importance'] = feature_importance_df['importance'].abs()
        top_features = feature_importance_df.sort_values('abs_importance', ascending=False).head(20)

        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
        plt.title(f'Top 20 Features for {best_model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Importance / Coefficient', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"Could not generate feature importance plot. Reason: {e}")

print("\nAnalysis complete! ✨")
