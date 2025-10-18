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
from sklearn.metrics import mean_squared_error, r2_score

# --- Models ---
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
print("Libraries imported successfully.")

# ============================================================
#  Configuration
# ============================================================
# --- File Paths & Data ---
# ‼️ IMPORTANT: Make sure this path is correct for your system
FILE_PATH = "PLS_FY2022 PUD_CSV/PLS_FY22_AE_pud22i.csv"
TARGET_VARIABLE = 'log_visits_per_capita' # TARGET: Log of visits per capita
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
if df_raw.empty:
    print("Exiting script because data could not be loaded.")
else:
    df_model = engineer_features(df_raw)

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

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

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
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(random_state=RANDOM_STATE),
        'Ridge': Ridge(random_state=RANDOM_STATE)
    }

    param_grids = {
        'Linear Regression': {},
        'Lasso': { 'model__alpha': np.logspace(-5, 1, 100) },
        'Ridge': { 'model__alpha': np.logspace(-3, 5, 100) }
    }

    results = []
    best_estimators = {}

    for name, model in models.items():
        start_time = time.time()
        print(f"--- Training {name} ---")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        search = RandomizedSearchCV(
            pipeline,
            param_grids[name],
            n_iter=1 if name == 'Linear Regression' else 20,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0
        )
        search.fit(X_train, y_train)
        best_estimators[name] = search.best_estimator_
        
        y_pred_val = search.predict(X_val)
        results.append({
            'Model': name,
            'Val R²': r2_score(y_val, y_pred_val),
            'Val RMSE': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'Best Alpha': search.best_params_.get('model__alpha', 'N/A')
        })
        print(f"Finished training {name} in {time.time() - start_time:.2f} seconds.")

    # ============================================================
    #  Final Evaluation
    # ============================================================
    results_df = pd.DataFrame(results).set_index('Model').sort_values('Val R²', ascending=False)
    print("\n--- Model Comparison (based on validation set) ---")
    print(results_df[['Val R²', 'Val RMSE', 'Best Alpha']].round(4))

    # ============================================================
    #  Plotting Actual vs. Predicted for Each Model
    # ============================================================
    print("\nGenerating Actual vs. Predicted plots for each model...")
    sns.set_theme(style="whitegrid", palette="muted")

    for name, pipeline in best_estimators.items():
        # Make predictions on the final, unseen test data
        y_pred_test = pipeline.predict(X_test)
        
        # Calculate performance metrics for the title
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Create the plot
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6, s=50, edgecolor='k')
        
        # Add the ideal fit line
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2, label='Ideal Fit')
        
        plt.title(f'Actual vs. Predicted - {name}\nTest R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}', fontsize=16, fontweight='bold')
        plt.xlabel('Actual ' + TARGET_VARIABLE, fontsize=12)
        plt.ylabel('Predicted ' + TARGET_VARIABLE, fontsize=12)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    print("\nAnalysis complete! ✨")
