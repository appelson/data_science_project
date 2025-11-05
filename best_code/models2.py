#===============================================================================
# LIBRARIES
#===============================================================================
import itertools
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
import joblib

np.random.seed(1)

#===============================================================================
# CONFIGURATION
#===============================================================================
CATEGORICAL_FEATURES = [
    'interlibrary_relation_code', 'fscs_definition_code',
    'overdue_policy', 'beac_code', 'locale_code'
]
LOGGABLE_FEATURES = ['population_lsa', 'county_population', 'print_volumes', 'ebook_volumes']
FIXED_FEATURES = ['num_lib_branches', 'num_bookmobiles']

CV_FOLDS = 1
RANDOM_ITER = 1

#===============================================================================
# LOAD DATA
#===============================================================================
df = pd.read_csv("cleaned_data/train_data.csv")
df['log_visits'] = np.log(df['visits'] + 1)

for feature in LOGGABLE_FEATURES:
    df[f'log_{feature}'] = np.log(df[feature] + 1)

numeric_features = LOGGABLE_FEATURES + FIXED_FEATURES
all_features = numeric_features + CATEGORICAL_FEATURES
X, y = df[all_features], df['log_visits']
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=1)

#===============================================================================
# TRANSFORMER & PREPROCESSOR
#===============================================================================
class ConditionalLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, population_lsa_log=True, county_population_log=True,
                 print_volumes_log=True, ebook_volumes_log=True):
        self.population_lsa_log = population_lsa_log
        self.county_population_log = county_population_log
        self.print_volumes_log = print_volumes_log
        self.ebook_volumes_log = ebook_volumes_log

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.population_lsa_log:
            X['population_lsa'] = np.log(X['population_lsa'] + 1)
        if self.county_population_log:
            X['county_population'] = np.log(X['county_population'] + 1)
        if self.print_volumes_log:
            X['print_volumes'] = np.log(X['print_volumes'] + 1)
        if self.ebook_volumes_log:
            X['ebook_volumes'] = np.log(X['ebook_volumes'] + 1)
        return X


preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
     CATEGORICAL_FEATURES)
])

#===============================================================================
# HELPER FUNCTIONS
#===============================================================================
def evaluate_model_for_logs(model_name, base_pipeline, param_dist):
    """
    Runs RandomizedSearchCV for all 16 combinations of log transformations.
    """
    best_config = None
    best_r2 = -np.inf
    best_result = None

    bools = [True, False]
    combinations = list(itertools.product(bools, repeat=4))

    for (pop_log, county_log, print_log, ebook_log) in combinations:
        config_name = f"pop={pop_log}, county={county_log}, print={print_log}, ebook={ebook_log}"
        print(f"\n{'-'*80}\n{model_name}: Testing log config -> {config_name}\n{'-'*80}")

        pipeline = clone(base_pipeline)
        pipeline.set_params(
            log_transformer__population_lsa_log=pop_log,
            log_transformer__county_population_log=county_log,
            log_transformer__print_volumes_log=print_log,
            log_transformer__ebook_volumes_log=ebook_log
        )

        search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist, n_iter=RANDOM_ITER,
            cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1,
            random_state=1, verbose=0
        )
        search.fit(X, y)

        best_pipeline = search.best_estimator_
        rmse_scores, r2_scores = [], []
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = clone(best_pipeline)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            r2_scores.append(r2_score(y_val, y_pred))

        cv_rmse, cv_r2 = np.mean(rmse_scores), np.mean(r2_scores)

        print(f"CV RMSE: {cv_rmse:.4f} | CV R²: {cv_r2:.4f}")

        if cv_r2 > best_r2:
            best_r2 = cv_r2
            best_config = config_name
            best_result = {
                'model': model_name,
                'cv_rmse': cv_rmse,
                'cv_r2': cv_r2,
                'best_params': search.best_params_,
                'best_pipeline': best_pipeline,
                'log_config': config_name
            }

    print(f"\n>>> Best log config for {model_name}: {best_config}")
    return best_result


#===============================================================================
# MODEL 1: OLS
#===============================================================================
ols_pipe = Pipeline([
    ('log_transformer', ConditionalLogTransformer()),
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

ols_params = {}  # no hyperparameters to tune
ols_result = evaluate_model_for_logs("OLS", ols_pipe, ols_params)

#===============================================================================
# MODEL 2: LASSO
#===============================================================================
lasso_pipe = Pipeline([
    ('log_transformer', ConditionalLogTransformer()),
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', Lasso(max_iter=10000, random_state=1))
])

lasso_params = {'regressor__alpha': loguniform(1e-6, 50)}
lasso_result = evaluate_model_for_logs("LASSO", lasso_pipe, lasso_params)

#===============================================================================
# MODEL 3: RIDGE
#===============================================================================
ridge_pipe = Pipeline([
    ('log_transformer', ConditionalLogTransformer()),
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(random_state=1))
])

ridge_params = {'regressor__alpha': loguniform(1e-6, 1e4)}
ridge_result = evaluate_model_for_logs("RIDGE", ridge_pipe, ridge_params)

#===============================================================================
# MODEL 4: DECISION TREE
#===============================================================================
tree_pipe = Pipeline([
    ('log_transformer', ConditionalLogTransformer()),
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=1))
])

tree_params = {
    'regressor__max_depth': randint(3, 50),
    'regressor__min_samples_split': randint(2, 50),
    'regressor__min_samples_leaf': randint(1, 20),
    'regressor__min_impurity_decrease': loguniform(1e-8, 1e-2)
}
tree_result = evaluate_model_for_logs("Decision Tree", tree_pipe, tree_params)

#===============================================================================
# MODEL 5: RANDOM FOREST
#===============================================================================
rf_pipe = Pipeline([
    ('log_transformer', ConditionalLogTransformer()),
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=1, n_jobs=-1))
])

rf_params = {
    'regressor__n_estimators': randint(200, 1000),
    'regressor__max_depth': randint(10, 60),
    'regressor__min_samples_split': randint(2, 30),
    'regressor__min_samples_leaf': randint(1, 15),
    'regressor__max_features': uniform(0.2, 0.6),
    'regressor__min_impurity_decrease': loguniform(1e-8, 1e-2)
}
rf_result = evaluate_model_for_logs("Random Forest", rf_pipe, rf_params)

#===============================================================================
# SUMMARY
#===============================================================================
all_results = [
    ols_result, lasso_result, ridge_result,
    tree_result, rf_result
]

results_df = pd.DataFrame([
    {k: v for k, v in res.items() if k != 'best_pipeline'}
    for res in all_results
])

import os
os.makedirs("saved_models", exist_ok=True)

for res in all_results:
    model_name = res['model'].replace(" ", "_")
    log_config = res['log_config'].replace(", ", "_").replace("=", "-")
    filename = f"saved_models/{model_name}__{log_config}.joblib"
    joblib.dump(res['best_pipeline'], filename)
    print(f"saved: {filename}")
