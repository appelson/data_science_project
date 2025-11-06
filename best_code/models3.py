#----------------------------- Loading Libraries -------------------------------

# Loading Libraries
import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor

# Setting the seed
np.random.seed(1)

# ----------------------------- Configuration ----------------------------------
CATEGORICAL_FEATURES = [
    'interlibrary_relation_code', 'fscs_definition_code',
    'overdue_policy', 'beac_code', 'locale_code'
]
LOGGABLE_FEATURES = ['population_lsa', 'county_population', 'print_volumes', 'ebook_volumes']
FIXED_FEATURES = ['num_lib_branches', 'num_bookmobiles']

CV_FOLDS = 10
RANDOM_ITER = 50

#--------------------------- Custom Log Transformer ----------------------------

# Optionally applies log(x + 1) to each of True/False flags.
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
            X['population_lsa'] = np.log1p(X['population_lsa'])
        if self.county_population_log:
            X['county_population'] = np.log1p(X['county_population'])
        if self.print_volumes_log:
            X['print_volumes'] = np.log1p(X['print_volumes'])
        if self.ebook_volumes_log:
            X['ebook_volumes'] = np.log1p(X['ebook_volumes'])
        return X

# ------------------------ Loadings + Preps Data ------------------------------

# Loading training data
df = pd.read_csv("cleaned_data/train_data.csv")

# Getting log visits
df['log_visits'] = np.log1p(df['visits'])

# Preparing all numerical features
numeric_features = LOGGABLE_FEATURES + FIXED_FEATURES

# Defining all features
all_features = numeric_features + CATEGORICAL_FEATURES
X, y = df[all_features], df['log_visits']

# Defining k fold cv 
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=1)

# Defining preprocessor to let numerical features through but one-hot encode categorical
preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
     CATEGORICAL_FEATURES)
])


#------------------------------- Helper Function -------------------------------

# Defining an evaluation function
#------------------------------- Helper Function -------------------------------

# Defining an evaluation function
def evaluate_model_for_logs(model_name, base_pipeline, param_dist):

    best_config = None
    best_rmse = np.inf
    best_result = None
    
    bools = [True, False]
    combinations = list(itertools.product(bools, repeat=4))
    
    # Runs through all 2^4 combinations of True/False for log(x + 1)
    for (pop_log, county_log, print_log, ebook_log) in combinations:
        config_name = f"pop={pop_log}, county={county_log}, print={print_log}, ebook={ebook_log}"

        
        pipeline = clone(base_pipeline)
        pipeline.set_params(
            log_transformer__population_lsa_log=pop_log,
            log_transformer__county_population_log=county_log,
            log_transformer__print_volumes_log=print_log,
            log_transformer__ebook_volumes_log=ebook_log
        )
        
        # Within each of the 16 combinations, runs randomized search CV
        search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist, 
            n_iter=RANDOM_ITER,
            cv=cv, 
            scoring='neg_root_mean_squared_error', 
            n_jobs=-1,
            random_state=1, 
            verbose=0
        )
        
        # Fitting the model on the CV
        search.fit(X, y)
        
        # Getting CV RMSE from RandomizedSearchCV (flip sign since it's negative)
        cv_rmse = -search.best_score_
        
        # Defining the best CV RMSE based on 2^4 combinations 
        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_config = config_name
            best_result = {
                'model': model_name,
                'cv_rmse': cv_rmse,
                'best_params': search.best_params_,
                'best_pipeline': search.best_estimator_,
                'log_config': config_name
            }
    
    # Outputting best model (only a single model), the model with best params and 
    # the best variable transformations
    
    print(f"Best transformations for {model_name}: {best_config}")
    return best_result

# -------------------------- Baseline Models ----------------------------------

# Getting single population LSA variable
uni_X = df[['population_lsa']]
uni_y = df['log_visits']

##  ---------------------------- Baseline 1 -------------------------------

# Baseline mean model (note that the mean may be skewed)
global_mean = uni_y.mean()

# Calculating CV RMSE scores for global mean
rmse_scores = []
for _, val_idx in cv.split(uni_X):
    y_val = uni_y.iloc[val_idx]
    y_pred = np.repeat(global_mean, len(y_val))
    rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
mean_rmse = np.mean(rmse_scores)
print(f"Global Mean model: CV RMSE: {mean_rmse:.4f}")

##  ---------------------------- Baseline 2 -------------------------------

# Univariate population model
rmse_scores = []
for train_idx, val_idx in cv.split(uni_X):
    X_train, X_val = uni_X.iloc[train_idx], uni_X.iloc[val_idx]
    y_train, y_val = uni_y.iloc[train_idx], uni_y.iloc[val_idx]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
pop_rmse = np.mean(rmse_scores)
print(f"Univariate population model: CV RMSE: {pop_rmse:.4f}")

##  ---------------------------- Baseline 3 -------------------------------

# Getting log of population lsa
X_logpop = np.log(df[['population_lsa']] + 1)
rmse_scores = []
for train_idx, val_idx in cv.split(X_logpop):
    X_train, X_val = X_logpop.iloc[train_idx], X_logpop.iloc[val_idx]
    y_train, y_val = uni_y.iloc[train_idx], uni_y.iloc[val_idx]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
logpop_rmse = np.mean(rmse_scores)
print(f"Univariate log(population) model: CV RMSE: {logpop_rmse:.4f}")

# ------------------------- Fitting Better Models -----------------------------

# OLS
ols_pipe = Pipeline([
    ('log_transformer', ConditionalLogTransformer()),
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
ols_params = {}
ols_result = evaluate_model_for_logs("OLS", ols_pipe, ols_params)

# LASSO
lasso_pipe = Pipeline([
    ('log_transformer', ConditionalLogTransformer()),
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', Lasso(random_state=1))
])
lasso_params = {'regressor__alpha': loguniform(1e-6, 50)}
lasso_result = evaluate_model_for_logs("LASSO", lasso_pipe, lasso_params)

# Ridge
ridge_pipe = Pipeline([
    ('log_transformer', ConditionalLogTransformer()),
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(random_state=1))
])
ridge_params = {'regressor__alpha': loguniform(1e-6, 50)}
ridge_result = evaluate_model_for_logs("RIDGE", ridge_pipe, ridge_params)

# Decision Tree
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

# RF
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

# ------------------------ Getting Model Results -------------------------------

# All Results
all_results = [
    ols_result, lasso_result, ridge_result, tree_result, 
    rf_result, gb_result, mlp_result
]

# Turning results into a dataframe
results_df = pd.DataFrame([
    {k: v for k, v in res.items() if k != 'best_pipeline'}
    for res in all_results
])

# Creating a save models director
os.makedirs("saved_models", exist_ok=True)

# Saving the models in the directory
for res in all_results:
    model_name = res['model'].replace(" ", "_")
    log_config = res['log_config'].replace(", ", "_").replace("=", "-")
    filename = f"saved_models/{model_name}__{log_config}.joblib"
    joblib.dump(res['best_pipeline'], filename)

# Saving the fit summary
results_df.to_csv("saved_models/model_summary.csv", index=False)
