#----------------------------- Loading Libraries -------------------------------

# Loading Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.tree import plot_tree
import joblib

# Setting random seed
np.random.seed(1)

#------------------------------- Load and Prepare Data -------------------------------

# Loading data
df_clean = pd.read_csv("cleaned_data/cleaned_data.csv")

# Transforming predictors into LOG versions
df_clean['log_visits'] = np.log(df_clean['visits'])
df_clean['log_population_lsa'] = np.log(df_clean['population_lsa'] + 1)
df_clean['log_county_population'] = np.log(df_clean['county_population'] + 1)
df_clean['log_print_volumes'] = np.log(df_clean['print_volumes'] + 1)
df_clean['log_ebook_volumes'] = np.log(df_clean['ebook_volumes'] + 1)

# Defining numeric variables
numeric_features = [
    'log_population_lsa', 'log_county_population', 
    'log_print_volumes', 'log_ebook_volumes',
    'num_lib_branches', 'num_bookmobiles'
]

# Defining categorical variables
categorical_features = [
    'interlibrary_relation_code', 'fscs_definition_code',
    'overdue_policy', 'beac_code', 'locale_code'
]

# Defining all variables
all_features = numeric_features + categorical_features

# Defining covariates
X = df_clean[all_features]

# Defining outcome variable
y = df_clean['log_visits']

#------------------------------- Preprocessing Pipeline ------------------------

# One-hot encoding the categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
         categorical_features)
    ]
)

#------------------------------- Train/Test Split ------------------------------

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}\n")

#------------------------------- Cross-Validation Setup ------------------------

# Defining CV
cv = KFold(n_splits=10, shuffle=True, random_state=1)

# Storing results
results = []
trained_models = {}

#------------------------------- Helper Function -------------------------------

# Creating helper function
def evaluate_model(name, pipeline, param_grid, X_train, y_train, cv, feature_subset=None):
    print(f"\nTraining: {name}")
    
    # Extracting feature subset if specified
    if feature_subset is not None:
        X_train_subset = X_train[feature_subset].copy()
    else:
        X_train_subset = X_train
    
    # Defining grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    # Fitting on training data
    grid_search.fit(X_train_subset, y_train)
    
    # Getting CV score from grid search
    cv_rmse = -grid_search.best_score_
    
    # Calculating R^2 on CV
    cv_r2_scores = []
    
    # Fitting model on data and calculating R^2
    for train_idx, val_idx in cv.split(X_train_subset):
        X_fold_train = X_train_subset.iloc[train_idx]
        X_fold_val = X_train_subset.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_model = clone(grid_search.best_estimator_)
        fold_model.fit(X_fold_train, y_fold_train)
        y_fold_pred = fold_model.predict(X_fold_val)
        fold_r2 = r2_score(y_fold_val, y_fold_pred)
        cv_r2_scores.append(fold_r2)
    
    # Getting mean R^2
    cv_r2 = np.mean(cv_r2_scores)
    
    print(f"  CV RMSE: {cv_rmse:.4f} | CV R²: {cv_r2:.4f}")
    
    # Returning name, RMSE, R^2, and Best Parameters
    return {
        'Model': name,
        'CV_RMSE': cv_rmse,
        'CV_R2': cv_r2,
        'Best_Params': grid_search.best_params_
    }, grid_search.best_estimator_

#------------------------------- Mean Baseline --------------------------------

print("\nTraining: Mean Baseline")

# Calculating CV mean 
cv_scores = []
for train_idx, val_idx in cv.split(X_train):
    y_train_fold = y_train.iloc[train_idx]
    y_val_fold = y_train.iloc[val_idx]
    fold_mean = np.mean(y_train_fold)
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, [fold_mean]*len(y_val_fold)))
    cv_scores.append(fold_rmse)

mean_cv_rmse = np.mean(cv_scores)

# Appending results
results.append({
    'Model': 'Mean Baseline',
    'CV_RMSE': mean_cv_rmse,
    'CV_R2': 0.0,
    'Best_Params': {}
})

print(f"  CV RMSE: {mean_cv_rmse:.4f} | CV R²: 0.0000")

# Storing model
trained_models['mean_baseline'] = np.mean(y_train)

#------------------------------- Univariate Population Model -------------------------------

# Creating linear regression pipe (univariate)
pipe_univ = Pipeline([
    ('regressor', LinearRegression())
])

# Evaluating model
result, model = evaluate_model(
    'Univariate (Pop)',
    pipe_univ,
    {},
    X_train, y_train,
    cv,
    feature_subset=['log_population_lsa']
)

# Appending results
results.append(result)

# Storing model
trained_models['univariate'] = model

#------------------------------- OLS Regression -------------------------------

# Creating linear regression pipe (full)
pipe_ols = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Evaluating model
result, model = evaluate_model(
    'OLS',
    pipe_ols,
    {},
    X_train, y_train,
    cv
)

# Appending results
results.append(result)

# Storing model
trained_models['ols'] = model

#------------------------------- LASSO -------------------------------

# Creating LASSO pipe
pipe_lasso = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', Lasso(max_iter=10000, random_state=123))
])

# Defining LASSO parameters
lasso_params = {
    'regressor__alpha': np.logspace(-3, 1, 100)
}

# Evaluating model
result, model = evaluate_model(
    'LASSO',
    pipe_lasso,
    lasso_params,
    X_train, y_train,
    cv
)

# Appending results
results.append(result)

# Storing model
trained_models['lasso'] = model

#------------------------------- Ridge -------------------------------

# Creating Ridge pipe
pipe_ridge = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(random_state=123))
])

# Defining Ridge parameters
ridge_params = {
    'regressor__alpha': np.logspace(-3, 1, 100)
}

# Evaluating model
result, model = evaluate_model(
    'Ridge',
    pipe_ridge,
    ridge_params,
    X_train, y_train,
    cv
)

# Appending results
results.append(result)

# Storing model
trained_models['ridge'] = model

#------------------------------- Decision Tree -------------------------------

# Creating Decision Tree pipe
pipe_tree = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=123))
])

# Defining Decision Tree parameters
tree_params = {
    'regressor__max_depth': [3, 5, 7, 10, 15, 20, None],
    'regressor__min_samples_split': [2, 5, 10, 20],
    'regressor__min_samples_leaf': [1, 2, 4, 8]
}

# Evaluating model
result, model = evaluate_model(
    'Decision Tree',
    pipe_tree,
    tree_params,
    X_train, y_train,
    cv
)

# Appending results
results.append(result)

# Storing model
trained_models['tree'] = model

#------------------------------- Random Forest -------------------------------

# Creating Random Forest pipe
pipe_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=123, n_jobs=-1))
])

# Defining Random Forest parameters
rf_params = {
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 10],
    'regressor__min_samples_leaf': [1, 4],
    'regressor__max_features': ['sqrt', 0.5]
}

# Evaluating model
result, model = evaluate_model(
    'Random Forest',
    pipe_rf,
    rf_params,
    X_train, y_train,
    cv
)

# Appending results
results.append(result)

# Storing model
trained_models['random_forest'] = model

#------------------------------- Gradient Boosting -------------------------------

# Creating Gradient Boosting pipe
pipe_gbm = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=123))
])

# Defining Gradient Boosting parameters
gbm_params = {
    'regressor__n_estimators': [500, 1000],
    'regressor__learning_rate': [0.01, 0.1],
    'regressor__max_depth': [3, 5],
    'regressor__min_samples_split': [2, 10],
    'regressor__subsample': [0.8, 1.0]
}

# Evaluating model
result, model = evaluate_model(
    'Gradient Boosting',
    pipe_gbm,
    gbm_params,
    X_train, y_train,
    cv
)

# Appending results
results.append(result)

# Storing model
trained_models['gbm'] = model

#------------------------------- Neural Network -------------------------------

# Creating Neural Network pipe
pipe_nn = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', MLPRegressor(max_iter=1000, random_state=123, early_stopping=True))
])

# Defining Neural Network parameters
nn_params = {
    'regressor__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'regressor__alpha': [0.0001, 0.001, 0.01, 0.1],
    'regressor__learning_rate_init': [0.001, 0.01]
}

# Evaluating model
result, model = evaluate_model(
    'Neural Network',
    pipe_nn,
    nn_params,
    X_train, y_train,
    cv
)

# Appending results
results.append(result)

# Storing model
trained_models['neural_network'] = model

#------------------------------- Results Summary -------------------------------

# Creating results dataframe
results_df = pd.DataFrame(results)

# Sorting by CV RMSE
results_df = results_df.sort_values('CV_RMSE')
print(results_df[['Model', 'CV_RMSE', 'CV_R2']].to_string(index=False))

results_df["Best_Params"][7]
# Getting best model
best_model_name = results_df.iloc[0]['Model']
best_cv_rmse = results_df.iloc[0]['CV_RMSE']
best_cv_r2 = results_df.iloc[0]['CV_R2']

print(f"\nBest Model: {best_model_name}")
print(f"CV RMSE: {best_cv_rmse:.4f}")
print(f"CV R²: {best_cv_r2:.4f}")
#-------------------- Residual Plots for Each Model ----------------------------

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

model_order = [
    ('mean_baseline', 'Mean Baseline'),
    ('univariate', 'Univariate (Pop)'),
    ('ols', 'OLS'),
    ('lasso', 'LASSO'),
    ('ridge', 'Ridge'),
    ('tree', 'Decision Tree'),
    ('random_forest', 'Random Forest'),
    ('gbm', 'Gradient Boosting'),
    # ('neural_network', 'Neural Network'),  # optional
]

feature_subsets = {
    'mean_baseline': None,
    'univariate': ['log_population_lsa'],
    'ols': all_features,
    'lasso': all_features,
    'ridge': all_features,
    'tree': all_features,
    'random_forest': all_features,
    'gbm': all_features,
    'neural_network': all_features,
}

for i, (key, name) in enumerate(model_order):
    ax = axes[i]

    # Select features for this model
    if feature_subsets[key] is not None:
        X_test_subset = X_test[feature_subsets[key]]
    else:
        X_test_subset = X_test

    # Predictions
    if key == 'mean_baseline':
        y_pred = np.repeat(trained_models[key], len(y_test))
    else:
        y_pred = trained_models[key].predict(X_test_subset)

    # Compute residuals
    residuals = y_test - y_pred

    # Scatter residuals vs predicted
    ax.scatter(y_pred, residuals, alpha=0.3, edgecolor='k')
    ax.axhline(0, color='r', linestyle='--', lw=2)

    # Axis limits
    ax.set_xlabel("Predicted log(visits)")
    ax.set_ylabel("Residuals")
    ax.set_title(name)
    ax.set_xlim([y_pred.min(), y_pred.max()])

plt.tight_layout()
plt.show()
