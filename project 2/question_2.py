import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

# !python -m pip install scikit-learn
# --------------------------- Custom Log Transformer ----------------------------
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
            X["population_lsa"] = np.log1p(X["population_lsa"])
        if self.county_population_log:
            X["county_population"] = np.log1p(X["county_population"])
        if self.print_volumes_log:
            X["print_volumes"] = np.log1p(X["print_volumes"])
        if self.ebook_volumes_log:
            X["ebook_volumes"] = np.log1p(X["ebook_volumes"])
        return X


# ----------------------------------------------------
# 1. Load trained Random Forest model
# ----------------------------------------------------
model_path = "Random_Forest__pop-True_county-False_print-False_ebook-False.joblib"
rf_model = joblib.load(model_path)

# ----------------------------------------------------
# 2. Load test data
# ----------------------------------------------------
df_test = pd.read_csv("~/Desktop/Classes/Data Science/project 1/cleaned_data/test_data.csv")

# ----------------------------------------------------
# 3. Prepare features (must match training exactly)
# ----------------------------------------------------
CATEGORICAL_FEATURES = [
    "interlibrary_relation_code", "fscs_definition_code",
    "overdue_policy", "beac_code", "locale_code"
]

LOGGABLE_FEATURES = ["population_lsa", "county_population", "print_volumes", "ebook_volumes"]
FIXED_FEATURES = ["num_lib_branches", "num_bookmobiles"]

all_features = LOGGABLE_FEATURES + FIXED_FEATURES + CATEGORICAL_FEATURES
X_test = df_test[all_features]

# ----------------------------------------------------
# 4. Predict log(visits) and actual visits
# ----------------------------------------------------
pred_log = rf_model.predict(X_test)
pred_visits = np.expm1(pred_log)

df_test["pred_log_visits"] = pred_log
df_test["pred_visits"] = pred_visits

print(df_test.head())

# ----------------------------------------------------
# 5. RMSE on log scale
# ----------------------------------------------------
rmse_log = np.sqrt(mean_squared_error(
    np.log1p(df_test["visits"]),
    df_test["pred_log_visits"]
))
print("RMSE (log scale):", rmse_log)


#!python -m pip install matplotlib
import matplotlib.pyplot as plt

true_visits = df_test["visits"]
pred_visits = df_test["pred_visits"]

plt.figure(figsize=(8, 8))

# Scatter plot
plt.scatter(true_visits, pred_visits, alpha=0.4)

# 45-degree reference line
max_val = max(true_visits.max(), pred_visits.max())
plt.plot([0, max_val], [0, max_val], linestyle="--")

plt.xlabel("True Visits (unlogged)")
plt.ylabel("Predicted Visits (unlogged)")
plt.title("True vs Predicted Visits (Unlogged)")
plt.grid(True)
plt.tight_layout()
plt.show()
