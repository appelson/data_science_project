# ---------------------------------------------------
# Libraries
# ---------------------------------------------------
library(tidyverse)
library(glmnetUtils)
library(pROC)
library(caret)

set.seed(123)  # reproducibility

# ---------------------------------------------------
# Data
# ---------------------------------------------------
df_clean <- read_csv("cleaned_data/cleaned_data.csv") %>%
  mutate(
    visits_per_capita_binary = as.factor(visits_per_capita_binary)
  ) %>%
  select(-visits_per_capita)

# ---------------------------------------------------
# Define formula
# ---------------------------------------------------
formula_full <- visits_per_capita_binary ~ 
  log(print_volumes + 1) +
  log(ebook_volumes + 1) + 
  log(audio_physical + 1) + 
  log(audio_digital + 1) + 
  log(video_physical + 1) + 
  log(video_digital + 1) + 
  log(total_e_collection + 1) + 
  log(tot_physical + 1) + 
  log(other_physical + 1) + 
  log(mls_librarians_fte + 1) + 
  log(total_staff_fte + 1) + 
  log(other_paid_fte + 1) + 
  geocode +
  interlibrary_relation_code + 
  legal_basis_code + 
  admin_structure_code + 
  fscs_definition_code + 
  overdue_policy + 
  beac_code + 
  geo_type + 
  locale_code + 
  metro + 
  num_central_lib + 
  num_lib_branches + 
  num_bookmobiles

# ---------------------------------------------------
# Train / Test Split
# ---------------------------------------------------
set.seed(123)
train_index <- createDataPartition(df_clean$visits_per_capita_binary, p = 0.8, list = FALSE)
train <- df_clean[train_index, ]
test  <- df_clean[-train_index, ]

# ---------------------------------------------------
# Logistic Regression (Baseline)
# ---------------------------------------------------
model_logit <- glm(formula_full, data = train, family = binomial)
summary(model_logit)

test_prob_logit <- predict(model_logit, newdata = test, type = "response")
test_pred_logit <- ifelse(test_prob_logit > 0.5, 1, 0)

cat("\n--- Confusion Matrix: Logistic Regression ---\n")
print(table(Predicted = test_pred_logit, Actual = as.numeric(test$visits_per_capita_binary) - 1))

roc_logit <- roc(as.numeric(test$visits_per_capita_binary) - 1, test_prob_logit)
auc_logit <- auc(roc_logit)
cat("AUC (Logistic):", round(auc_logit, 3), "\n")

# ---------------------------------------------------
# LASSO Logistic Regression (alpha = 1)
# ---------------------------------------------------
set.seed(123)
model_lasso <- cv.glmnet(
  formula_full, data = train,
  family = "binomial", alpha = 1
)
lasso_lambda <- model_lasso$lambda.min

test_prob_lasso <- predict(model_lasso, newdata = test, s = "lambda.min", type = "response")
test_pred_lasso <- ifelse(test_prob_lasso > 0.5, 1, 0)

cat("\n--- Confusion Matrix: LASSO Logistic ---\n")
print(table(Predicted = test_pred_lasso, Actual = as.numeric(test$visits_per_capita_binary) - 1))

roc_lasso <- roc(as.numeric(test$visits_per_capita_binary) - 1, test_prob_lasso)
auc_lasso <- auc(roc_lasso)
cat("AUC (LASSO):", round(auc_lasso, 3), "\n")

# ---------------------------------------------------
# Ridge Logistic Regression (alpha = 0)
# ---------------------------------------------------
set.seed(123)
model_ridge <- cv.glmnet(
  formula_full, data = train,
  family = "binomial", alpha = 0
)
ridge_lambda <- model_ridge$lambda.min

test_prob_ridge <- predict(model_ridge, newdata = test, s = "lambda.min", type = "response")
test_pred_ridge <- ifelse(test_prob_ridge > 0.5, 1, 0)

cat("\n--- Confusion Matrix: Ridge Logistic ---\n")
print(table(Predicted = test_pred_ridge, Actual = as.numeric(test$visits_per_capita_binary) - 1))

roc_ridge <- roc(as.numeric(test$visits_per_capita_binary) - 1, test_prob_ridge)
auc_ridge <- auc(roc_ridge)
cat("AUC (Ridge):", round(auc_ridge, 3), "\n")

# ---------------------------------------------------
# ROC Curve Plot
# ---------------------------------------------------
plot(roc_logit, col = "blue", lwd = 2, main = "ROC Curves: Logistic vs LASSO vs Ridge")
plot(roc_lasso, col = "red", lwd = 2, add = TRUE)
plot(roc_ridge, col = "green", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright",
       legend = c(
         paste0("Logistic (AUC=", round(auc_logit, 3), ")"),
         paste0("LASSO (AUC=", round(auc_lasso, 3), ")"),
         paste0("Ridge (AUC=", round(auc_ridge, 3), ")")
       ),
       col = c("blue", "red", "green"), lwd = 2)
