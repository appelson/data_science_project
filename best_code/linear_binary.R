# ---------------------------------------------------
# Libraries
# ---------------------------------------------------
library(tidyverse)
library(glmnetUtils)
library(caret)
library(pROC)
set.seed(123)

# ---------------------------------------------------
# Data
# ---------------------------------------------------
df_clean <- read_csv("cleaned_data/cleaned_data.csv")

# Create binary target: e.g., above median visits = 1, else 0
median_visits <- median(df_clean$visits, na.rm = TRUE)
df_clean <- df_clean %>%
  mutate(visits_binary = ifelse(visits > 50000, 1, 0))

# ---------------------------------------------------
# Define Formula
# ---------------------------------------------------
formula_full <- visits_binary ~ 
  log(population_lsa + 1) + 
  log(county_population + 1) + 
  log(print_volumes + 1) +
  log(ebook_volumes + 1) + 
  num_lib_branches + 
  num_bookmobiles + 
  interlibrary_relation_code + 
  fscs_definition_code + 
  overdue_policy + 
  beac_code + 
  locale_code

# ---------------------------------------------------
# Train / Test Split
# ---------------------------------------------------
set.seed(123)
train_index <- createDataPartition(df_clean$visits_binary, p = 0.8, list = FALSE)
train <- df_clean[train_index, ]
test  <- df_clean[-train_index, ]

# ---------------------------------------------------
# Logistic Regression (OLS analog)
# ---------------------------------------------------
model_logit <- glm(formula_full, data = train, family = "binomial")
summary(model_logit)

# Predictions
train_prob_logit <- predict(model_logit, newdata = train, type = "response")
test_prob_logit  <- predict(model_logit, newdata = test, type = "response")

train_pred_logit <- ifelse(train_prob_logit > 0.5, 1, 0)
test_pred_logit  <- ifelse(test_prob_logit  > 0.5, 1, 0)

# Metrics
logit_train_acc <- mean(train_pred_logit == train$visits_binary)
logit_test_acc  <- mean(test_pred_logit  == test$visits_binary)
logit_auc <- roc(test$visits_binary, test_prob_logit)$auc

# ---------------------------------------------------
# LASSO (alpha = 1)
# ---------------------------------------------------
set.seed(123)
model_lasso <- cv.glmnet(
  formula_full, data = train,
  alpha = 1,
  family = "binomial"
)
lasso_lambda <- model_lasso$lambda.min
coef(model_lasso, s = "lambda.min")

# Predictions
train_prob_lasso <- predict(model_lasso, newdata = train, s = "lambda.min", type = "response")
test_prob_lasso  <- predict(model_lasso, newdata = test,  s = "lambda.min", type = "response")

train_pred_lasso <- ifelse(train_prob_lasso > 0.5, 1, 0)
test_pred_lasso  <- ifelse(test_prob_lasso  > 0.5, 1, 0)

# Metrics
lasso_train_acc <- mean(train_pred_lasso == train$visits_binary)
lasso_test_acc  <- mean(test_pred_lasso  == test$visits_binary)
lasso_auc <- roc(test$visits_binary, as.numeric(test_prob_lasso))$auc

# ---------------------------------------------------
# Ridge (alpha = 0)
# ---------------------------------------------------
set.seed(123)
model_ridge <- cv.glmnet(
  formula_full, data = train,
  alpha = 0,
  family = "binomial"
)
ridge_lambda <- model_ridge$lambda.min
coef(model_ridge, s = "lambda.min")

# Predictions
train_prob_ridge <- predict(model_ridge, newdata = train, s = "lambda.min", type = "response")
test_prob_ridge  <- predict(model_ridge, newdata = test,  s = "lambda.min", type = "response")

train_pred_ridge <- ifelse(train_prob_ridge > 0.5, 1, 0)
test_pred_ridge  <- ifelse(test_prob_ridge  > 0.5, 1, 0)

# Metrics
ridge_train_acc <- mean(train_pred_ridge == train$visits_binary)
ridge_test_acc  <- mean(test_pred_ridge  == test$visits_binary)
ridge_auc <- roc(test$visits_binary, as.numeric(test_prob_ridge))$auc

# ---------------------------------------------------
# Compare Performance
# ---------------------------------------------------
results <- tibble(
  Model = c("Logistic", "LASSO", "Ridge"),
  Train_Accuracy = c(logit_train_acc, lasso_train_acc, ridge_train_acc),
  Test_Accuracy  = c(logit_test_acc,  lasso_test_acc,  ridge_test_acc),
  Test_AUC       = c(logit_auc, lasso_auc, ridge_auc)
)

print(results)

# ---------------------------------------------------
# ROC Curves
# ---------------------------------------------------
roc_logit <- roc(test$visits_binary, test_prob_logit)
roc_lasso <- roc(test$visits_binary, as.numeric(test_prob_lasso))
roc_ridge <- roc(test$visits_binary, as.numeric(test_prob_ridge))

plot(roc_logit, col = "steelblue", main = "ROC Curves for Binary Models")
lines(roc_lasso, col = "forestgreen")
lines(roc_ridge, col = "darkorange")
legend("bottomright",
       legend = c("Logistic", "LASSO", "Ridge"),
       col = c("steelblue", "forestgreen", "darkorange"),
       lwd = 2)

# ---------------------------------------------------
# Confusion Matrices
# ---------------------------------------------------
cat("\nLogistic Confusion Matrix:\n")
print(confusionMatrix(as.factor(test_pred_logit), as.factor(test$visits_binary)))

cat("\nLASSO Confusion Matrix:\n")
print(confusionMatrix(as.factor(test_pred_lasso), as.factor(test$visits_binary)))

cat("\nRidge Confusion Matrix:\n")
print(confusionMatrix(as.factor(test_pred_ridge), as.factor(test$visits_binary)))
