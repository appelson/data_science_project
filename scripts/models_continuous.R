# ---------------------------------------------------
# Libraries
# ---------------------------------------------------
library(tidyverse)
library(glmnetUtils)
library(caret)

set.seed(123)  # for reproducibility

# ---------------------------------------------------
# Data
# ---------------------------------------------------
df_clean <- read_csv("cleaned_data/cleaned_data.csv") %>%
  select(-c(visits_per_capita, visits_per_capita_binary))

# Define formula
formula_full <- log(visits) ~ 
  log(county_population + 1) + 
  log(population_lsa + 1) +
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
train_index <- createDataPartition(df_clean$visits, p = 0.8, list = FALSE)
train <- df_clean[train_index, ]
test  <- df_clean[-train_index, ]

# ---------------------------------------------------
# OLS Baseline
# ---------------------------------------------------
model_ols <- lm(formula_full, data = train)
summary(model_ols)

# Predictions
train_pred_ols <- predict(model_ols, newdata = train)
test_pred_ols  <- predict(model_ols, newdata = test)

# R² and RMSE
ols_train_r2 <- cor(train_pred_ols, log(train$visits))^2
ols_test_r2  <- cor(test_pred_ols, log(test$visits))^2
ols_train_rmse <- sqrt(mean((log(train$visits) - train_pred_ols)^2))
ols_test_rmse  <- sqrt(mean((log(test$visits) - test_pred_ols)^2))

# ---------------------------------------------------
# LASSO (alpha = 1)
# ---------------------------------------------------
set.seed(123)
model_lasso <- cv.glmnet(
  formula_full, data = train,
  alpha = 1
)
plot(model_lasso)

lasso_lambda <- model_lasso$lambda.min
coef(model_lasso, s = "lambda.min")

# Predictions
train_pred_lasso <- predict(model_lasso, newdata = train, s = "lambda.min")
test_pred_lasso  <- predict(model_lasso, newdata = test,  s = "lambda.min")

lasso_train_r2 <- cor(train_pred_lasso, log(train$visits))^2
lasso_test_r2  <- cor(test_pred_lasso,  log(test$visits))^2
lasso_train_rmse <- sqrt(mean((log(train$visits) - train_pred_lasso)^2))
lasso_test_rmse  <- sqrt(mean((log(test$visits) - test_pred_lasso)^2))

# ---------------------------------------------------
# Ridge (alpha = 0)
# ---------------------------------------------------
set.seed(123)
model_ridge <- cv.glmnet(
  formula_full, data = train,
  alpha = 0
)
plot(model_ridge)

ridge_lambda <- model_ridge$lambda.min
coef(model_ridge, s = "lambda.min")

# Predictions
train_pred_ridge <- predict(model_ridge, newdata = train, s = "lambda.min")
test_pred_ridge  <- predict(model_ridge, newdata = test,  s = "lambda.min")

ridge_train_r2 <- cor(train_pred_ridge, log(train$visits))^2
ridge_test_r2  <- cor(test_pred_ridge,  log(test$visits))^2
ridge_train_rmse <- sqrt(mean((log(train$visits) - train_pred_ridge)^2))
ridge_test_rmse  <- sqrt(mean((log(test$visits) - test_pred_ridge)^2))

# ---------------------------------------------------
# Compare Performance
# ---------------------------------------------------
results <- tibble(
  Model = c("OLS", "LASSO", "Ridge"),
  Train_R2 = c(ols_train_r2, lasso_train_r2, ridge_train_r2),
  Test_R2  = c(ols_test_r2,  lasso_test_r2,  ridge_test_r2),
  Train_RMSE = c(ols_train_rmse, lasso_train_rmse, ridge_train_rmse),
  Test_RMSE  = c(ols_test_rmse,  lasso_test_rmse,  ridge_test_rmse)
)

print(results)

