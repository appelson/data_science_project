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
df_clean <- read_csv("cleaned_data/cleaned_data.csv")

# Define formula
formula_full <- log(visits) ~ 
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
summary(model_lasso)

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
summary(model_ridge)

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

test <- test %>%
  mutate(log_visits = log(visits))

# ---------------------------------------------------
# Plot True vs. Predicted for OLS, LASSO, Ridge
# ---------------------------------------------------
library(ggplot2)

ggplot(data.frame(True = test$log_visits, Pred = test_pred_ols),
       aes(x = True, y = Pred)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "OLS: Predicted vs. True log(visits)",
    x = "True log(visits)",
    y = "Predicted log(visits)"
  ) +
  theme_minimal()

# LASSO
ggplot(data.frame(True = test$log_visits, Pred = as.numeric(test_pred_lasso)),
       aes(x = True, y = Pred)) +
  geom_point(alpha = 0.6, color = "forestgreen") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "LASSO: Predicted vs. True log(visits)",
    x = "True log(visits)",
    y = "Predicted log(visits)"
  ) +
  theme_minimal()

# Ridge
ggplot(data.frame(True = test$log_visits, Pred = as.numeric(test_pred_ridge)),
       aes(x = True, y = Pred)) +
  geom_point(alpha = 0.6, color = "darkorange") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "Ridge: Predicted vs. True log(visits)",
    x = "True log(visits)",
    y = "Predicted log(visits)"
  ) +
  theme_minimal()



# ---------------------------------------------------
# OLS Assumption Checks
# ---------------------------------------------------

# 1. Linearity (Partial Residual Plots)
#    Each plot shows the linearity of relationship between predictor and response.
car::crPlots(model_ols)

# 2. Normality of Residuals
resid_df <- data.frame(residuals = resid(model_ols))

# Histogram of residuals
ggplot(resid_df, aes(x = residuals)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  theme_minimal() +
  labs(title = "Histogram of OLS Residuals",
       x = "Residuals", y = "Count")

# Q-Q plot for normality
qqnorm(resid(model_ols))
qqline(resid(model_ols), col = "red")

# 3. Homoscedasticity (Constant Variance)
ggplot(data.frame(fitted = fitted(model_ols), resid = resid(model_ols)), 
       aes(x = fitted, y = resid)) +
  geom_point(alpha = 0.5, color = "gray30") +
  geom_smooth(se = FALSE, color = "red") +
  theme_minimal() +
  labs(title = "OLS: Residuals vs Fitted Values",
       x = "Fitted Values", y = "Residuals")

# 4. Independence of Errors (Durbin–Watson test)
lmtest::dwtest(model_ols)

# 5. Multicollinearity (Variance Inflation Factors)
car::vif(model_ols)
