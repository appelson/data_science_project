# ============================================================
#   Prepare for Modeling
# ============================================================
df_model <- df_clean_per_lsa %>% drop_na()
set.seed(123)
train_idx <- sample(1:nrow(df_model), 0.8 * nrow(df_model))
train <- df_model[train_idx, ]
test  <- df_model[-train_idx, ]

# ============================================================
#   Helper function for RMSE
# ============================================================
rmse <- function(actual, predicted) sqrt(mean((predicted - actual)^2))

# ============================================================
#   Linear Model (Baseline)
# ============================================================
lm_model <- lm(visits_per_lsa ~ ., data = train)
lm_train_pred <- predict(lm_model, newdata = train)
lm_test_pred <- predict(lm_model, newdata = test)
lm_rmse_train <- rmse(train$visits_per_lsa, lm_train_pred)
lm_rmse_test  <- rmse(test$visits_per_lsa, lm_test_pred)

# ============================================================
#   Random Forest
# ============================================================
rf_model <- randomForest(visits_per_lsa ~ ., data = train, ntree = 500, mtry = floor(sqrt(ncol(train)-1)), importance = TRUE)
rf_train_pred <- predict(rf_model, newdata = train)
rf_test_pred <- predict(rf_model, newdata = test)
rf_rmse_train <- rmse(train$visits_per_lsa, rf_train_pred)
rf_rmse_test  <- rmse(test$visits_per_lsa, rf_test_pred)

# ============================================================
#   XGBoost
# ============================================================
train_x <- model.matrix(visits_per_lsa ~ ., train)[, -1]
test_x  <- model.matrix(visits_per_lsa ~ ., test)[, -1]
train_y <- train$visits_per_lsa
test_y  <- test$visits_per_lsa

xgb_model <- xgboost(
  data = as.matrix(train_x),
  label = train_y,
  nrounds = 500,
  objective = "reg:squarederror",
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

xgb_train_pred <- predict(xgb_model, newdata = train_x)
xgb_test_pred <- predict(xgb_model, newdata = test_x)
xgb_rmse_train <- rmse(train_y, xgb_train_pred)
xgb_rmse_test  <- rmse(test_y, xgb_test_pred)

# ============================================================
#   Elastic Net
# ============================================================
x <- model.matrix(visits_per_lsa ~ ., df_model)[, -1]
y <- df_model$visits_per_lsa
cv_fit <- cv.glmnet(x, y, alpha = 0.5)
enet_train_pred <- predict(cv_fit, s = "lambda.min", newx = train_x)
enet_test_pred  <- predict(cv_fit, s = "lambda.min", newx = test_x)
enet_rmse_train <- rmse(train_y, enet_train_pred)
enet_rmse_test  <- rmse(test_y, enet_test_pred)

# ============================================================
#   Neural Network
# ============================================================
nn_model <- nnet(visits_per_lsa ~ ., data = train, size = 10, linout = TRUE, maxit = 1000, trace = FALSE)
nn_train_pred <- predict(nn_model, newdata = train)
nn_test_pred <- predict(nn_model, newdata = test)
nn_rmse_train <- rmse(train$visits_per_lsa, nn_train_pred)
nn_rmse_test  <- rmse(test$visits_per_lsa, nn_test_pred)

# ============================================================
#   Compare Models
# ============================================================
results <- data.frame(
  Model = c("Linear", "Random Forest", "XGBoost", "Elastic Net", "Neural Net"),
  Train_RMSE = c(lm_rmse_train, rf_rmse_train, xgb_rmse_train, enet_rmse_train, nn_rmse_train),
  Test_RMSE = c(lm_rmse_test, rf_rmse_test, xgb_rmse_test, enet_rmse_test, nn_rmse_test)
)

print(results)

ggplot(results, aes(x = reorder(Model, Test_RMSE), y = Test_RMSE, fill = Model)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Model Test RMSE Comparison", x = "Model", y = "Test RMSE")

# ============================================================
#   Plot: Actual vs Predicted (Best Model Example - XGBoost)
# ============================================================
ggplot(data.frame(actual = test_y, predicted = xgb_test_pred),
       aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.5, color = "darkorange") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(title = "XGBoost: Actual vs Predicted Visits per LSA",
       x = "Actual Visits per LSA",
       y = "Predicted Visits per LSA")