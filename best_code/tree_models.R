# ---------------------------------------------------
# Tree-Based Models (Simplest Fixed Version)
# ---------------------------------------------------
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)

# ---- Feature Engineering ----
df_clean <- df_clean %>%
  mutate(
    log_visits = log(visits),
    log_population_lsa = log(population_lsa + 1),
    log_county_population = log(county_population + 1),
    log_print_volumes = log(print_volumes + 1),
    log_ebook_volumes = log(ebook_volumes + 1)
  )

# ---- Updated formula ----
formula_full <- log_visits ~ 
  log_population_lsa + 
  log_county_population + 
  log_print_volumes +
  log_ebook_volumes + 
  num_lib_branches + 
  num_bookmobiles + 
  interlibrary_relation_code + 
  fscs_definition_code + 
  overdue_policy + 
  beac_code + 
  locale_code

# ---- Train/Test Split ----
set.seed(123)
train_index <- createDataPartition(df_clean$visits, p = 0.8, list = FALSE)
train <- df_clean[train_index, ]
test  <- df_clean[-train_index, ]

train <- train %>%
  mutate(across(where(is.character), as.factor))

test <- test %>%
  mutate(across(where(is.character), as.factor))

# ---------------------------------------------------
# Decision Tree
# ---------------------------------------------------
set.seed(123)
model_tree <- rpart(
  formula_full,
  data = train,
  method = "anova"
)

rpart.plot(model_tree)

train_pred_tree <- predict(model_tree, newdata = train)
test_pred_tree  <- predict(model_tree, newdata = test)

tree_train_r2   <- cor(train_pred_tree, train$log_visits)^2
tree_test_r2    <- cor(test_pred_tree,  test$log_visits)^2
tree_train_rmse <- sqrt(mean((train$log_visits - train_pred_tree)^2))
tree_test_rmse  <- sqrt(mean((test$log_visits  - test_pred_tree)^2))

# ---------------------------------------------------
# Random Forest
# ---------------------------------------------------
set.seed(123)
model_rf <- randomForest(
  formula_full,
  data = train,
  ntree = 500,
  importance = TRUE
)

train_pred_rf <- predict(model_rf, newdata = train)
test_pred_rf  <- predict(model_rf, newdata = test)

rf_train_r2   <- cor(train_pred_rf, train$log_visits)^2
rf_test_r2    <- cor(test_pred_rf,  test$log_visits)^2
rf_train_rmse <- sqrt(mean((train$log_visits - train_pred_rf)^2))
rf_test_rmse  <- sqrt(mean((test$log_visits  - test_pred_rf)^2))

varImpPlot(model_rf)

# ---------------------------------------------------
# Gradient Boosting Machine
# ---------------------------------------------------
set.seed(123)
model_gbm <- gbm(
  formula = formula_full,
  data = train,
  distribution = "gaussian",
  n.trees = 2000,
  interaction.depth = 3,
  shrinkage = 0.01,
  n.minobsinnode = 10,
  verbose = FALSE
)

train_pred_gbm <- predict(model_gbm, newdata = train, n.trees = 2000)
test_pred_gbm  <- predict(model_gbm, newdata = test,  n.trees = 2000)

gbm_train_r2   <- cor(train_pred_gbm, train$log_visits)^2
gbm_test_r2    <- cor(test_pred_gbm,  test$log_visits)^2
gbm_train_rmse <- sqrt(mean((train$log_visits - train_pred_gbm)^2))
gbm_test_rmse  <- sqrt(mean((test$log_visits  - test_pred_gbm)^2))

# ---------------------------------------------------
# Combine Results
# ---------------------------------------------------
results_all <- tibble(
  Model = c("Decision Tree", "Random Forest", "GBM"),
  Train_R2 = c(tree_train_r2, rf_train_r2, gbm_train_r2),
  Test_R2  = c(tree_test_r2,  rf_test_r2,  gbm_test_r2),
  Train_RMSE = c(tree_train_rmse, rf_train_rmse, gbm_train_rmse),
  Test_RMSE  = c(tree_test_rmse,  rf_test_rmse,  gbm_test_rmse)
)

print(results_all %>% arrange(desc(Test_R2)))

# --- Decision Tree ---
ggplot(data.frame(True = test$log_visits, Pred = test_pred_tree),
       aes(x = True, y = Pred)) +
  geom_point(alpha = 0.6, color = "purple") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Decision Tree: Predicted vs. True log(visits)",
       x = "True log(visits)", y = "Predicted log(visits)") +
  theme_minimal()

# --- Random Forest ---
ggplot(data.frame(True = test$log_visits, Pred = test_pred_rf),
       aes(x = True, y = Pred)) +
  geom_point(alpha = 0.6, color = "brown") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Random Forest: Predicted vs. True log(visits)",
       x = "True log(visits)", y = "Predicted log(visits)") +
  theme_minimal()

# --- GBM ---
ggplot(data.frame(True = test$log_visits, Pred = test_pred_gbm),
       aes(x = True, y = Pred)) +
  geom_point(alpha = 0.6, color = "darkcyan") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "GBM: Predicted vs. True log(visits)",
       x = "True log(visits)", y = "Predicted log(visits)") +
  theme_minimal()

