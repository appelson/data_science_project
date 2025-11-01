# ---------------------------------------------------
# Tree-Based Models (Binary Classification Version)
# ---------------------------------------------------
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(pROC)

set.seed(123)

# ---- Feature Engineering ----
df_clean <- read_csv("cleaned_data/cleaned_data.csv")

# Binary target: above-median visits = 1, else 0
median_visits <- median(df_clean$visits, na.rm = TRUE)
df_clean <- df_clean %>%
  mutate(
    visits_binary = ifelse(visits > 50000, 1, 0),
    log_population_lsa = log(population_lsa + 1),
    log_county_population = log(county_population + 1),
    log_print_volumes = log(print_volumes + 1),
    log_ebook_volumes = log(ebook_volumes + 1)
  )

# ---- Updated formula ----
formula_full <- visits_binary ~ 
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
train_index <- createDataPartition(df_clean$visits_binary, p = 0.8, list = FALSE)
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
  method = "class"
)
rpart.plot(model_tree)

# Predict probabilities and classes
train_prob_tree <- predict(model_tree, newdata = train, type = "prob")[,2]
test_prob_tree  <- predict(model_tree, newdata = test,  type = "prob")[,2]

train_pred_tree <- ifelse(train_prob_tree > 0.5, 1, 0)
test_pred_tree  <- ifelse(test_prob_tree  > 0.5, 1, 0)

# Metrics
tree_train_acc <- mean(train_pred_tree == train$visits_binary)
tree_test_acc  <- mean(test_pred_tree  == test$visits_binary)
tree_auc <- roc(test$visits_binary, test_prob_tree)$auc

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

train_prob_rf <- predict(model_rf, newdata = train)
test_prob_rf  <- predict(model_rf, newdata = test)

train_pred_rf <- ifelse(train_prob_rf > 0.5, 1, 0)
test_pred_rf  <- ifelse(test_prob_rf  > 0.5, 1, 0)

rf_train_acc <- mean(train_pred_rf == as.numeric(train$visits_binary) - 1)
rf_test_acc  <- mean(test_pred_rf  == as.numeric(test$visits_binary) - 1)
rf_auc <- roc(as.numeric(test$visits_binary) - 1, test_prob_rf)$auc

varImpPlot(model_rf)

# ---------------------------------------------------
# Gradient Boosting Machine
# ---------------------------------------------------
set.seed(123)
model_gbm <- gbm(
  formula = formula_full,
  data = train,
  distribution = "bernoulli",
  n.trees = 2000,
  interaction.depth = 3,
  shrinkage = 0.01,
  n.minobsinnode = 10,
  verbose = FALSE
)

train_prob_gbm <- predict(model_gbm, newdata = train, n.trees = 2000, type = "response")
test_prob_gbm  <- predict(model_gbm, newdata = test,  n.trees = 2000, type = "response")

train_pred_gbm <- ifelse(train_prob_gbm > 0.5, 1, 0)
test_pred_gbm  <- ifelse(test_prob_gbm  > 0.5, 1, 0)

gbm_train_acc <- mean(train_pred_gbm == train$visits_binary)
gbm_test_acc  <- mean(test_pred_gbm  == test$visits_binary)
gbm_auc <- roc(test$visits_binary, test_prob_gbm)$auc

# ---------------------------------------------------
# Combine Results
# ---------------------------------------------------
results_all <- tibble(
  Model = c("Decision Tree", "Random Forest", "GBM"),
  Train_Accuracy = c(tree_train_acc, rf_train_acc, gbm_train_acc),
  Test_Accuracy  = c(tree_test_acc,  rf_test_acc,  gbm_test_acc),
  Test_AUC       = c(tree_auc, rf_auc, gbm_auc)
)

print(results_all %>% arrange(desc(Test_AUC)))

# ---------------------------------------------------
# ROC Curves
# ---------------------------------------------------
roc_tree <- roc(test$visits_binary, test_prob_tree)
roc_rf   <- roc(test$visits_binary, test_prob_rf)
roc_gbm  <- roc(test$visits_binary, test_prob_gbm)

plot(roc_tree, col = "purple", main = "ROC Curves: Binary Tree Models")
lines(roc_rf, col = "brown")
lines(roc_gbm, col = "darkcyan")
legend("bottomright",
       legend = c("Decision Tree", "Random Forest", "GBM"),
       col = c("purple", "brown", "darkcyan"),
       lwd = 2)

# ---------------------------------------------------
# Confusion Matrices
# ---------------------------------------------------
cat("\nDecision Tree Confusion Matrix:\n")
print(confusionMatrix(as.factor(test_pred_tree), as.factor(test$visits_binary)))

cat("\nRandom Forest Confusion Matrix:\n")
print(confusionMatrix(as.factor(test_pred_rf), as.factor(test$visits_binary)))

cat("\nGBM Confusion Matrix:\n")
print(confusionMatrix(as.factor(test_pred_gbm), as.factor(test$visits_binary)))