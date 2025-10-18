# ============================================================
#   Libraries
# ============================================================
library(tidyverse)
library(janitor)
library(corrplot)
library(randomForest)
library(xgboost)
library(glmnet)
library(e1071)
library(Metrics)
library(caret)

# ============================================================
#   Load and Clean Data
# ============================================================
cat("Loading data...\n")
df <- read_csv("PLS_FY2022 PUD_CSV/PLS_FY22_AE_pud22i.csv", show_col_types = FALSE)

df_clean <- df %>%
  select(
    "Visits" = VISITS,
    "Interlibrary Relationship Code" = C_RELATN,
    "Legal Basis Code" = C_LEGBAS,
    "Administrative Structure Code" = C_ADMIN,
    "FSCS Public Library Definition" = C_FSCS,
    "Geographic Code" = GEOCODE,
    "Population of LSA" = POPU_LSA,
    
    # ---- Collections ----
    "Print materials" = BKVOL,
    "Electronic Books (E-books)" = EBOOK,
    "Audio - physical units" = AUDIO_PH,
    "Audio - downloadable units" = AUDIO_DL,
    "Video - physical units" = VIDEO_PH,
    "Video - downloadable units" = VIDEO_DL,
    "Total electronic collections" = ELECCOLL,
    
    # ---- Staffing ----
    "ALA-MLS Librarians" = MASTER,
    "Total number of FTE employees" = LIBRARIA,
    "All other paid FTE employees" = OTHPAID,
    
    # ---- Funding & Expenditures ----
    "Operating revenue from local government" = LOCGVT,
    "Operating revenue from state government" = STGVT,
    "Operating revenue from federal government" = FEDGVT,
    "Other operating revenue" = OTHINCM,
    "Total staff expenditures" = STAFFEXP,
    "Total expenditures on library collection" = TOTEXPCO,
    "Total capital revenue" = CAP_REV,
    
    # ---- Service Metrics ----
    "Total annual public service hours for all service outlets" = HRS_OPEN,
    "Total number of synchronous program sessions" = TOTPRO,
    
    # ---- Demographics ----
    "County Population" = CNTYPOP,
    
    # ---- Context ----
    "Impute Status" = RSTATUS
  ) %>%
  clean_names() %>%
  mutate(
    across(everything(), ~ ifelse(. %in% c(-1, -3, -4, -9), NA, .))
  ) %>%
  filter(impute_status %in% c(1, 3)) %>%
  filter(!is.na(visits), visits > 0) %>%
  mutate(
    across(
      where(~ all(!is.na(suppressWarnings(as.numeric(na.omit(as.character(.))))))),
      as.numeric
    )
  )

cat("Cleaned data: ", nrow(df_clean), "rows,", ncol(df_clean), "columns\n\n")

# ============================================================
#   Create Utilization Rate (Visits per LSA)
# ============================================================
df_utilization <- df_clean %>%
  filter(!is.na(population_of_lsa), population_of_lsa > 0) %>%
  mutate(
    utilization_rate = visits / population_of_lsa
  ) %>%
  select(
    utilization_rate,
    interlibrary_relationship_code,
    legal_basis_code,
    administrative_structure_code,
    fscs_public_library_definition,
    geographic_code,
    print_materials,
    electronic_books_e_books,
    audio_physical_units,
    audio_downloadable_units,
    video_physical_units,
    video_downloadable_units,
    total_electronic_collections,
    ala_mls_librarians,
    total_number_of_fte_employees,
    all_other_paid_fte_employees,
    operating_revenue_from_local_government,
    operating_revenue_from_state_government,
    operating_revenue_from_federal_government,
    other_operating_revenue,
    total_staff_expenditures,
    total_expenditures_on_library_collection,
    total_capital_revenue
  ) %>%
  mutate(across(where(is.character), as.factor),
         log_utilization_rate = log(utilization_rate)) %>%
  na.omit() %>%
  select(-utilization_rate)

# ============================================================
#   Train/Test Split (80/20)
# ============================================================
set.seed(123)
train_index <- createDataPartition(df_utilization$log_utilization_rate, p = 0.8, list = FALSE)
train <- df_utilization[train_index, ]
test <- df_utilization[-train_index, ]

cat("Train set:", nrow(train), "| Test set:", nrow(test), "\n\n")

# ============================================================
#   Helper: RMSE and R² functions
# ============================================================
get_rmse <- function(actual, predicted) {
  rmse(actual, predicted)
}

get_r2 <- function(actual, predicted) {
  1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
}

# ============================================================
#   1. Linear Regression
# ============================================================
cat("Training Linear Regression...\n")
lm_model <- lm(log_utilization_rate ~ ., data = train)
lm_train_pred <- predict(lm_model, newdata = train)
lm_test_pred <- predict(lm_model, newdata = test)

lm_train_rmse <- get_rmse(train$log_utilization_rate, lm_train_pred)
lm_test_rmse <- get_rmse(test$log_utilization_rate, lm_test_pred)
lm_train_r2 <- get_r2(train$log_utilization_rate, lm_train_pred)
lm_test_r2 <- get_r2(test$log_utilization_rate, lm_test_pred)

# ============================================================
#   2. Random Forest
# ============================================================
cat("Training Random Forest...\n")
rf_model <- randomForest(
  log_utilization_rate ~ ., 
  data = train, 
  ntree = 500, 
  mtry = floor(sqrt(ncol(train) - 1)),
  importance = TRUE
)

rf_train_pred <- predict(rf_model, newdata = train)
rf_test_pred <- predict(rf_model, newdata = test)

rf_train_rmse <- get_rmse(train$log_utilization_rate, rf_train_pred)
rf_test_rmse <- get_rmse(test$log_utilization_rate, rf_test_pred)
rf_train_r2 <- get_r2(train$log_utilization_rate, rf_train_pred)
rf_test_r2 <- get_r2(test$log_utilization_rate, rf_test_pred)

# ============================================================
#   3. XGBoost
# ============================================================
cat("Training XGBoost...\n")
to_xgb_matrix <- function(df) {
  df %>% 
    select(-log_utilization_rate) %>% 
    mutate(across(where(is.factor), as.numeric)) %>% 
    as.matrix()
}

X_train <- to_xgb_matrix(train)
X_test  <- to_xgb_matrix(test)

y_train <- train$log_utilization_rate
y_test  <- test$log_utilization_rate

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test, label = y_test)

xgb_model <- xgboost(
  data = dtrain,
  objective = "reg:squarederror",
  nrounds = 500,
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

xgb_train_pred <- predict(xgb_model, dtrain)
xgb_test_pred <- predict(xgb_model, dtest)

xgb_train_rmse <- get_rmse(y_train, xgb_train_pred)
xgb_test_rmse <- get_rmse(y_test, xgb_test_pred)
xgb_train_r2 <- get_r2(y_train, xgb_train_pred)
xgb_test_r2 <- get_r2(y_test, xgb_test_pred)

# ============================================================
#   4. LASSO Regression
# ============================================================
cat("Training LASSO Regression...\n")
X_mat <- model.matrix(log_utilization_rate ~ ., train)[, -1]
y_vec <- train$log_utilization_rate
X_test_mat <- model.matrix(log_utilization_rate ~ ., test)[, -1]

cv_lasso <- cv.glmnet(X_mat, y_vec, alpha = 1, nfolds = 10)

lasso_train_pred <- as.vector(predict(cv_lasso, newx = X_mat, s = "lambda.min"))
lasso_test_pred <- as.vector(predict(cv_lasso, newx = X_test_mat, s = "lambda.min"))

lasso_train_rmse <- get_rmse(y_vec, lasso_train_pred)
lasso_test_rmse <- get_rmse(test$log_utilization_rate, lasso_test_pred)
lasso_train_r2 <- get_r2(y_vec, lasso_train_pred)
lasso_test_r2 <- get_r2(test$log_utilization_rate, lasso_test_pred)

# ============================================================
#   5. Support Vector Regression
# ============================================================
cat("Training Support Vector Regression...\n")
svr_model <- svm(
  log_utilization_rate ~ ., 
  data = train, 
  kernel = "radial", 
  cost = 1, 
  epsilon = 0.1
)

svr_train_pred <- predict(svr_model, newdata = train)
svr_test_pred <- predict(svr_model, newdata = test)

svr_train_rmse <- get_rmse(train$log_utilization_rate, svr_train_pred)
svr_test_rmse <- get_rmse(test$log_utilization_rate, svr_test_pred)
svr_train_r2 <- get_r2(train$log_utilization_rate, svr_train_pred)
svr_test_r2 <- get_r2(test$log_utilization_rate, svr_test_pred)

# ============================================================
#   Combine Model Performance Results
# ============================================================
cat("\n")
cat(paste(rep("=", 60), collapse=""), "\n")
cat("MODEL PERFORMANCE COMPARISON\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

results <- tibble(
  Model = c("Linear", "Random Forest", "XGBoost", "LASSO", "SVR"),
  Train_RMSE = c(lm_train_rmse, rf_train_rmse, xgb_train_rmse, lasso_train_rmse, svr_train_rmse),
  Test_RMSE = c(lm_test_rmse, rf_test_rmse, xgb_test_rmse, lasso_test_rmse, svr_test_rmse),
  Train_R2 = c(lm_train_r2, rf_train_r2, xgb_train_r2, lasso_train_r2, svr_train_r2),
  Test_R2 = c(lm_test_r2, rf_test_r2, xgb_test_r2, lasso_test_r2, svr_test_r2),
  Overfit = Train_RMSE - Test_RMSE
) %>%
  arrange(Test_RMSE)

print(results, n = Inf)

# ============================================================
#   Best Model Results
# ============================================================
best_model <- results$Model[1]
cat("\n")
cat(paste(rep("=", 60), collapse=""), "\n")
cat("BEST MODEL:", best_model, "\n")
cat(paste(rep("=", 60), collapse=""), "\n")

if (best_model == "Linear") {
  test_pred <- lm_test_pred
} else if (best_model == "Random Forest") {
  test_pred <- rf_test_pred
} else if (best_model == "XGBoost") {
  test_pred <- xgb_test_pred
} else if (best_model == "LASSO") {
  test_pred <- lasso_test_pred
} else if (best_model == "SVR") {
  test_pred <- svr_test_pred
}

test_rmse <- get_rmse(test$log_utilization_rate, test_pred)
test_r2 <- get_r2(test$log_utilization_rate, test_pred)

cat("Test RMSE:", round(test_rmse, 5), "\n")
cat("Test R²:", round(test_r2, 4), "\n\n")

# ============================================================
#   Plot: Actual vs Predicted for Best Model
# ============================================================
plot_data <- data.frame(
  actual = test$log_utilization_rate, 
  predicted = test_pred
)

p1 <- ggplot(plot_data, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.4, color = "steelblue", size = 1.5) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  labs(
    title = paste("Actual vs Predicted —", best_model),
    subtitle = paste("Test RMSE:", round(test_rmse, 4), "| R²:", round(test_r2, 3)),
    x = "Actual Log Utilization Rate",
    y = "Predicted Log Utilization Rate"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11)
  )

print(p1)

# ============================================================
#   Residual Plot
# ============================================================
plot_data$residual <- plot_data$actual - plot_data$predicted

p2 <- ggplot(plot_data, aes(x = predicted, y = residual)) +
  geom_point(alpha = 0.4, color = "darkorange", size = 1.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  labs(
    title = paste("Residual Plot —", best_model),
    x = "Predicted Log Utilization Rate",
    y = "Residuals"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

print(p2)

# ============================================================
#   Feature Importance (if tree-based model)
# ============================================================
if (best_model == "Random Forest") {
  varImpPlot(rf_model, main = "Random Forest Variable Importance", col = "steelblue")
} else if (best_model == "XGBoost") {
  importance_matrix <- xgb.importance(model = xgb_model)
  xgb.plot.importance(importance_matrix, top_n = 15, main = "XGBoost Feature Importance")
}

cat("\nAnalysis complete!\n")

