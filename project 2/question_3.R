# ----------------------------- 1. LOADING DATA -----------------------------------
library(tidyverse)
library(janitor)
library(boot)
library(broom)
library(patchwork)
library(sandwich)

set.seed(1)

df_train_file <- read_csv("~/Desktop/Classes/Data Science/project 1/cleaned_data/cleaned_data.csv")
df_test_file  <- read_csv("~/Desktop/Classes/Data Science/project 1/cleaned_data/test_data.csv")

data_full <- bind_rows(df_train_file, df_test_file)

categorical_cols <- c(
  "overdue_policy",
  "interlibrary_relation_code",
  "fscs_definition_code",
  "locale_code",
  "beac_code"
)

data_full <- data_full %>%
  mutate(across(all_of(categorical_cols), as.factor)) %>%
  mutate(across(where(is.character), as.factor))

n <- nrow(data_full)
index <- sample(seq_len(n), size = floor(n / 2))
df_train <- data_full[index, ]
df_test  <- data_full[-index, ]

cat("Training data dimensions:", dim(df_train), "\n")
cat("Test data dimensions:", dim(df_test), "\n")

# -----------------------------3(a) Fitting Models ------------------------------
formula_obj <- as.formula(
  "log(visits) ~ log(population_lsa) +
   log1p(print_volumes) +
   log1p(ebook_volumes) +
   log(county_population) + 
   num_bookmobiles +
   log1p(num_lib_branches) +
   overdue_policy +
   interlibrary_relation_code +
   fscs_definition_code +
   locale_code +
   beac_code"
)

model_train <- lm(formula_obj, data = df_train)
summary(model_train)

tidy_train <- broom::tidy(model_train) %>%
  clean_names() %>%
  rename(
    estimate = estimate,
    std_error = std_error,
    t_value = statistic,
    pr_t = p_value
  ) %>%
  mutate(significant_train = pr_t < 0.05)

res_train <- broom::augment(model_train, df_train)

ggplot(res_train, aes(.fitted, .resid)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_smooth(se = FALSE) +
  labs(title = "Residuals vs Fitted — TRAIN", x = "Fitted values", y = "Residuals") +
  theme_minimal()

ggplot(res_train, aes(.fitted, sqrt(abs(.std.resid)))) +
  geom_point(alpha = 0.5) +
  geom_smooth(se = FALSE) +
  labs(title = "Scale–Location — TRAIN",
       x = "Fitted values", y = "sqrt(|Standardized Residual|)") +
  theme_minimal()

ggplot(res_train, aes(x = .resid)) +
  geom_histogram(aes(y = ..density..), bins = 100, fill = "grey70", color = "black") +
  stat_function(fun = dnorm,
                args = list(mean = mean(res_train$.resid, na.rm = TRUE),
                            sd = sd(res_train$.resid, na.rm = TRUE)),
                linewidth = 1.2) +
  labs(title = "Residual Histogram — TRAIN", x = "Residuals", y = "Density") +
  theme_minimal()

ggplot(res_train, aes(sample = .resid)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "QQ Plot — TRAIN", x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()

# -----------------------3(b) Fitting Models on Test ------------------------------
model_test <- lm(formula_obj, data = df_test)
summary(model_test)

tidy_test <- broom::tidy(model_test) %>%
  clean_names() %>%
  rename(
    estimate = estimate,
    std_error = std_error,
    t_value = statistic,
    pr_t = p_value
  ) %>%
  mutate(significant_test = pr_t < 0.05)

res_test <- broom::augment(model_test, df_test)

ggplot(res_test, aes(.fitted, .resid)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_smooth(se = FALSE) +
  labs(title = "Residuals vs Fitted — TEST", x = "Fitted values", y = "Residuals") +
  theme_minimal()

ggplot(res_test, aes(.fitted, sqrt(abs(.std.resid)))) +
  geom_point(alpha = 0.5) +
  geom_smooth(se = FALSE) +
  labs(title = "Scale–Location — TEST",
       x = "Fitted values", y = "sqrt(|Standardized Residual|)") +
  theme_minimal()

ggplot(res_test, aes(x = .resid)) +
  geom_histogram(aes(y = ..density..), bins = 100, fill = "grey70", color = "black") +
  stat_function(fun = dnorm,
                args = list(mean = mean(res_test$.resid, na.rm = TRUE),
                            sd = sd(res_test$.resid, na.rm = TRUE)),
                linewidth = 1.2) +
  labs(title = "Residual Histogram — TEST", x = "Residuals", y = "Density") +
  theme_minimal()

ggplot(res_test, aes(sample = .resid)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "QQ Plot — TEST", x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()

comparison <- tidy_train %>%
  dplyr::select(term, train_p = pr_t, significant_train) %>%
  left_join(
    tidy_test %>% dplyr::select(term, test_p = pr_t, significant_test),
    by = "term"
  ) %>%
  filter(significant_train != significant_test)

comparison

# -----------------------3(c) Bootstrap ------------------------------
df_boot_base <- df_train %>%
  mutate(
    log_visits = log(visits),
    log_pop = log(population_lsa),
    log_print = log1p(print_volumes),
    log_ebook = log1p(ebook_volumes),
    log_county = log(county_population),
    log_branches = log1p(num_lib_branches)
  )

boot_formula <- as.formula(
  "log_visits ~ log_pop + log_print + log_ebook + log_county +
   num_bookmobiles + log_branches +
   overdue_policy + interlibrary_relation_code +
   fscs_definition_code + locale_code + beac_code"
)

model_boot_ref <- lm(boot_formula, data = df_boot_base)
coef_names <- names(coef(model_boot_ref))
K <- length(coef_names)

factor_levels_list <- lapply(dplyr::select(data_full, all_of(categorical_cols)), levels)
names(factor_levels_list) <- categorical_cols

coef_bootstrap <- function(data, indices) {
  d <- data[indices, , drop = FALSE]
  for (v in categorical_cols) d[[v]] <- factor(d[[v]], levels = factor_levels_list[[v]])
  fit <- tryCatch(lm(boot_formula, data = d), error = function(e) NULL)
  if (is.null(fit)) return(rep(NA_real_, K))
  coefs <- coef(fit)
  out <- rep(NA_real_, K)
  names(out) <- coef_names
  out[names(coefs)] <- coefs
  as.numeric(out)
}

B <- 2000
boot_res <- boot(data = df_boot_base, statistic = coef_bootstrap, R = B)

boot_se <- apply(boot_res$t, 2, sd, na.rm = TRUE)

boot_ci_results <- tibble(
  term = coef_names,
  estimate = coef(model_boot_ref),
  boot_se = boot_se,
  perc_low = NA_real_,
  perc_high = NA_real_
)

for (i in seq_along(coef_names)) {
  valid_count <- sum(!is.na(boot_res$t[, i]))
  if (valid_count >= 100) {
    ci <- tryCatch(boot.ci(boot_res, type = "perc", index = i), error = function(e) NULL)
    if (!is.null(ci) && !is.null(ci$percent)) {
      boot_ci_results$perc_low[i]  <- ci$percent[4]
      boot_ci_results$perc_high[i] <- ci$percent[5]
    }
  }
}

standard_ci <- confint(model_boot_ref)
standard_se <- coef(summary(model_boot_ref))[, "Std. Error"]

standard_results <- tibble(
  term = rownames(standard_ci),
  estimate = coef(model_boot_ref),
  std_se = standard_se,
  std_low = standard_ci[, 1],
  std_high = standard_ci[, 2]
)

comparison <- standard_results %>%
  left_join(boot_ci_results, by = "term", suffix = c("_std", "_boot")) %>%
  transmute(
    term,
    estimate = estimate_std,
    std_se,
    std_low,
    std_high,
    boot_se,
    perc_low,
    perc_high
  )

comparison %>%
  mutate(
    sig_standard  = (std_low  > 0 | std_high < 0),
    sig_bootstrap = (perc_low > 0 | perc_high < 0),
    changed_significance = sig_standard != sig_bootstrap
  )

# -----------------------3(e) Bootstrap ------------------------------
tidy_train_corrected <- tidy_train %>%
  mutate(
    p_raw = pr_t,
    p_bonf = p.adjust(p_raw, method = "bonferroni"),
    p_bh   = p.adjust(p_raw, method = "BH"),
    sig_raw = p_raw < 0.05,
    sig_bonf = p_bonf < 0.05,
    sig_bh = p_bh < 0.05
  )

tidy_train_corrected %>%
  filter(sig_raw != sig_bonf | sig_raw != sig_bh | sig_bonf != sig_bh)