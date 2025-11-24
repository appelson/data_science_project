library(tidyverse)
library(boot)
library(tableone)
library(ggplot2)

# ------------------------------------------------------------
# 1. DATA PREP
# ------------------------------------------------------------
df <- df_train %>%
  mutate(
    Y = log(visits),
    A = ifelse(overdue_policy == "Has Overdue Policy", 1, 0)
  )

Xvars <- c("population_lsa", "print_volumes", "ebook_volumes",
           "county_population", "num_lib_branches", "num_bookmobiles")

# ------------------------------------------------------------
# 2. PROPENSITY SCORE MODEL (Logistic)
# ------------------------------------------------------------
ps_model <- glm(
  A ~ ., 
  data = df %>% dplyr::select(A, all_of(Xvars)),
  family = binomial()
)

summary(ps_model)

df$ps <- predict(ps_model, type = "response")

# ------------------------------------------------------------
# 3. OUTCOME MODELS (OLS)
# ------------------------------------------------------------
outcome_model_1 <- lm(
  Y ~ ., 
  data = df %>% filter(A == 1) %>% dplyr::select(Y, all_of(Xvars))
)

summary(outcome_model_1)

outcome_model_0 <- lm(
  Y ~ ., 
  data = df %>% filter(A == 0) %>% dplyr::select(Y, all_of(Xvars))
)

summary(outcome_model_0)

df$m1 <- predict(outcome_model_1, newdata = df)
df$m0 <- predict(outcome_model_0, newdata = df)

# ------------------------------------------------------------
# 4. AIPW PSEUDO-OUTCOMES
# ------------------------------------------------------------
df$aipw_1 <- df$m1 + (df$A / df$ps) * (df$Y - df$m1)
df$aipw_0 <- df$m0 + ((1 - df$A) / (1 - df$ps)) * (df$Y - df$m0)

# ------------------------------------------------------------
# 5. AIPW ATE, SE, CI (Lecture Formula)
# ------------------------------------------------------------
ATE <- mean(df$aipw_1 - df$aipw_0)
SE  <- sd(df$aipw_1 - df$aipw_0) / sqrt(nrow(df))

CI_low  <- ATE - 1.96 * SE
CI_high <- ATE + 1.96 * SE

cat("\n=== AIPW ESTIMATE ===\n")
cat("ATE (log outcome):", round(ATE, 4), "\n")
cat("SE:               ", round(SE, 4), "\n")
cat("95% CI:           [", round(CI_low, 4), ",", round(CI_high, 4), "]\n")

# ------------------------------------------------------------
# 6. EXPONENTIATED EFFECT (Interpretation)
# ------------------------------------------------------------
mult_eff <- exp(ATE)
pct_eff  <- (exp(ATE) - 1) * 100

cat("\n=== INTERPRETATION ===\n")
cat("exp(ATE): multiplicative effect =", round(mult_eff, 3), "\n")
cat("Percent change in visits:", round(pct_eff, 1), "%\n")

# ------------------------------------------------------------
# 7. BOOTSTRAP CI FOR ATE
# ------------------------------------------------------------
aipw_stat <- function(data, idx) {
  d <- data[idx, ]
  
  # Refit PS
  ps_m <- glm(A ~ ., data = d %>% dplyr::select(A, all_of(Xvars)), family = binomial())
  d$ps <- predict(ps_m, type = "response")
  
  # Refit outcome models
  m1 <- lm(Y ~ ., data = d %>% filter(A == 1) %>% dplyr::select(Y, all_of(Xvars)))
  m0 <- lm(Y ~ ., data = d %>% filter(A == 0) %>% dplyr::select(Y, all_of(Xvars)))
  
  d$m1 <- predict(m1, newdata = d)
  d$m0 <- predict(m0, newdata = d)
  
  aipw1 <- d$m1 + (d$A / d$ps) * (d$Y - d$m1)
  aipw0 <- d$m0 + ((1 - d$A) / (1 - d$ps)) * (d$Y - d$m0)
  
  mean(aipw1 - aipw0)
}

set.seed(1)
boot_out <- boot(df, statistic = aipw_stat, R = 1000)

boot_se <- sd(boot_out$t, na.rm = TRUE)
boot_ci <- quantile(boot_out$t, c(0.025, 0.975), na.rm = TRUE)

cat("\n=== BOOTSTRAP RESULTS ===\n")
cat("Bootstrap SE:", round(boot_se, 4), "\n")
cat("Bootstrap CI: [", round(boot_ci[1],4), ",", round(boot_ci[2],4), "]\n")


boot_ate <- mean(boot_out$t, na.rm = TRUE)
boot_mult_eff <- exp(boot_ate)
boot_pct_eff  <- (exp(boot_ate) - 1) * 100

cat("\n=== BOOTSTRAP POINT ESTIMATE ===\n")
cat("Bootstrap ATE (log scale):", round(boot_ate, 4), "\n")
cat("Multiplicative effect:", round(boot_mult_eff, 3), "\n")
cat("Percent change:", round(boot_pct_eff, 1), "%\n")