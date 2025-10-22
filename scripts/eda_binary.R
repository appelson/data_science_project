# Loading Libraries
library(ggplot2)
library(tidyverse)
library(patchwork)

# Loading data BINARY
df_clean <- read_csv("cleaned_data/cleaned_data.csv") %>%
  select(-c(visits_per_capita,visits, population_lsa))

# Define palette
col_low  <- "#2166AC"
col_high <- "#B2182B"

# ------------------ Defining functions ------------------------------------ 

# Bar chart for binary or categorical variables
make_bar_gg <- function(data, var, fill = col_low) {
  var_name <- deparse(substitute(var))
  
  ggplot(data, aes(x = factor({{ var }}))) +
    geom_bar(fill = fill) +
    labs(
      title = paste("Bar Plot of", var_name),
      x = var_name,
      y = "Count"
    ) +
    theme_minimal(base_size = 14)
}

# Correlation heatmap function
make_corr_heatmap <- function(data, exclude = NULL) {
  df_num <- data %>% select(where(is.numeric), -all_of(exclude))
  
  cor_mat <- cor(df_num, use = "pairwise.complete.obs", method = "pearson")
  cor_df <- as.data.frame(as.table(cor_mat))
  names(cor_df) <- c("Var1", "Var2", "Correlation")
  
  cor_df <- cor_df %>%
    mutate(
      Var1 = factor(Var1, levels = colnames(cor_mat)),
      Var2 = factor(Var2, levels = colnames(cor_mat))
    ) %>%
    filter(as.numeric(Var1) > as.numeric(Var2))
  
  ggplot(cor_df, aes(x = Var2, y = Var1, fill = Correlation)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", Correlation)), size = 3) +
    scale_fill_gradient2(
      low = col_high,
      mid = "white",
      high = col_low,
      midpoint = 0,
      limits = c(-1, 1),
      name = "Correlation"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
      panel.grid = element_blank()
    ) +
    labs(title = "Correlation Matrix of Numeric Predictors")
}

# Binary Outcome Distribution
make_bar_gg(df_clean, visits_per_capita_binary)

#  -------------- Numeric Predictors by Binary Outcome ---------------------
# Boxplots
df_clean %>%
  pivot_longer(cols = where(is.numeric) & !matches("visits_per_capita_binary")) %>%
  ggplot(aes(x = factor(visits_per_capita_binary),
             y = log(value + 1),
             fill = factor(visits_per_capita_binary))) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ name, scales = "free_y") +
  theme_minimal(base_size = 13) +
  scale_fill_manual(values = c(col_low, col_high)) +
  labs(
    title = "Log-Transformed Numeric Predictors by visits_per_capita_binary",
    x = "Visits per Capita (Binary: 0/1)",
    y = "log(x + 1)"
  )

# Density plots
df_clean %>%
  pivot_longer(cols = where(is.numeric) & !matches("visits_per_capita_binary")) %>%
  ggplot(aes(x = log(value + 1), fill = factor(visits_per_capita_binary))) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ name, scales = "free") +
  theme_minimal(base_size = 13) +
  scale_fill_manual(values = c(col_low, col_high)) +
  labs(
    title = "Density of Log-Transformed Numeric Predictors by visits_per_capita_binary",
    x = "log(x + 1)",
    y = "Density",
    fill = "Visits (Binary)"
  )


# Relationship Between Y and X

# --- Difference in Means Between Binary Groups ---
# --- Scaled Difference in Means Between Binary Groups ---
df_clean %>%
  pivot_longer(cols = where(is.numeric) & !matches("visits_per_capita_binary")) %>%
  group_by(name) %>%
  mutate(value_scaled = scale(value)) %>%
  group_by(name, visits_per_capita_binary) %>%
  summarise(mean_value = mean(value_scaled, na.rm = TRUE), .groups = "drop_last") %>%
  summarise(mean_diff = diff(mean_value)) %>%  # group 1 - group 0
  ggplot(aes(x = reorder(name, mean_diff), y = mean_diff, fill = mean_diff > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_manual(values = c(col_low, col_high)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  theme_minimal(base_size = 13) +
  labs(
    title = "Standardized Difference in Means Between Binary Groups",
    x = NULL,
    y = "Mean Difference (in SD Units)"
  )



# --- Improved Categorical Predictors Plot ---
df_clean %>%
  select(where(is.character) | where(is.factor), visits_per_capita_binary) %>%
  pivot_longer(cols = -visits_per_capita_binary) %>%
  group_by(name, value) %>%
  summarise(
    prop_positive = mean(visits_per_capita_binary, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    label = paste0(round(100 * prop_positive, 1), "%"),
    value = fct_reorder(value, prop_positive)
  ) %>%
  ggplot(aes(x = value, y = prop_positive, fill = prop_positive)) +
  geom_col(show.legend = FALSE, width = 0.7) +
  geom_text(aes(label = label), hjust = -0.1, size = 3.2, color = "gray20") +
  facet_wrap(~ name, scales = "free_y", ncol = 2) +
  coord_flip() +
  scale_fill_gradient(low = col_low, high = col_high) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal(base_size = 13) +
  theme(
    strip.text = element_text(face = "bold", size = 12),
    axis.text.y = element_text(size = 10),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 11, color = "gray40")
  ) +
  labs(
    title = "Share of Binary Outcome = 1 by Category",
    subtitle = "Each bar shows the proportion of observations with visits_per_capita_binary = 1",
    x = NULL,
    y = "Proportion (of 1s)"
  )
