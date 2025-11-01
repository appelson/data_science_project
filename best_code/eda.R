# Loading Libraries
library(ggplot2)
library(tidyverse)
library(patchwork)

# Loading data CONTINUOUS
df_clean <- read_csv("cleaned_data/cleaned_data.csv")

# ------------------- Function Creation -------------------------

# Defining a histogram function to make normal and log histograms
make_hist_gg <- function(data, var, fill = "steelblue", color = "white", bins = 30) {
  var_name <- deparse(substitute(var))
  
  p1 <- ggplot(data, aes(x = {{ var }})) +
    geom_histogram(bins = bins, fill = fill, color = color) +
    labs(
      title = paste("Histogram of", var_name),
      x = var_name,
      y = "Count"
    ) +
    theme_minimal(base_size = 14)
  
  p2 <- ggplot(data, aes(x = log({{ var }} + 1))) +
    geom_histogram(bins = bins, fill = fill, color = color) +
    labs(
      title = paste("Histogram of log(", var_name, " + 1)", sep = ""),
      x = paste0("log(", var_name, " + 1)"),
      y = "Count"
    ) +
    theme_minimal(base_size = 14)
  
  p1 + p2
}

# ------------------- Histogram Fitting -------------------------
make_hist_gg(df_clean,visits)

make_hist_gg(df_clean,population_lsa)

make_hist_gg(df_clean,county_population)

make_hist_gg(df_clean,print_volumes)

make_hist_gg(df_clean,ebook_volumes)

# --------------- Correlation Between Variables -------------------------

df_num <- df_clean %>% select(where(is.numeric))
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
    low = "#B2182B",
    mid = "white",
    high = "#2166AC",
    midpoint = 0,
    limits = c(-1, 1),
    name = "Correlation"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    panel.grid = element_blank()
  )

# --------------- Relationship Between Y and X -------------------------

# Numeric
df_num %>%
  pivot_longer(cols = -visits) %>%
  ggplot(aes(x = value, y = visits)) +
  geom_point(alpha = 0.5, color = "#2166AC") +
  geom_smooth(method = "lm", se = FALSE, color = "#B2182B") +
  facet_wrap(~ name, scales = "free") +
  scale_x_log10() +
  scale_y_log10() +
  theme_minimal() +
  labs(
    title = "Log–Log Scatterplots: Visits vs Numeric Predictors",
    x = "Predictor (log10 scale)",
    y = "Visits (log10 scale)"
  )

# Categorical 
df_clean %>%
  select(where(is.character) | where(is.factor), visits) %>%
  mutate(
    visits_log = log10(visits)
  ) %>%
  pivot_longer(cols = -c(visits, visits_log)) %>%
  mutate(
    # Cut off long text values and facet names
    value = str_trunc(value, 10, side = "right"),  # truncate to 25 characters
    name = str_trunc(name, 10, side = "right")
  ) %>%
  ggplot(aes(x = value, y = visits_log)) +
  geom_boxplot(fill = "#B2182B", outlier.alpha = 0.4) +
  facet_wrap(~ name, scales = "free_x") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold")
  ) +
  labs(
    title = "Log(Visits) by Categorical Variables",
    x = NULL,
    y = "Log10(Visits)"
  )
