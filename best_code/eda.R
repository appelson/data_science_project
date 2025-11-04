# Loading Libraries
library(ggplot2)
library(tidyverse)
library(patchwork)

# Loading data CONTINUOUS
df <- read_csv("cleaned_data/train_data.csv")

# ------------------- Function Creation -------------------------

# Defining a histogram function to make normal and log histograms
make_hist_gg <- function(data, var, fill = "steelblue", color = "white", bins = 30) {
  var_name <- deparse(substitute(var))
  
  p1 <- ggplot(data, aes(x = {{ var }})) +
    geom_histogram(bins = bins, fill = fill, color = color) +
    labs(
      title = paste(var_name),
      x = var_name,
      y = "Count"
    ) +
    theme_minimal(base_size = 8)
  
  p2 <- ggplot(data, aes(x = log({{ var }} + 1))) +
    geom_histogram(bins = bins, fill = fill, color = color) +
    labs(
      title = paste("log(", var_name, " + 1)", sep = ""),
      x = paste0("log(", var_name, " + 1)"),
      y = "Count"
    ) +
    theme_minimal(base_size = 8)
  
  p1 + p2
}

make_qq_plot <- function(data, var) {
  var_name <- deparse(substitute(var))
  
  p1 <- ggplot(data, aes(sample = {{ var }})) +
    stat_qq(color = "steelblue") +
    stat_qq_line(color = "red") +
    labs(title = paste("QQ Plot -", var_name)) +
    theme_minimal()
  
  p2 <- ggplot(data, aes(sample = log({{ var }} + 1))) +
    stat_qq(color = "steelblue") +
    stat_qq_line(color = "red") +
    labs(title = paste("QQ Plot - log(", var_name, "+1)", sep = "")) +
    theme_minimal()
  
  p1 + p2
}

# -------------- Histogram / QQ-Plot Fitting -------------------------

# Creating histograms
hist_visits <- make_hist_gg(df, visits)
hist_pop_lsa <- make_hist_gg(df, population_lsa)
hist_county_pop <- make_hist_gg(df, county_population)
hist_print_vol <- make_hist_gg(df, print_volumes)
hist_ebook_vol <- make_hist_gg(df, ebook_volumes)

# Creating QQ plots
qq_visits <- make_qq_plot(df, visits)
qq_pop_lsa <- make_qq_plot(df, population_lsa)
qq_county_pop <- make_qq_plot(df, county_population)
qq_print_vol <- make_qq_plot(df, print_volumes)
qq_ebook_vol <- make_qq_plot(df, ebook_volumes)

# --------------- Correlation Between Variables -------------------------
df_num <- df %>%
  select(where(is.numeric)) %>%
  mutate(across(
    .cols = -c(num_bookmobiles),
    .fns = ~ log(.x + 1)
  ))

# Compute correlation matrix
cor_mat <- cor(df_num, use = "pairwise.complete.obs", method = "pearson")

# Convert to long format for plotting
cor_df <- as.data.frame(as.table(cor_mat))
names(cor_df) <- c("Var1", "Var2", "Correlation")

# Keep only lower triangle
cor_df <- cor_df %>%
  mutate(
    Var1 = factor(Var1, levels = colnames(cor_mat)),
    Var2 = factor(Var2, levels = colnames(cor_mat))
  ) %>%
  filter(as.numeric(Var1) > as.numeric(Var2))

# Plot correlation heatmap
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
  ) +
  labs(
    title = "Correlation Heatmap (Log Except Bookmobiles)",
    x = NULL,
    y = NULL
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
df %>%
  select(where(is.character) | where(is.factor), visits) %>%
  mutate(
    visits_log = log10(visits)
  ) %>%
  pivot_longer(cols = -c(visits, visits_log)) %>%
  mutate(
    # Cut off long text values and facet names
    value = str_trunc(value, 10, side = "right"),
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
