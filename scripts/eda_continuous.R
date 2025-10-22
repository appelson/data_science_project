# Loading Libraries
library(ggplot2)
library(tidyverse)
library(patchwork)

# Loading data CONTINUOUS
df_clean <- read_csv("cleaned_data/cleaned_data.csv") %>%
  select(-c(visits_per_capita, visits_per_capita_binary))

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

make_hist_gg(df_clean,print_volumes)

make_hist_gg(df_clean,ebook_volumes)

make_hist_gg(df_clean,audio_physical)

make_hist_gg(df_clean,audio_digital)

make_hist_gg(df_clean,video_physical)

make_hist_gg(df_clean,video_digital)

make_hist_gg(df_clean,total_e_collection)

make_hist_gg(df_clean,tot_physical)

make_hist_gg(df_clean,other_physical)

make_hist_gg(df_clean,mls_librarians_fte)

make_hist_gg(df_clean,total_staff_fte)

make_hist_gg(df_clean,other_paid_fte)

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
  facet_wrap(~ name, scales = "free_x") +
  theme_minimal() +
  labs(title = "Scatterplots: visits vs Numeric Predictors")

# Categorical 
df_clean %>%
  select(where(is.character) | where(is.factor), visits) %>%
  pivot_longer(cols = -visits) %>%
  ggplot(aes(x = value, y = visits)) +
  geom_boxplot(fill = "#B2182B", outlier.alpha = 0.4) +
  facet_wrap(~ name, scales = "free_x") +
  theme_minimal() +
  labs(title = "Visits by Categorical Variables", x = NULL, y = "Visits")
