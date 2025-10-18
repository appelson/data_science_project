# ============================================================
#   Libraries
# ============================================================
library(tidyverse)
library(janitor)
library(gridExtra)
library(grid)   # 👈 Needed for textGrob() and gpar()

# ============================================================
#   Load and Clean Data
# ============================================================
df <- read_csv("PLS_FY2022 PUD_CSV/PLS_FY22_AE_pud22i.csv")

missing_codes <- c(-1, -3, -4, -9)

df_clean <- df %>%
  mutate(across(everything(), ~ ifelse(.x %in% missing_codes, NA, .x))) %>%
  filter(RSTATUS %in% c(1, 3)) %>%
  filter(VISITS > 0, POPU_LSA > 0)

# ============================================================
#   Continuous Variables (RAW NAMES)
# ============================================================
continuous_vars <- c(
  "BKVOL", "EBOOK", "AUDIO_PH", "AUDIO_DL",
  "VIDEO_PH", "VIDEO_DL", "ELECCOLL", "TOTPHYS", "OTHPHYS",
  "MASTER", "LIBRARIA", "OTHPAID",
  "HRS_OPEN", "TOTPRO", "CNTYPOP",
  "CENTLIB", "BRANLIB", "BKMOB"
)

continuous_vars <- continuous_vars[continuous_vars %in% names(df_clean)]
cat("Number of continuous variables found:", length(continuous_vars), "\n")

# ============================================================
#   Plot Each Variable (raw, log, per capita, log per capita)
# ============================================================
for (colname in continuous_vars) {
  df_plot <- df_clean %>%
    select(all_of(c(colname, "POPU_LSA"))) %>%
    drop_na() %>%
    filter(.data[[colname]] > 0, POPU_LSA > 0) %>%
    mutate(
      log_value = log(.data[[colname]]),
      per_capita = .data[[colname]] / POPU_LSA,
      log_per_capita = log(per_capita)
    ) %>%
    filter(is.finite(log_per_capita))
  
  if (nrow(df_plot) > 0) {
    p1 <- ggplot(df_plot, aes(x = .data[[colname]])) +
      geom_histogram(bins = 30, fill = "skyblue", color = "black") +
      theme_minimal(base_size = 11) +
      labs(title = paste("Raw:", colname), x = colname, y = "Count")
    
    p2 <- ggplot(df_plot, aes(x = log_value)) +
      geom_histogram(bins = 30, fill = "salmon", color = "black") +
      theme_minimal(base_size = 11) +
      labs(title = paste("Log of", colname), x = paste0("log(", colname, ")"), y = "Count")
    
    p3 <- ggplot(df_plot, aes(x = per_capita)) +
      geom_histogram(bins = 30, fill = "lightgreen", color = "black") +
      theme_minimal(base_size = 11) +
      labs(title = paste(colname, "(per capita)"), x = paste0(colname, " / POPU_LSA"), y = "Count")
    
    p4 <- ggplot(df_plot, aes(x = log_per_capita)) +
      geom_histogram(bins = 30, fill = "orange", color = "black") +
      theme_minimal(base_size = 11) +
      labs(title = paste("Log of", colname, "(per capita)"), x = paste0("log(", colname, "/POPU_LSA)"), y = "Count")
    
    print(grid.arrange(
      p1, p2, p3, p4, ncol = 2,
      top = textGrob(
        paste("Distribution comparison for:", colname),
        gp = gpar(fontsize = 14, fontface = "bold")
      )
    ))
  }
}
