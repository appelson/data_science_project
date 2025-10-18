library(tidyverse)
library(janitor)
library(corrplot)

df <- read_csv("PLS_FY2022 PUD_CSV/PLS_FY22_AE_pud22i.csv")

df_clean <- df %>%
  select(
    "Visits" = VISITS,
    "Interlibrary Relationship Code" = C_RELATN,
    "Legal Basis Code" = C_LEGBAS,
    "Administrative Structure Code" = C_ADMIN,
    "FSCS Public Library Definition" = C_FSCS,
    "Geographic Code" = GEOCODE,
    "Population of the Legal Service Area" = POPU_LSA,
   # "ALA-MLS Librarians" = MASTER,
   # "Total number of FTE employees" = LIBRARIA,
   # "All other paid FTE employees" = OTHPAID,
   # "Operating revenue from local government" = LOCGVT,
   # "Operating revenue from state government" = STGVT,
   # "Operating revenue from federal government" = FEDGVT,
   # "Other operating revenue" = OTHINCM,
   # "Total staff expenditures" = STAFFEXP,
   # "Total expenditures on library collection" = TOTEXPCO,
    "Total capital revenue" = CAP_REV,
    "Print materials" = BKVOL,
    "Electronic Books (E-books)" = EBOOK,
    "Audio - physical units" = AUDIO_PH,
    "Audio - downloadable units" = AUDIO_DL,
    "Video - physical units" = VIDEO_PH,
    "Video - downloadable units" = VIDEO_DL,
    "Total electronic collections" = ELECCOLL,
   # "Total annual public service hours for all service outlets" = HRS_OPEN,
   # "Total number of synchronous program sessions" = TOTPRO,
    "Bureau of Economic Analysis Code" = OBEREG,
   # "Type of Census Geography Aligned with Legal Service Area" = LSAGEOTYPE,
   # "County Population" = CNTYPOP,
   # "Core based statistical area" = CBSA,
    "Metropolitan and Micropolitan Statistical Area" = MICROF,
    "Impute Status" = RSTATUS
  ) %>%
  clean_names() %>%
  mutate(
    across(everything(), ~ ifelse(. == -1, NA, .)),
    across(everything(), ~ ifelse(. == -4, NA, .)),
    across(everything(), ~ ifelse(. == -3, "Missing", .)),
    across(everything(), ~ ifelse(. == -9, "Suppressed for Confidentiality", .))
  ) %>%
  # Keep impute_status 1 or 3 and remove any "Missing" or "Suppressed..."
  filter(
    impute_status %in% c(1, 3)
  ) %>%
  filter(
    if_all(everything(), ~ !.x %in% c("Missing", "Suppressed for Confidentiality"))
  ) %>%
  filter(
    !is.na(visits),
    visits != 0
  ) %>%
  select(-impute_status) %>%
  mutate(
    across(
      where(~ all(!is.na(suppressWarnings(as.numeric(na.omit(as.character(.))))))),
      ~ as.numeric(.)
    )
  )

# ---- Create Per-Capita Dataset ----
df_clean_per_capita <- df_clean %>%
  filter(!is.na(population_of_the_legal_service_area), population_of_the_legal_service_area > 0) %>%
  mutate(
    visits_per_capita_served = visits / population_of_the_legal_service_area
  ) %>%
  select(-visits) %>%
  mutate(across(where(is.character), as.factor))

# ============================================================
#   Modeling
# ============================================================

# ---- Model 1: Raw Visits ----
model_1 <- lm(visits ~ ., data = df_clean)
summary(model_1)

# ---- Model 2: Log(Visits per Capita) ----
model_2 <- lm(log(visits_per_capita_served) ~ ., data = df_clean_per_capita)
summary(model_2)

# ============================================================
#   Visualizations
# ============================================================

# ---- Visits per Capita Histogram ----
ggplot(df_clean_per_capita, aes(x = visits_per_capita_served)) +
  geom_histogram(fill = "steelblue", color = "orange", alpha = 0.8, bins = 30) +
  scale_x_log10() +
  labs(
    title = "Distribution of Visits per Capita Served (Log Scale)",
    x = "Visits per Capita Served",
    y = "Count"
  ) +
  theme_minimal()

# ---- Visits Histogram ----
ggplot(df_clean, aes(x = visits)) +
  geom_histogram(fill = "steelblue", color = "orange", alpha = 0.8, bins = 30) +
  scale_x_log10() +
  labs(
    title = "Distribution of Visits (Log Scale)",
    x = "Visits",
    y = "Count"
  ) +
  theme_minimal()







# ============================================================
#   Predicted vs. True — Model Evaluation
# ============================================================

# ---- Model 1: Visits ----
df_clean <- df_clean %>%
  mutate(pred_visits = predict(model_1, newdata = df_clean))

ggplot(df_clean, aes(x = visits, y = pred_visits)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "orange", linetype = "dashed", size = 1) +
  scale_x_log10() +
  scale_y_log10() +
  labs(
    title = "Predicted vs. Actual Visits",
    x = "Actual Visits",
    y = "Predicted Visits"
  ) +
  theme_minimal()

# ---- Model 2: Log(Visits per Capita Served) ----
df_clean_per_capita <- df_clean_per_capita %>%
  mutate(pred_log_vpc = predict(model_2, newdata = df_clean_per_capita),
         pred_vpc = exp(pred_log_vpc))

ggplot(df_clean_per_capita, aes(x = visits_per_capita_served, y = pred_vpc)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "orange", linetype = "dashed", size = 1) +
  scale_x_log10() +
  scale_y_log10() +
  labs(
    title = "Predicted vs. Actual Visits per Capita Served",
    x = "Actual Visits per Capita",
    y = "Predicted Visits per Capita"
  ) +
  theme_minimal()

