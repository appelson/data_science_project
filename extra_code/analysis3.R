# ============================================================
#   Libraries
# ============================================================
library(tidyverse)
library(janitor)
library(corrplot)

# ============================================================
#   Load and Clean Data
# ============================================================
df <- read_csv("PLS_FY2022 PUD_CSV/PLS_FY22_AE_pud22i.csv")

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
      ~ as.numeric(.)
    )
  )

# ============================================================
#   Create Utilization Rate (Visits per LSA)
# ============================================================
df_utilization <- df_clean %>%
  filter(!is.na(population_of_lsa), population_of_lsa > 0) %>%
  mutate(
    utilization_rate = visits / population_of_lsa  # Target variable
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
  mutate(across(where(is.character), as.factor))

# ============================================================
#   Log Transform All Numeric Variables (+1 to avoid log(0))
# ============================================================
df_utilization_log <- df_utilization %>%
  mutate(across(where(is.numeric), ~ log(. + 1)))

# ============================================================
#   Modeling — Log(Utilization Rate)
# ============================================================
model_log_utilization <- lm(
  utilization_rate ~ .,
  data = df_utilization_log
)

summary(model_log_utilization)

# ============================================================
#   Plot: Actual vs Predicted (Log Scale)
# ============================================================
df_utilization_log <- df_utilization_log %>%
  mutate(pred = predict(model_log_utilization, newdata = df_utilization_log))

ggplot(df_utilization_log, aes(x = utilization_rate, y = pred)) +
  geom_point(alpha = 0.5, color = "darkorange") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "Actual vs Predicted (Log-Transformed Utilization Rate)",
    x = "Actual log(Utilization Rate)",
    y = "Predicted log(Utilization Rate)"
  ) +
  theme_minimal()
