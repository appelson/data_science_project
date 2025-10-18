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
#   Create Per-LSA Dataset
# ============================================================
df_clean_per_lsa <- df_clean %>%
  filter(!is.na(population_of_lsa), population_of_lsa > 0) %>%
  mutate(
    # ---- Target ----
    visits_per_lsa = visits / population_of_lsa,
    
    # ---- Collections ----
    print_per_lsa = print_materials / population_of_lsa,
    ebook_per_lsa = electronic_books_e_books / population_of_lsa,
    audio_physical_per_lsa = audio_physical_units / population_of_lsa,
    audio_downloadable_per_lsa = audio_downloadable_units / population_of_lsa,
    video_physical_per_lsa = video_physical_units / population_of_lsa,
    video_downloadable_per_lsa = video_downloadable_units / population_of_lsa,
    eleccoll_per_lsa = total_electronic_collections / population_of_lsa,
    
    # ---- Staffing ----
    ala_mls_librarians_per_lsa = ala_mls_librarians / population_of_lsa,
    fte_librarians_per_lsa = total_number_of_fte_employees / population_of_lsa,
    other_staff_per_lsa = all_other_paid_fte_employees / population_of_lsa,
    
    # ---- Funding & Expenditures ----
    staff_expenditure_per_lsa = total_staff_expenditures / population_of_lsa,
    collection_expenditure_per_lsa = total_expenditures_on_library_collection / population_of_lsa,
    # capital_revenue_per_lsa = total_capital_revenue / population_of_lsa,
    # local_revenue_per_lsa = operating_revenue_from_local_government / population_of_lsa,
    # state_revenue_per_lsa = operating_revenue_from_state_government / population_of_lsa,
    # federal_revenue_per_lsa = operating_revenue_from_federal_government / population_of_lsa,
    # other_revenue_per_lsa = other_operating_revenue / population_of_lsa
  ) %>%
  select(
    -visits, -print_materials, -electronic_books_e_books,
    -audio_physical_units, -audio_downloadable_units,
    -video_physical_units, -video_downloadable_units,
    -total_electronic_collections,
    -ala_mls_librarians, -total_number_of_fte_employees,
    -all_other_paid_fte_employees,
    -total_staff_expenditures, -total_expenditures_on_library_collection,
    -operating_revenue_from_local_government, -operating_revenue_from_state_government,
    -operating_revenue_from_federal_government, -other_operating_revenue,
    -total_capital_revenue,
    -population_of_lsa
  ) %>%
  mutate(across(where(is.character), as.factor))

# ============================================================
#   Modeling — Log(Visits per LSA)
# ============================================================
model_lsa <- lm(
  visits_per_lsa ~ .,
  data = df_clean_per_lsa
)

summary(model_lsa)

# ============================================================
#   Plot: Actual vs Predicted (Per LSA)
# ============================================================
df_clean_per_lsa <- df_clean_per_lsa %>%
  mutate(
    pred = predict(model_lsa, newdata = df_clean_per_lsa),
    pred_visits_per_lsa = pred
  )

ggplot(df_clean_per_lsa, aes(x = visits_per_lsa, y = pred_visits_per_lsa)) +
  geom_point(alpha = 0.5, color = "darkorange") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "Actual vs Predicted Visits per LSA",
    x = "Actual Visits per LSA",
    y = "Predicted Visits per LSA"
  ) +
  theme_minimal()
