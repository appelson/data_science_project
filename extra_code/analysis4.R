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
#   Create Per-LSA Dataset (predict Visits, not per capita)
# ============================================================
df_clean_per_lsa <- df_clean %>%
  filter(!is.na(population_of_lsa), population_of_lsa > 0) %>%
  mutate(
    # ---- Target (keep raw visits) ----
    visits = visits,
    
    # ---- Collections per LSA ----
    print_per_lsa = print_materials / population_of_lsa,
    ebook_per_lsa = electronic_books_e_books / population_of_lsa,
    audio_physical_per_lsa = audio_physical_units / population_of_lsa,
    audio_downloadable_per_lsa = audio_downloadable_units / population_of_lsa,
    video_physical_per_lsa = video_physical_units / population_of_lsa,
    video_downloadable_per_lsa = video_downloadable_units / population_of_lsa,
    eleccoll_per_lsa = total_electronic_collections / population_of_lsa,
    
    # ---- Staffing per LSA ----
    ala_mls_librarians_per_lsa = ala_mls_librarians / population_of_lsa,
    fte_librarians_per_lsa = total_number_of_fte_employees / population_of_lsa,
    other_staff_per_lsa = all_other_paid_fte_employees / population_of_lsa,
    
    # ---- Funding & Expenditures per LSA ----
    staff_expenditure_per_lsa = total_staff_expenditures / population_of_lsa,
    collection_expenditure_per_lsa = total_expenditures_on_library_collection / population_of_lsa
  ) %>%
  select(
    visits,
    interlibrary_relationship_code,
    legal_basis_code,
    administrative_structure_code,
    fscs_public_library_definition,
    geographic_code,
    print_per_lsa, ebook_per_lsa,
    audio_physical_per_lsa, audio_downloadable_per_lsa,
    video_physical_per_lsa, video_downloadable_per_lsa,
    eleccoll_per_lsa,
    ala_mls_librarians_per_lsa, fte_librarians_per_lsa,
    other_staff_per_lsa,
    staff_expenditure_per_lsa, collection_expenditure_per_lsa
  ) %>%
  mutate(across(where(is.character), as.factor))

# ============================================================
#   Modeling — Visits (raw)
# ============================================================
model_visits <- lm(
  visits ~ .,
  data = df_clean_per_lsa
)

summary(model_visits)

# ============================================================
#   Plot: Actual vs Predicted (Visits)
# ============================================================
df_clean_per_lsa <- df_clean_per_lsa %>%
  mutate(
    pred = predict(model_visits, newdata = df_clean_per_lsa),
    pred_visits = pred
  )

ggplot(df_clean_per_lsa, aes(x = visits, y = pred_visits)) +
  geom_point(alpha = 0.5, color = "darkorange") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "Actual vs Predicted Visits",
    x = "Actual Visits",
    y = "Predicted Visits"
  ) +
  theme_minimal()
