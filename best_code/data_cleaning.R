library(readr)
library(dplyr)
library(janitor)
library(tidyverse)

# ------------------ Cleaning Data ---------------------------

# Load data
df <- read_csv("raw_data/PLS_FY23_AE_pud23i.csv")

# Clean data
df_clean <- df %>%
  
  # Select relevant columns
  select(
    # Target
    visits = VISITS,
    
    # Categorical
    interlibrary_relation_code = C_RELATN,
    fscs_definition_code = C_FSCS,
    overdue_policy = ODFINE,
    beac_code = OBEREG,
    locale_code = LOCALE_ADD,
    
    # Continuous
    population_lsa = POPU_LSA,
    county_population = CNTYPOP,
    print_volumes = BKVOL,
    ebook_volumes = EBOOK,
    
    # Small continuous
    num_lib_branches = BRANLIB,
    num_bookmobiles = BKMOB,
    
    # Helper
    impute_status = RSTATUS
  ) %>%
  
  clean_names() %>%
  
  mutate(
    across(
      everything(),
      ~ case_when(
        . %in% c(-1, "-1", -3, "-3", -4, "-4", -9, "-9") ~ NA,
        TRUE ~ as.character(.)
      )
    )
  ) %>%
  
  mutate(
    across(
      c(
        visits, population_lsa, county_population, print_volumes,
        ebook_volumes, num_lib_branches, num_bookmobiles, impute_status
      ),
      ~ suppressWarnings(as.numeric(.))
    ),
    across(
      c(
        interlibrary_relation_code,
        fscs_definition_code,
        overdue_policy,
        beac_code,
        locale_code
      ),
      as.character
    )
  ) %>%
  filter(
    impute_status %in% c(1, 3),
    !is.na(visits),
    !is.na(population_lsa),
    visits != 0
  ) %>%
  mutate(
    beac_code = case_when(
      beac_code == "01" ~ "New England",
      beac_code == "02" ~ "Mid East",
      beac_code == "03" ~ "Great Lakes",
      beac_code == "04" ~ "Plains",
      beac_code == "05" ~ "Southeast",
      beac_code == "06" ~ "Southwest",
      beac_code == "07" ~ "Rocky Mountains",
      beac_code == "08" ~ "Far West",
      beac_code == "09" ~ "Outlying Areas",
      TRUE ~ NA
    ),
    locale_code = case_when(
      locale_code %in% c("11", "12", "13") ~ "City",
      locale_code %in% c("21", "22", "23") ~ "Suburb",
      locale_code %in% c("31", "32", "33") ~ "Town",
      locale_code %in% c("41", "42", "43") ~ "Rural",
      TRUE ~ NA
    ),
    interlibrary_relation_code = case_when(
      interlibrary_relation_code %in% c("HQ", "ME") ~ "Headquarters or Member of Federation or Cooperative",
      interlibrary_relation_code == "NO" ~ "Not a member of a federation or cooperative",
      TRUE ~ NA
    ),
    fscs_definition_code = case_when(
      fscs_definition_code == "Y" ~ "Meets FSCS Public Library Definition",
      fscs_definition_code == "N" ~ "Does not Meet FSCS Public Library Definition",
      TRUE ~ NA
    ),
    overdue_policy = case_when(
      overdue_policy == "Y" ~ "Has Overdue Policy",
      overdue_policy == "N" ~ "Does not Have Overdue Policy",
      TRUE ~ NA
    )
  ) %>%
  drop_na() %>%
  select(-impute_status)


write_csv(df_clean, "cleaned_data/cleaned_data.csv")