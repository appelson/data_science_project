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
    population_lsa = POPU_LSA,
    
    # Categorical
    geocode = GEOCODE,
    interlibrary_relation_code = C_RELATN,
    legal_basis_code = C_LEGBAS,
    admin_structure_code = C_ADMIN,
    fscs_definition_code = C_FSCS,
    overdue_policy = ODFINE,
    beac_code = OBEREG,
    geo_type = LSAGEOTYPE,
    locale_code = LOCALE_ADD,
    metro = MICROF,
    
    # Continuous
    county_population = CNTYPOP,
    print_volumes = BKVOL,
    ebook_volumes = EBOOK,
    audio_physical = AUDIO_PH,
    audio_digital = AUDIO_DL,
    video_physical = VIDEO_PH,
    video_digital = VIDEO_DL,
    total_e_collection = ELECCOLL,
    tot_physical = TOTPHYS,
    other_physical = OTHPHYS,
    
    # Staff
    mls_librarians_fte = MASTER,
    total_staff_fte = LIBRARIA,
    other_paid_fte = OTHPAID,
    
    # Small continuous
    num_central_lib = CENTLIB,
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
        . %in% c(-1, "-1") ~ NA,
        . %in% c(-3, "-3") ~ NA,
        . %in% c(-4, "-4") ~ NA,
        . %in% c(-9, "-9") ~ NA,
        TRUE ~ as.character(.)
      )
    )
  ) %>%
  mutate(
    across(
      c(
        visits, population_lsa, county_population, print_volumes, ebook_volumes,
        audio_physical, audio_digital, video_physical, video_digital,
        total_e_collection, tot_physical, other_physical,
        mls_librarians_fte, total_staff_fte, other_paid_fte,
        num_central_lib, num_lib_branches, num_bookmobiles, impute_status
      ),
      ~ suppressWarnings(as.numeric(.))
    ),
    across(
      c(
        geocode, interlibrary_relation_code, legal_basis_code, admin_structure_code,
        fscs_definition_code, overdue_policy, beac_code, geo_type, locale_code, metro
      ),
      as.character
    )
  ) %>%
  mutate(visits_per_capita = visits / population_lsa,
         
         # Arbitrary threshold
         visits_per_capita_binary = ifelse(visits_per_capita>3,1,0)) %>%
    filter(
    impute_status %in% c(1, 3),
    !is.na(visits),
    !is.na(population_lsa),
    visits != 0
  ) %>%
  drop_na() %>%
  select(-c(impute_status))

write_csv(df_clean, "cleaned_data/cleaned_data.csv")

df_clean_per_capita <- df_clean %>%
  mutate(
    print_volumes_pc = print_volumes / population_lsa,
    ebook_volumes_pc = ebook_volumes / population_lsa,
    audio_physical_pc = audio_physical / population_lsa,
    audio_digital_pc = audio_digital / population_lsa,
    video_physical_pc = video_physical / population_lsa,
    video_digital_pc = video_digital / population_lsa,
    total_e_collection_pc = total_e_collection / population_lsa,
    tot_physical_pc = tot_physical / population_lsa,
    other_physical_pc = other_physical / population_lsa,
    
    mls_librarians_fte_pc = mls_librarians_fte / population_lsa,
    total_staff_fte_pc = total_staff_fte / population_lsa,
    other_paid_fte_pc = other_paid_fte / population_lsa
    
  ) %>%
  select(
    -c(
      print_volumes, ebook_volumes, audio_physical, audio_digital,
      video_physical, video_digital, total_e_collection, tot_physical, other_physical,
      mls_librarians_fte, total_staff_fte, other_paid_fte,
    )
  )

write_csv(df_clean_per_capita, "cleaned_data_per_capita.csv")