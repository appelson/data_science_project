df_clean <- read_csv("cleaned_data/cleaned_data.csv")
df_clean_per_capita <- read_csv("cleaned_data/cleaned_data_per_capita.csv")

# Linear model
linear_model <- lm(
  visits_per_capita ~ . - population_lsa, 
  data = df_clean
)

# Linear model per capita
linear_model_per_capita <- lm(
  visits_per_capita ~ . - population_lsa, 
  data = df_clean_per_capita
)

# Log linear model
log_linear_model <- lm(
  log(visits_per_capita) ~ 
    log(population_lsa + 1) +
    log(county_population + 1) +
    log(print_volumes + 1) +
    log(ebook_volumes + 1) +
    log(audio_physical + 1) +
    log(audio_digital + 1) +
    log(video_physical + 1) +
    log(video_digital + 1) +
    log(total_e_collection + 1) +
    log(tot_physical + 1) +
    log(other_physical + 1) +
    log(mls_librarians_fte + 1) +
    log(total_staff_fte + 1) +
    log(other_paid_fte + 1) +
    log(num_central_lib + 1) +
    log(num_lib_branches + 1) +
    log(num_bookmobiles + 1) +
    geocode +
    interlibrary_relation_code +
    legal_basis_code +
    admin_structure_code +
    fscs_definition_code +
    overdue_policy +
    beac_code +
    geo_type +
    locale_code +
    metro,
  data = df_clean
)

# Log linear model per capita
log_linear_model_per_capita <- lm(
  log(visits_per_capita) ~ 
    log(print_volumes_pc + 1) +
    log(ebook_volumes_pc + 1) +
    log(audio_physical_pc + 1) +
    log(audio_digital_pc + 1) +
    log(video_physical_pc + 1) +
    log(video_digital_pc + 1) +
    log(total_e_collection_pc + 1) +
    log(tot_physical_pc + 1) +
    log(other_physical_pc + 1) +
    log(mls_librarians_fte_pc + 1) +
    log(total_staff_fte_pc + 1) +
    log(other_paid_fte_pc + 1) +
    log(num_central_lib + 1) +
    log(num_lib_branches + 1) +
    log(num_bookmobiles + 1) +
    log(county_population + 1) +
    log(population_lsa + 1) +
    geocode +
    interlibrary_relation_code +
    legal_basis_code +
    admin_structure_code +
    fscs_definition_code +
    overdue_policy +
    beac_code +
    geo_type +
    locale_code +
    metro,
  data = df_clean_per_capita
)

# Outcome
summary(linear_model)
summary(linear_model_per_capita)
summary(log_linear_model)
summary(log_linear_model_per_capita)

# Logistic model
logistic_model <- glm(
  visits_per_capita_binary ~ 
    population_lsa +
    county_population +
    print_volumes +
    ebook_volumes +
    audio_physical +
    audio_digital +
    video_physical +
    video_digital +
    total_e_collection +
    tot_physical +
    other_physical +
    mls_librarians_fte +
    total_staff_fte +
    other_paid_fte +
    num_central_lib +
    num_lib_branches +
    num_bookmobiles +
    geocode +
    interlibrary_relation_code +
    legal_basis_code +
    admin_structure_code +
    fscs_definition_code +
    overdue_policy +
    beac_code +
    geo_type +
    locale_code +
    metro,
  data = df_clean,
  family = binomial
)

# Logistic per capita
logistic_model_per_capita <- glm(
  visits_per_capita_binary ~ 
    print_volumes_pc +
    ebook_volumes_pc +
    audio_physical_pc +
    audio_digital_pc +
    video_physical_pc +
    video_digital_pc +
    total_e_collection_pc +
    tot_physical_pc +
    other_physical_pc +
    mls_librarians_fte_pc +
    total_staff_fte_pc +
    other_paid_fte_pc +
    num_central_lib +
    num_lib_branches +
    num_bookmobiles +
    county_population +
    population_lsa +
    geocode +
    interlibrary_relation_code +
    legal_basis_code +
    admin_structure_code +
    fscs_definition_code +
    overdue_policy +
    beac_code +
    geo_type +
    locale_code +
    metro,
  data = df_clean_per_capita,
  family = binomial
)

# Logistic outcome
summary(logistic_model)
summary(logistic_model_per_capita)

df_clean$pred_logistic <- predict(logistic_model, type = "response")
df_clean_per_capita$pred_logistic_pc <- predict(logistic_model_per_capita, type = "response")

df_clean$pred_class <- ifelse(df_clean$pred_logistic > 0.5, 1, 0)
df_clean_per_capita$pred_class_pc <- ifelse(df_clean_per_capita$pred_logistic_pc > 0.5, 1, 0)

table(df_clean$visits_per_capita_binary, df_clean$pred_class)
table(df_clean_per_capita$visits_per_capita_binary, df_clean_per_capita$pred_class_pc)