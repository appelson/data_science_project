# Foundations of Data Science Project

## Data Source
This data comes from the [Public Libraries Survey (PLS) 2023](https://www.imls.gov/research-evaluation/surveys/public-libraries-survey-pls) survey data of 9,252 libraries across the United States. Specifically, we examine "PLS_FY23_AE_pud23i.csv".

## Data Stakeholder
Our stakeholder is the Director of Capital Planning for at the Institute of Museum and Library Services or another large Federal body who needs to assess whether or not to provide a loan to a library based on their ability to maximize the number of annual visits per capita. This director is responsible for a multi-million dollar annual budget allocated for building new library branches and renovating existing ones. Their primary challenge is to make high-stakes investment decisions that maximize community impact. Before committing years of planning and significant public funds to a project, they need a reliable way to forecast a potential branch's success. They would commission this analysis to develop a predictive tool that can forecast public engagement, ensuring that resources are allocated to projects with the highest potential return on investment for the community.

## Variable of Prediction
We will be predicting the annual visits per capita served.

## Variables in Covariate
The covariates include:

## Variables in Covariate

The covariates include (for more information, see: https://www.imls.gov/sites/default/files/2025-08/PLS-FY-2023-Data-Documentation-508.pdf)

### Categorical Variables
- `geocode` = `GEOCODE`
- `interlibrary_relation_code` = `C_RELATN`
- `legal_basis_code` = `C_LEGBAS`
- `admin_structure_code` = `C_ADMIN`
- `fscs_definition_code` = `C_FSCS`
- `overdue_policy` = `ODFINE`
- `beac_code` = `OBEREG`
- `geo_type` = `LSAGEOTYPE`
- `locale_code` = `LOCALE_ADD`
- `metro` = `MICROF`

### Continuous Variables
- `county_population` = `CNTYPOP`
- `print_volumes` = `BKVOL`
- `ebook_volumes` = `EBOOK`
- `audio_physical` = `AUDIO_PH`
- `audio_digital` = `AUDIO_DL`
- `video_physical` = `VIDEO_PH`
- `video_digital` = `VIDEO_DL`
- `total_e_collection` = `ELECCOLL`
- `tot_physical` = `TOTPHYS`
- `other_physical` = `OTHPHYS`

### Staff Variables
- `mls_librarians_fte` = `MASTER`
- `total_staff_fte` = `LIBRARIA`
- `other_paid_fte` = `OTHPAID`

### Small Continuous Variables
- `num_central_lib` = `CENTLIB`
- `num_lib_branches` = `BRANLIB`
- `num_bookmobiles` = `BKMOB`
