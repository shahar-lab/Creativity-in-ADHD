library(readr)
library(dplyr)

# load cleaned task data and participant data
cfg_all <- read_csv("Data/task/cfg_all_ready_for_merge.csv")
participant_data <- read_csv("Data/lab_screening/df_for_creativity_all.csv")

# identify participants without valid group assignment
missing_group <- participant_data %>%
  filter(
    is.na(declared_group) |
      !declared_group %in% c("ADHD", "TD")
  )

nrow(missing_group)

missing_group %>%
  select(subjectid, declared_group)

# exclude participants without valid group assignment
participant_data <- participant_data %>%
  filter(declared_group %in% c("ADHD", "TD"))

# check dataset sizes
nrow(cfg_all)
nrow(participant_data)

# check for duplicate IDs in task data
cfg_all %>%
  count(ID) %>%
  filter(n > 1)

# find participants who appear in participant_data but not in cfg_all
participant_only <- participant_data %>%
  distinct(subjectid, .keep_all = TRUE) %>%
  anti_join(
    cfg_all %>% distinct(ID, .keep_all = TRUE),
    by = c("subjectid" = "ID")
  )

# find participants who appear in cfg_all but not in participant_data
cfg_only <- cfg_all %>%
  distinct(ID, .keep_all = TRUE) %>%
  anti_join(
    participant_data %>% distinct(subjectid, .keep_all = TRUE),
    by = c("ID" = "subjectid")
  )

# check mismatch counts
nrow(participant_only)
nrow(cfg_only)

# inspect mismatch tables if needed
View(participant_only)
View(cfg_only)

# check task exclusions for participants missing from cfg_all
exclusions <- read_csv("Data/task/exclusions.csv")

participant_excluded <- participant_only %>%
  left_join(exclusions, by = c("subjectid" = "ID"))

# check how many participant-only cases are explained by exclusions
sum(!is.na(participant_excluded$reason))
sum(is.na(participant_excluded$reason))

# inspect participant-only cases with exclusion reasons
participant_excluded %>%
  select(subjectid, declared_group, cohort, reason) %>%
  View()

# merge task data with participant data for matched IDs only
df_creativity_ADHD <- cfg_all %>%
  inner_join(participant_data, by = c("ID" = "subjectid"))

# load study location data
round1_location <- read_csv("Data/task/Study_Location/round1_location.csv") %>%
  select(`Shahar ID`, Study_Location)

round2_location <- read_csv("Data/task/Study_Location/round2_location.csv") %>%
  select(`Shahar ID`, Study_Location)

# combine rounds
study_location <- bind_rows(
  round1_location,
  round2_location
) %>%
  rename(ID = `Shahar ID`)

# add study location to main dataset
df_creativity_ADHD <- df_creativity_ADHD %>%
  left_join(study_location, by = "ID")

# check study location merge
nrow(df_creativity_ADHD)

table(df_creativity_ADHD$Study_Location, useNA = "ifany")


# create cleaner ADHD subtype variable
df_creativity_ADHD <- df_creativity_ADHD %>%
  mutate(
    ADHD_subtype = case_when(
      is.na(diva_diagnosis_type) ~ "none",
      diva_diagnosis_type == "below_diva_criteria" ~ "none",
      diva_diagnosis_type == "primary_hyperactive/impulsive" ~ "combined",
      diva_diagnosis_type == "combined" ~ "combined",
      diva_diagnosis_type == "primary_inattentive" ~ "inattentive"
    )
  )

# save final merged dataset with study location
write_csv(df_creativity_ADHD, "Data/df_creativity_ADHD.csv")

